from __future__ import annotations

import dataclasses

import torch
import torch.nn as nn

from skrl.models.torch import CategoricalMixin, DeterministicMixin, GaussianMixin, Model


@dataclasses.dataclass(frozen=True)
class ObsLayout:
    """Slices of the flat observation produced by ``PrivilegedHistoryWrapper``:
    ``[o_t (D) | history (H*D) | privileged (P) | mask (1)]``."""

    obs_dim: int
    history_length: int
    privileged_dim: int

    @property
    def size(self) -> int:
        return self.obs_dim * (1 + self.history_length) + self.privileged_dim + 1

    def current(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, : self.obs_dim]

    def with_history(self, x: torch.Tensor) -> torch.Tensor:
        """``o_t`` and its history, the estimator input."""
        return x[:, : self.obs_dim * (1 + self.history_length)]

    def sequence(self, x: torch.Tensor) -> torch.Tensor:
        """History and ``o_t`` as a time sequence ``[B, 1 + H, D]``, oldest first (``o_{t-H} .. o_t``)."""
        d = self.obs_dim
        history = x[:, d : d * (1 + self.history_length)].reshape(-1, self.history_length, d)
        return torch.cat([history, x[:, None, :d]], dim=1)

    def privileged(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, -self.privileged_dim - 1 : -1]

    def mask(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, -1:]


def _mlp(sizes: list[int], activation=nn.ELU) -> nn.Sequential:
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), activation()]
    return nn.Sequential(*layers[:-1])


class _Conv1dEstimator(nn.Module):
    """Temporal convolutions over the history (as the RMA adaptation module), then a linear head."""

    def __init__(self, layout: ObsLayout, hidden: int, latent_dim: int) -> None:
        super().__init__()
        self.layout = layout
        self.net = nn.Sequential(
            nn.Conv1d(layout.obs_dim, hidden, 3, padding=1), nn.ELU(),
            nn.Conv1d(hidden, hidden, 3, stride=2, padding=1), nn.ELU(),
            nn.Conv1d(hidden, hidden, 3, stride=2, padding=1), nn.ELU(),
            nn.Flatten(),
        )
        length = self.net(torch.zeros(1, layout.obs_dim, 1 + layout.history_length)).shape[-1]
        self.head = nn.Linear(length, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(self.layout.sequence(x).transpose(1, 2)))


class _LSTMEstimator(nn.Module):
    """LSTM over the history window (reset every step, so PPO needs no sequence batching); last hidden state."""

    def __init__(self, layout: ObsLayout, hidden: int, latent_dim: int) -> None:
        super().__init__()
        self.layout = layout
        self.inp = nn.Linear(layout.obs_dim, hidden)
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(self.inp(self.layout.sequence(x)))
        return self.head(out[:, -1])


class _CausalTransformerEstimator(nn.Module):
    """Causal transformer with one token per time step; the latent is read from the newest token (``o_t``)."""

    def __init__(self, layout: ObsLayout, hidden: int, latent_dim: int, layers: int = 2, heads: int = 4) -> None:
        super().__init__()
        self.layout = layout
        steps = 1 + layout.history_length
        self.inp = nn.Linear(layout.obs_dim, hidden)
        self.position = nn.Parameter(torch.zeros(1, steps, hidden))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            hidden, heads, dim_feedforward=2 * hidden, dropout=0.0, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, latent_dim)
        self.register_buffer("causal_mask", nn.Transformer.generate_square_subsequent_mask(steps), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.inp(self.layout.sequence(x)) + self.position
        out = self.transformer(tokens, mask=self.causal_mask, is_causal=True)
        return self.head(self.norm(out[:, -1]))


ESTIMATORS = ("mlp", "conv1d", "lstm", "transformer")


def make_estimator(arch: str, layout: ObsLayout, hidden: int, latent_dim: int) -> nn.Module:
    """History estimator ``φ``: ``mlp`` on the flattened ``[o_t, history]``, or a temporal model that takes the full
    observation and reads the time sequence itself."""
    if arch not in ESTIMATORS:
        raise ValueError(f"unknown estimator '{arch}', expected one of {ESTIMATORS}")
    if arch == "mlp":
        return _mlp([layout.obs_dim * (1 + layout.history_length), hidden, hidden, latent_dim])
    if layout.history_length < 1:
        raise ValueError(f"estimator '{arch}' needs a history (history_length >= 1)")
    return {"conv1d": _Conv1dEstimator, "lstm": _LSTMEstimator, "transformer": _CausalTransformerEstimator}[arch](
        layout, hidden, latent_dim
    )


class PrivilegedActorTrunk(nn.Module):
    """Encoder (privileged → z), estimator (history → ẑ) and the actor MLP on ``[o_t, z or ẑ]``.

    Episodes with mask = 0 act on the encoder latent z, masked episodes act on the estimator
    latent ẑ. ``PrivilegedPPO`` adds the regularization between z and ẑ.
    """

    def __init__(
        self,
        layout: ObsLayout,
        *,
        latent_dim: int = 8,
        hidden: int = 64,
        estimator: str = "mlp",
        sysid_head: bool = False,
    ) -> None:
        super().__init__()
        self.layout = layout
        self.encoder = _mlp([layout.privileged_dim, hidden, latent_dim])
        self.estimator = make_estimator(estimator, layout, hidden, latent_dim)
        self._flat_history = estimator == "mlp"
        self.actor = _mlp([layout.obs_dim + latent_dim, hidden, hidden])
        self.output_dim = hidden
        # decoder ψ: latent → standardized privileged values. ψ(z) makes (μ, ψ) an autoencoder of the privileged
        # values, ψ(ẑ) a supervised system-identification head on the history (trained by PrivilegedPPO)
        self.sysid = _mlp([latent_dim, hidden, layout.privileged_dim]) if sysid_head else None

    def latents(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(z, ẑ)``: encoder latent of the privileged values and estimator latent of the history."""
        z_hat = self.estimator(self.layout.with_history(x) if self._flat_history else x)
        return self.encoder(self.layout.privileged(x)), z_hat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, z_hat = self.latents(x)
        masked = self.layout.mask(x) > 0.5
        latent = torch.where(masked, z_hat, z)
        return nn.functional.elu(self.actor(torch.cat([self.layout.current(x), latent], dim=-1)))


class PrivilegedCategoricalPolicy(CategoricalMixin, Model):
    """Discrete-action actor on top of :class:`PrivilegedActorTrunk`."""

    def __init__(
        self, observation_space, action_space, device=None, *, layout: ObsLayout, latent_dim=8, hidden=64, **trunk_kwargs
    ):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        CategoricalMixin.__init__(self, unnormalized_log_prob=True)
        self.trunk = PrivilegedActorTrunk(layout, latent_dim=latent_dim, hidden=hidden, **trunk_kwargs)
        self.head = nn.Linear(hidden, self.num_actions)

    def compute(self, inputs, role=""):
        return self.head(self.trunk(inputs["observations"])), {}


class PrivilegedGaussianPolicy(GaussianMixin, Model):
    """Continuous-action actor on top of :class:`PrivilegedActorTrunk`."""

    def __init__(
        self, observation_space, action_space, device=None, *, layout: ObsLayout, latent_dim=8, hidden=64, **trunk_kwargs
    ):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        GaussianMixin.__init__(self, clip_actions=False, clip_log_std=True, min_log_std=-20.0, max_log_std=2.0)
        self.trunk = PrivilegedActorTrunk(layout, latent_dim=latent_dim, hidden=hidden, **trunk_kwargs)
        self.head = nn.Linear(hidden, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role=""):
        return self.head(self.trunk(inputs["observations"])), {"log_std": self.log_std_parameter}


class PrivilegedValue(DeterministicMixin, Model):
    """Asymmetric critic: always sees ``o_t`` and the privileged values, whatever the mask."""

    def __init__(self, observation_space, action_space, device=None, *, layout: ObsLayout, hidden=64):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self)
        self.layout = layout
        self.net = _mlp([layout.obs_dim + layout.privileged_dim, hidden, hidden, 1])

    def compute(self, inputs, role=""):
        x = inputs["observations"]
        return self.net(torch.cat([self.layout.current(x), self.layout.privileged(x)], dim=-1)), {}
