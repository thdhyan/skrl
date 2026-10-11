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

    def privileged(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, -self.privileged_dim - 1 : -1]

    def mask(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, -1:]


def _mlp(sizes: list[int], activation=nn.ELU) -> nn.Sequential:
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), activation()]
    return nn.Sequential(*layers[:-1])


class PrivilegedActorTrunk(nn.Module):
    """Encoder (privileged → z), estimator (history → ẑ) and the actor MLP on ``[o_t, z or ẑ]``.

    Episodes with mask = 0 act on the encoder latent z, masked episodes act on the estimator
    latent ẑ. ``PrivilegedPPO`` adds the regularization between z and ẑ.
    """

    def __init__(self, layout: ObsLayout, *, latent_dim: int = 8, hidden: int = 64) -> None:
        super().__init__()
        self.layout = layout
        self.encoder = _mlp([layout.privileged_dim, hidden, latent_dim])
        self.estimator = _mlp([layout.obs_dim * (1 + layout.history_length), hidden, hidden, latent_dim])
        self.actor = _mlp([layout.obs_dim + latent_dim, hidden, hidden])
        self.output_dim = hidden

    def latents(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(z, ẑ)``: encoder latent of the privileged values and estimator latent of the history."""
        return self.encoder(self.layout.privileged(x)), self.estimator(self.layout.with_history(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, z_hat = self.latents(x)
        masked = self.layout.mask(x) > 0.5
        latent = torch.where(masked, z_hat, z)
        return nn.functional.elu(self.actor(torch.cat([self.layout.current(x), latent], dim=-1)))


class PrivilegedCategoricalPolicy(CategoricalMixin, Model):
    """Discrete-action actor on top of :class:`PrivilegedActorTrunk`."""

    def __init__(self, observation_space, action_space, device=None, *, layout: ObsLayout, latent_dim=8, hidden=64):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        CategoricalMixin.__init__(self, unnormalized_log_prob=True)
        self.trunk = PrivilegedActorTrunk(layout, latent_dim=latent_dim, hidden=hidden)
        self.head = nn.Linear(hidden, self.num_actions)

    def compute(self, inputs, role=""):
        return self.head(self.trunk(inputs["observations"])), {}


class PrivilegedGaussianPolicy(GaussianMixin, Model):
    """Continuous-action actor on top of :class:`PrivilegedActorTrunk`."""

    def __init__(self, observation_space, action_space, device=None, *, layout: ObsLayout, latent_dim=8, hidden=64):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        GaussianMixin.__init__(self, clip_actions=False, clip_log_std=True, min_log_std=-20.0, max_log_std=2.0)
        self.trunk = PrivilegedActorTrunk(layout, latent_dim=latent_dim, hidden=hidden)
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
