from __future__ import annotations

from typing import Any

import gymnasium
import torch
import torch.nn as nn

from skrl.models.torch.base import Model
from skrl.models.torch.deterministic import DeterministicMixin
from skrl.models.torch.gaussian import GaussianMixin


class ObsTokenizer(nn.Module):
    """Split a flat observation vector ``[B, D]`` into ``D`` tokens ``[B, D, d_model]``.

    Each scalar is embedded with a linear layer shared across dimensions, and a learned
    per-dimension embedding tells the transformer which observation the token encodes.
    """

    def __init__(self, obs_dim: int, d_model: int) -> None:
        super().__init__()
        self.value_proj = nn.Linear(1, d_model)
        self.dim_embedding = nn.Parameter(torch.randn(obs_dim, d_model) * 0.02)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.value_proj(obs.unsqueeze(-1)) + self.dim_embedding


class TransformerEncoderTrunk(nn.Module):
    """Observation tokenizer + learned [CLS] token + Transformer encoder. Returns the CLS embedding ``[B, d_model]``."""

    def __init__(
        self,
        obs_dim: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_ff: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.tokenizer = ObsTokenizer(obs_dim, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(obs)
        cls = self.cls_token.expand(obs.shape[0], -1, -1)
        x = self.encoder(torch.cat([cls, tokens], dim=1))
        return self.norm(x[:, 0])


class TransformerGaussianPolicy(GaussianMixin, Model):
    def __init__(
        self,
        observation_space: gymnasium.Space,
        action_space: gymnasium.Space,
        device: str | torch.device | None = None,
        *,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_ff: int = 128,
        dropout: float = 0.0,
        clip_actions: bool = False,
        clip_log_std: bool = True,
        min_log_std: float = -20,
        max_log_std: float = 2,
        reduction: str = "sum",
    ) -> None:
        """Gaussian policy: transformer trunk -> linear mean, state-independent log_std parameter."""
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        GaussianMixin.__init__(
            self,
            clip_actions=clip_actions,
            clip_log_std=clip_log_std,
            min_log_std=min_log_std,
            max_log_std=max_log_std,
            reduction=reduction,
        )
        self.trunk = TransformerEncoderTrunk(
            self.num_observations, d_model, n_heads, n_layers, dim_ff, dropout
        )
        self.mean_head = nn.Linear(d_model, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs: dict[str, Any], role: str = "") -> tuple[torch.Tensor, dict[str, Any]]:
        mean = self.mean_head(self.trunk(inputs["observations"]))
        return mean, {"log_std": self.log_std_parameter}


class TransformerValue(DeterministicMixin, Model):
    def __init__(
        self,
        observation_space: gymnasium.Space,
        action_space: gymnasium.Space,
        device: str | torch.device | None = None,
        *,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_ff: int = 128,
        dropout: float = 0.0,
    ) -> None:
        """State-value function: transformer trunk -> linear head (1 output)."""
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self)
        self.trunk = TransformerEncoderTrunk(
            self.num_observations, d_model, n_heads, n_layers, dim_ff, dropout
        )
        self.value_head = nn.Linear(d_model, 1)

    def compute(self, inputs: dict[str, Any], role: str = "") -> tuple[torch.Tensor, dict[str, Any]]:
        return self.value_head(self.trunk(inputs["observations"])), {}
