from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from skrl.agents.torch.belief_dqn.models import ResNetBlock
from skrl.models.torch import CategoricalMixin, DeterministicMixin, Model


class _ResNetTrunk(nn.Module):
    """Dense(width) → ELU → ``resnet_blocks`` × ResNetBlock(width), same trunk as BeliefDQN (paper §V-A)."""

    def __init__(self, belief_dim: int, width: int, resnet_blocks: int) -> None:
        super().__init__()
        self.input_layer = nn.Linear(belief_dim, width)
        self.resnet = nn.Sequential(*[ResNetBlock(width) for _ in range(resnet_blocks)])
        self.elu = nn.ELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(self.elu(self.input_layer(x)))


class ResNetCategoricalPolicy(CategoricalMixin, Model):
    """π(a | b̃): ResNet trunk → logits over NC discrete controllers."""

    def __init__(
        self,
        observation_space,
        action_space,
        device: str | torch.device | None = None,
        *,
        resnet_blocks: int = 4,
        width: int = 64,
    ) -> None:
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        CategoricalMixin.__init__(self, unnormalized_log_prob=True)

        self.trunk = _ResNetTrunk(self.num_observations, width, resnet_blocks)
        self.logits = nn.Linear(width, self.num_actions)

    def compute(self, inputs: dict[str, Any], role: str = "") -> tuple[torch.Tensor, dict[str, Any]]:
        return self.logits(self.trunk(inputs["observations"])), {}


class ResNetValue(DeterministicMixin, Model):
    """V(b̃): ResNet trunk → scalar value."""

    def __init__(
        self,
        observation_space,
        action_space,
        device: str | torch.device | None = None,
        *,
        resnet_blocks: int = 4,
        width: int = 64,
    ) -> None:
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=False)

        self.trunk = _ResNetTrunk(self.num_observations, width, resnet_blocks)
        self.head = nn.Linear(width, 1)

    def compute(self, inputs: dict[str, Any], role: str = "") -> tuple[torch.Tensor, dict[str, Any]]:
        return self.head(self.trunk(inputs["observations"])), {}
