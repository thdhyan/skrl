"""Small MLP actor/critic models used by the self-play runner."""

from __future__ import annotations

from typing import Sequence

import gymnasium

import torch
import torch.nn as nn

from skrl.models.torch import CategoricalMixin, DeterministicMixin, Model


def _mlp(inputs: int, hidden: Sequence[int], outputs: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    for size in hidden:
        layers += [nn.Linear(inputs, size), nn.ELU()]
        inputs = size
    layers.append(nn.Linear(inputs, outputs))
    return nn.Sequential(*layers)


class MLPCategoricalPolicy(CategoricalMixin, Model):
    def __init__(
        self,
        observation_space: gymnasium.Space,
        state_space: gymnasium.Space | None,
        action_space: gymnasium.Space,
        device: str | torch.device | None = None,
        hidden_sizes: Sequence[int] = (64, 64),
    ) -> None:
        """Categorical (discrete actions) MLP policy on the agent's observations."""
        Model.__init__(
            self, observation_space=observation_space, state_space=state_space, action_space=action_space, device=device
        )
        CategoricalMixin.__init__(self, unnormalized_log_prob=True)
        self.net = _mlp(self.num_observations, hidden_sizes, self.num_actions)

    def compute(self, inputs, role=""):
        return self.net(inputs["observations"]), {}


class MLPValue(DeterministicMixin, Model):
    def __init__(
        self,
        observation_space: gymnasium.Space,
        state_space: gymnasium.Space | None,
        action_space: gymnasium.Space,
        device: str | torch.device | None = None,
        hidden_sizes: Sequence[int] = (64, 64),
        use_states: bool = True,
    ) -> None:
        """MLP value function on the global state (MAPPO, centralized critic) or on the observations (IPPO)."""
        Model.__init__(
            self, observation_space=observation_space, state_space=state_space, action_space=action_space, device=device
        )
        DeterministicMixin.__init__(self)
        self.use_states = use_states and state_space is not None
        self.net = _mlp(self.num_states if self.use_states else self.num_observations, hidden_sizes, 1)

    def compute(self, inputs, role=""):
        return self.net(inputs["states"] if self.use_states else inputs["observations"]), {}
