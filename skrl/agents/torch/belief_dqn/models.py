from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from skrl.models.torch import Model, DeterministicMixin


class ResNetBlock(nn.Module):
    """Two dense layers with ELU activations and a residual connection.

    Matches paper §V-A: each block is Dense(width) → ELU → Dense(width) → ELU
    with input added back (residual).
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)
        self.elu = nn.ELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.elu(self.fc2(self.elu(self.fc1(x))) + x)


class ResNetDuelingQNetwork(DeterministicMixin, Model):
    """Belief-conditioned Q-network from paper §V-A.

    Architecture (exact):
        Dense(64) → 4 × ResNetBlock(64) → Dueling head (V + A − mean A)

    Input:  compressed belief b̃, shape ``[batch, belief_dim]``
    Output: Q-values for each of NC discrete actions, shape ``[batch, NC]``

    Dueling head (Wang et al. 2016):
        V(b̃):     Linear(64, 1)
        A(b̃, a):  Linear(64, NC)
        Q(b̃, a) = V(b̃) + A(b̃, a) − mean_a A(b̃, a)

    :param observation_space: skrl observation space (belief_dim).
    :param action_space: skrl action space (Discrete(NC)).
    :param device: Torch device.
    :param resnet_blocks: Number of ResNet blocks.  Paper: 4.
    :param width: Hidden width.  Paper: 64.
    :param dueling: Use dueling head.  Always True per paper.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        device: str | torch.device | None = None,
        *,
        resnet_blocks: int = 4,
        width: int = 64,
        dueling: bool = True,
    ) -> None:
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=False)

        if hasattr(action_space, "n"):
            n_actions = int(action_space.n)
        else:
            raise ValueError("BeliefDQN requires a Discrete action space.")

        try:
            belief_dim = int(observation_space.shape[0])
        except Exception:
            belief_dim = int(observation_space)

        self._dueling = dueling

        # trunk
        self.input_layer = nn.Linear(belief_dim, width)
        self.resnet = nn.Sequential(*[ResNetBlock(width) for _ in range(resnet_blocks)])
        self.elu = nn.ELU()

        if dueling:
            self.value_head = nn.Linear(width, 1)
            self.advantage_head = nn.Linear(width, n_actions)
        else:
            self.q_head = nn.Linear(width, n_actions)

    def compute(
        self, inputs: dict[str, Any], role: str = ""
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Forward pass returning Q-values.

        :param inputs: Dict with key ``"observations"`` (compressed belief b̃).
        :param role: Model role string (unused here).
        :return: Q-values ``[batch, NC]`` and empty extras dict.
        """
        x = inputs["observations"]
        x = self.elu(self.input_layer(x))
        x = self.resnet(x)

        if self._dueling:
            v = self.value_head(x)                # [B, 1]
            a = self.advantage_head(x)            # [B, NC]
            q = v + a - a.mean(dim=-1, keepdim=True)
        else:
            q = self.q_head(x)

        return q, {}
