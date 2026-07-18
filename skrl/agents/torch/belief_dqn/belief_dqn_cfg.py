from __future__ import annotations

import dataclasses
from typing import Callable, Optional

from skrl.agents.torch import AgentCfg


@dataclasses.dataclass(kw_only=True)
class BeliefDQN_CFG(AgentCfg):
    """Configuration for the BeliefDQN agent.

    Hyperparameters match the paper (§V-A) unless noted otherwise.
    """

    # ------------------------------------------------------------------ belief
    belief_factory: Optional[Callable] = None
    """Callable() -> Belief instance.  Called once at agent init.
    Must return a configured ParticleFilter (or subclass) with the
    backend already injected.  Required — no default."""

    num_controllers: int = 20
    """NC — number of discrete low-level motion controllers.
    NC = n_target_poses * NG.  Paper: 20 for peg-in-hole."""

    # ------------------------------------------------------------------ GMM
    gmm_components: int = 2
    """NG — number of GMM mixture components.  Paper uses 2."""

    # ------------------------------------------------------------------ network
    resnet_blocks: int = 4
    """Number of ResNet blocks in the Q-network trunk.  Paper: 4."""

    resnet_width: int = 64
    """Hidden width per ResNet block.  Paper: 64."""

    dueling: bool = True
    """Use a dueling head (V + A - mean A).  Paper uses dueling."""

    double_q: bool = True
    """Use double Q-learning to reduce overestimation.  Paper uses double-Q."""

    # ------------------------------------------------------------------ training
    gradient_steps: int = 1
    """Gradient steps per environment step."""

    batch_size: int = 64
    """Replay buffer mini-batch size."""

    discount_factor: float = 0.98
    """Discount factor γ.  Paper: 0.98."""

    polyak: float = 0.005
    """Soft target-network update coefficient."""

    learning_rate: float = 1e-4
    """Adam learning rate.  Paper: 0.0001."""

    l2_reg: float = 0.01
    """L2 kernel regularisation weight (Adam weight_decay).  Paper: 0.01."""

    learning_rate_scheduler: Optional[type] = None
    """Optional LR scheduler class."""

    learning_rate_scheduler_kwargs: dict = dataclasses.field(default_factory=dict)

    # ------------------------------------------------------------------ exploration
    random_timesteps: int = 0
    """Pure random action steps before epsilon-greedy kicks in."""

    learning_starts: int = 0
    """Steps before the first gradient update."""

    update_interval: int = 1
    """Environment steps between gradient updates."""

    target_update_interval: int = 10
    """Gradient steps between soft target-network updates."""

    exploration_scheduler: Optional[Callable[[int, int], float]] = None
    """Callable(timestep, total_timesteps) -> epsilon.
    If None, greedy (epsilon = 0) after random_timesteps."""

    # ------------------------------------------------------------------ data
    observation_preprocessor: Optional[type] = None
    """Preprocessor for the compressed belief input b̃."""

    observation_preprocessor_kwargs: dict = dataclasses.field(default_factory=dict)

    rewards_shaper: Optional[Callable] = None
    """Optional reward shaping function."""

    # ------------------------------------------------------------------ misc
    mixed_precision: bool = False
    """Use automatic mixed precision (AMP) during training."""

    n_substeps: int = 100
    """Inner-loop substeps per high-level action (200 Hz / 2 Hz = 100)."""
