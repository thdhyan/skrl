from __future__ import annotations

import dataclasses
from typing import Callable, Optional

from skrl.agents.torch import AgentCfg


@dataclasses.dataclass(kw_only=True)
class BeliefPPO_CFG(AgentCfg):
    """Configuration for the BeliefPPO agent.

    On-policy counterpart to BeliefDQN_CFG: same belief/network shape knobs,
    PPO's rollout/GAE/clip hyperparameters in place of replay/Q-learning ones.
    """

    # ------------------------------------------------------------------ belief
    belief_factory: Optional[Callable] = None
    """Callable() -> Belief instance.  Called once at agent init.
    Must return a configured ParticleFilter (or subclass) with the
    backend already injected.  Required — no default."""

    belief_prior_fn: Optional[Callable] = None
    """Callable(observation) -> initial particles ``[NP, state_dim]``.
    Called with the first raw observation of every episode (shape ``[obs_dim]``)
    to re-seed the filter.  ``None`` keeps the filter running across episodes."""

    num_controllers: int = 20
    """NC — number of discrete low-level motion controllers."""

    gmm_components: int = 2
    """NG — number of GMM mixture components."""

    # ------------------------------------------------------------------ network
    resnet_blocks: int = 4
    """Number of ResNet blocks in the policy/value trunk."""

    resnet_width: int = 64
    """Hidden width per ResNet block."""

    # ------------------------------------------------------------------ rollout / GAE
    rollouts: int = 16
    """Number of collection steps to perform between updates."""

    learning_epochs: int = 8
    """Number of learning epochs to perform during updates."""

    mini_batches: int = 2
    """Number of mini batches to sample when updating."""

    discount_factor: float = 0.98
    """Discount factor γ.  Matches BeliefDQN's paper value."""

    gae_lambda: float = 0.95
    """TD(lambda) coefficient for GAE."""

    # ------------------------------------------------------------------ optimization
    learning_rate: float = 1e-4
    """Adam learning rate for the shared policy/value optimizer."""

    learning_rate_scheduler: Optional[type] = None
    """Optional LR scheduler class."""

    learning_rate_scheduler_kwargs: dict = dataclasses.field(default_factory=dict)

    grad_norm_clip: float = 0.5
    """Clipping coefficient for gradients by global norm.  <=0 disables."""

    ratio_clip: float = 0.2
    """Clipping coefficient for the clipped surrogate objective."""

    value_clip: float = 0.2
    """Clipping coefficient for predicted value.  <=0 disables."""

    entropy_loss_scale: float = 0.0
    """Entropy loss scaling factor."""

    value_loss_scale: float = 2.5
    """Value loss scaling factor."""

    kl_threshold: float = 0.0
    """KL-divergence threshold for early stopping a mini-batch epoch."""

    # ------------------------------------------------------------------ exploration / scheduling
    random_timesteps: int = 0
    """Pure random action steps before sampling from the policy."""

    learning_starts: int = 0
    """Steps before the first update is allowed."""

    # ------------------------------------------------------------------ data
    observation_preprocessor: Optional[type] = None
    """Preprocessor for the compressed belief input b̃."""

    observation_preprocessor_kwargs: dict = dataclasses.field(default_factory=dict)

    value_preprocessor: Optional[type] = None
    """Preprocessor for the value network's output."""

    value_preprocessor_kwargs: dict = dataclasses.field(default_factory=dict)

    rewards_shaper: Optional[Callable] = None
    """Optional reward shaping function."""

    # ------------------------------------------------------------------ misc
    mixed_precision: bool = False
    """Use automatic mixed precision (AMP) during training."""

    n_substeps: int = 100
    """Inner-loop substeps per high-level action."""
