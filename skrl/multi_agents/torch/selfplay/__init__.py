"""Generation-based self-play (opponent pools, opponent samplers and runner) for two-team multi-agent games."""

from skrl.multi_agents.torch.selfplay.models import MLPCategoricalPolicy, MLPValue
from skrl.multi_agents.torch.selfplay.pool import (
    OpponentPool,
    PolicySnapshot,
    elo_expected,
    elo_update,
)
from skrl.multi_agents.torch.selfplay.samplers import (
    SAMPLERS,
    AdaptivePBTSampler,
    BestSampler,
    HedgeSampler,
    LastSampler,
    OpponentSampler,
    UniformSampler,
    make_sampler,
    register_sampler,
)
from skrl.multi_agents.torch.selfplay.runner import DEFAULT_AGENT_CFG, SelfPlayCfg, SelfPlayRunner

__all__ = [
    "AdaptivePBTSampler",
    "BestSampler",
    "DEFAULT_AGENT_CFG",
    "HedgeSampler",
    "LastSampler",
    "MLPCategoricalPolicy",
    "MLPValue",
    "OpponentPool",
    "OpponentSampler",
    "PolicySnapshot",
    "SAMPLERS",
    "SelfPlayCfg",
    "SelfPlayRunner",
    "UniformSampler",
    "elo_expected",
    "elo_update",
    "make_sampler",
    "register_sampler",
]
