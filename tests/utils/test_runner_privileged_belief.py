"""Runner (YAML) support for PrivilegedPPO, BeliefPPO and BeliefDQN, and the batched privileged-history wrapper."""
from __future__ import annotations

import os
import types

import gymnasium
import numpy as np
import pytest
import torch

from skrl.agents.torch.belief_dqn import BeliefDQN
from skrl.agents.torch.belief_ppo import BeliefPPO
from skrl.agents.torch.privileged_ppo import (
    STRATEGIES,
    PrivilegedPPO,
    SoftmaxMaskCurriculum,
    TwoPhaseMaskCurriculum,
    apply_strategy,
    make_curriculum,
)
from skrl.envs.wrappers.torch.base import Wrapper
from skrl.envs.wrappers.torch.privileged_history_vec_wrapper import PrivilegedHistoryVecWrapper
from skrl.utils.runner.torch import Runner

EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "privileged_curriculum", "isaaclab")


class FakeIsaacLabEnv(Wrapper):
    """Isaac Lab-like skrl env: batched torch tensors, ``state()`` = privileged "critic" group, same-step auto-reset.

    Observation: the step counter of each env (so the history can be checked) and a constant 1. State: the
    per-episode mass and the step counter."""

    def __init__(self, num_envs: int = 4, episode_length: int = 5, discrete: bool = False) -> None:
        super().__init__(types.SimpleNamespace(device="cpu"))
        self._n, self._length, self._discrete = num_envs, episode_length, discrete
        self.t = torch.zeros(num_envs)
        self.mass = torch.rand(num_envs) + 0.5

    @property
    def num_envs(self) -> int:
        return self._n

    @property
    def observation_space(self):
        return gymnasium.spaces.Box(-np.inf, np.inf, (2,))

    @property
    def state_space(self):
        return gymnasium.spaces.Box(-np.inf, np.inf, (2,))

    @property
    def action_space(self):
        return gymnasium.spaces.Discrete(4) if self._discrete else gymnasium.spaces.Box(-1.0, 1.0, (1,))

    def _obs(self):
        return torch.stack([self.t, torch.ones(self._n)], dim=-1)

    def reset(self):
        return self._obs(), {}

    def step(self, actions):
        self.t += 1
        truncated = (self.t >= self._length).view(-1, 1)
        rewards = -actions.float().reshape(self._n, -1).pow(2).sum(-1, keepdim=True) / self.mass.view(-1, 1)
        done = truncated.view(-1)
        self.t[done] = 0  # auto-reset: the returned observation starts the next episode
        self.mass[done] = torch.rand(int(done.sum())) + 0.5
        return self._obs(), rewards, torch.zeros_like(truncated), truncated, {}

    def state(self):
        return torch.stack([self.mass, self.t], dim=-1)

    def render(self, *args, **kwargs):
        return None

    def close(self):
        pass


def _privileged_cfg(**agent):
    return {
        "seed": 0,
        "privileged_history": {"history_length": 3, "include_last_action": True},
        "models": {
            "separate": True,
            "policy": {"class": "PrivilegedGaussianPolicy", "latent_dim": 4, "hidden": 16, "estimator": "transformer"},
            "value": {"class": "PrivilegedValue", "hidden": 16},
        },
        "memory": {"class": "RandomMemory", "memory_size": -1},
        "agent": {
            "class": "PrivilegedPPO",
            "rollouts": 8,
            "learning_epochs": 1,
            "mini_batches": 2,
            "experiment": {"write_interval": 0, "checkpoint_interval": 0},
            **agent,
        },
        "trainer": {"class": "SequentialTrainer", "timesteps": 32, "headless": True, "disable_progressbar": True},
    }


# --- wrapper ------------------------------------------------------------------------------------------------------


def test_vec_wrapper_packs_history_privileged_and_mask():
    env = PrivilegedHistoryVecWrapper(
        FakeIsaacLabEnv(num_envs=3, episode_length=4), history_length=2, include_last_action=True,
        curriculum=make_curriculum({"class": "FixedMaskCurriculum", "p": 1.0}),
    )
    layout = env.layout
    assert (layout.obs_dim, layout.history_length, layout.privileged_dim) == (3, 2, 2)
    assert env.observation_space.shape == (layout.size,)
    x, _ = env.reset()
    assert torch.all(layout.mask(x) == 1.0)
    assert torch.all(layout.with_history(x)[:, 3:] == 0.0)  # empty history at episode start
    actions = torch.full((3, 1), 0.5)
    for _ in range(2):
        x, *_ = env.step(actions)
    seq = layout.sequence(x)  # [B, 1 + H, D], oldest first: steps 0, 1, 2 and the previous actions 0, .5, .5
    assert torch.equal(seq[0, :, 0], torch.tensor([0.0, 1.0, 2.0]))
    assert torch.equal(seq[0, :, 2], torch.tensor([0.0, 0.5, 0.5]))
    assert torch.allclose(layout.privileged(x), env.state())
    x, *_ = env.step(actions)
    x, *_ = env.step(actions)  # episode length 4: auto-reset, history and last action cleared
    assert torch.all(layout.current(x)[:, 0] == 0.0) and torch.all(layout.with_history(x)[:, 2:] == 0.0)


def test_vec_wrapper_draws_mask_per_episode():
    curriculum = SoftmaxMaskCurriculum(start=0.0, end=1.0, p_start=0.5, p_end=0.5)
    env = PrivilegedHistoryVecWrapper(FakeIsaacLabEnv(num_envs=256, episode_length=3), history_length=1,
                                      curriculum=curriculum, seed=0)
    x, _ = env.reset()
    first = env.layout.mask(x).clone()
    assert 0.3 < first.mean() < 0.7
    x, *_ = env.step(torch.zeros(256, 1))
    assert torch.equal(env.layout.mask(x), first)  # constant within the episode
    for _ in range(2):
        x, *_ = env.step(torch.zeros(256, 1))
    assert not torch.equal(env.layout.mask(x), first)  # redrawn for the new episodes


# --- strategies ---------------------------------------------------------------------------------------------------


def test_strategy_presets_and_overrides():
    policy, agent = apply_strategy("sysid", {"estimator": "lstm"}, {"sysid_loss_scale": 0.5})
    assert policy == {"sysid_head": True, "estimator": "lstm"}
    assert agent["sysid_loss_scale"] == 0.5 and agent["encoder_loss_scale"] == 0.0
    policy, agent = apply_strategy("uposi", {}, {"freeze_policy_after": 0.6})
    assert policy["osi"] and isinstance(make_curriculum(agent["curriculum"]), TwoPhaseMaskCurriculum)
    assert agent["curriculum"]["switch"] == 0.75  # the preset curriculum keeps its own switch
    _, agent = apply_strategy("roa", {}, {"curriculum": {"class": "FixedMaskCurriculum"}})
    assert agent["curriculum"] == {"class": "FixedMaskCurriculum"}
    with pytest.raises(ValueError):
        apply_strategy("nope", {}, {})


# --- runner -------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", ["roa", "sysid", "whiten", "osi", "rma"])
def test_runner_builds_and_trains_privileged_ppo(strategy):
    runner = Runner(FakeIsaacLabEnv(), _privileged_cfg(strategy=strategy,
                                                       curriculum={"class": "SoftmaxMaskCurriculum"}))
    agent = runner.agent
    assert isinstance(agent, PrivilegedPPO)
    assert isinstance(runner.trainer.env, PrivilegedHistoryVecWrapper)
    assert agent.cfg.curriculum is runner.trainer.env._curriculum  # shared, not copied
    trunk = agent.policy.trunk
    assert trunk.estimator.__class__.__name__ == "_CausalTransformerEstimator"
    assert (trunk.sysid is not None) == (strategy == "sysid")
    assert trunk.osi == (strategy == "osi")
    assert agent.cfg.estimator_whiten == (strategy == "whiten")
    assert agent.cfg.freeze_policy_after == (0.75 if strategy == "rma" else None)
    runner.run("train")
    assert agent.cfg.curriculum.progress > 0.9


def test_runner_contrastive_needs_dims_and_takes_them_from_yaml():
    with pytest.raises(ValueError):
        Runner(FakeIsaacLabEnv(), _privileged_cfg(strategy="contrastive"))
    runner = Runner(FakeIsaacLabEnv(), _privileged_cfg(strategy="contrastive", contrastive_dims=[0]))
    runner.run("train")


def test_runner_layout_without_wrapper_and_import_string_model():
    env = PrivilegedHistoryVecWrapper(FakeIsaacLabEnv(), history_length=2)
    cfg = _privileged_cfg()
    del cfg["privileged_history"]
    layout = {"obs_dim": 2, "history_length": 2, "privileged_dim": 2}
    cfg["models"]["policy"] = {"class": "skrl.agents.torch.privileged_ppo:PrivilegedGaussianPolicy",
                               "layout": layout, "estimator": "lstm"}
    cfg["models"]["value"]["layout"] = layout
    runner = Runner(env, cfg)
    assert runner.agent.policy.trunk.estimator.__class__.__name__ == "_LSTMEstimator"
    runner.run("train")


def test_example_privileged_yaml_builds():
    cfg = Runner.load_cfg_from_yaml(os.path.join(EXAMPLES, "skrl_privileged_ppo_cfg.yaml"))
    assert cfg["agent"]["strategy"] == "sysid"
    cfg["trainer"].update(timesteps=48, headless=True, disable_progressbar=True)
    cfg["agent"]["experiment"].update(write_interval=0, checkpoint_interval=0)
    runner = Runner(FakeIsaacLabEnv(), cfg)
    assert runner.agent.cfg.sysid_loss_scale == 1.0
    runner.run("train")


def test_example_belief_yaml_builds_belief_ppo():
    cfg = Runner.load_cfg_from_yaml(os.path.join(EXAMPLES, "skrl_belief_ppo_cfg.yaml"))
    cfg["belief"]["num_particles"] = 30
    cfg["trainer"].update(timesteps=20, headless=True, disable_progressbar=True)
    cfg["agent"].update(rollouts=8, experiment={"write_interval": 0, "checkpoint_interval": 0})
    runner = Runner(FakeIsaacLabEnv(num_envs=1, discrete=True), cfg)
    agent = runner.agent
    assert isinstance(agent, BeliefPPO)
    assert agent.policy.num_observations == agent.belief_dim == 2 + 2 * 2 * 10
    runner.run("train")


def test_runner_belief_dqn_with_import_string_factory():
    cfg = {
        "seed": 0,
        "models": {
            "separate": True,
            "q_network": {"class": "ResNetDuelingQNetwork", "resnet_blocks": 1, "width": 16},
            "target_q_network": {"class": "ResNetDuelingQNetwork", "resnet_blocks": 1, "width": 16},
        },
        "memory": {"class": "RandomMemory", "memory_size": 100},
        "agent": {
            "class": "BeliefDQN",
            "belief_factory": "tests.utils.test_runner_privileged_belief:small_filter",
            "belief_dim": 42,
            "batch_size": 4,
            "learning_starts": 4,
            "experiment": {"write_interval": 0, "checkpoint_interval": 0},
        },
        "trainer": {"class": "SequentialTrainer", "timesteps": 10, "headless": True, "disable_progressbar": True},
    }
    runner = Runner(FakeIsaacLabEnv(num_envs=1, discrete=True), cfg)
    assert isinstance(runner.agent, BeliefDQN)
    runner.run("train")


def small_filter():
    from skrl.belief.backends.stub_backend import StubBackend
    from skrl.belief.torch.particle_filter import ParticleFilter

    backend = StubBackend(n_joints=2, seed=0)
    return ParticleFilter(
        num_particles=30,
        state_dim=10,
        motion_model_fn=lambda p, c: backend.step(p, c, n_substeps=1),
        measurement_model_fn=lambda p, z: backend.get_measurement(p),
    )


def test_strategies_cover_the_documented_names():
    assert set(STRATEGIES) == {"roa", "roa_lam0", "sysid", "contrastive", "whiten", "osi", "rma", "uposi"}
