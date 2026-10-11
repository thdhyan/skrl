"""Tests for BeliefPPO agent — mirrors test_belief_dqn.py."""
from __future__ import annotations

import pytest
import torch
import gymnasium

from skrl.agents.torch.belief_ppo import BeliefPPO, BeliefPPO_CFG, ResNetCategoricalPolicy, ResNetValue
from skrl.belief.backends.stub_backend import StubBackend
from skrl.belief.torch.particle_filter import ParticleFilter
from skrl.memories.torch import RandomMemory


N_JOINTS = 2
STATE_DIM = 10   # StubBackend default
NG = 2
NC = 4
NP = 50
ROLLOUTS = 8
BELIEF_DIM = NG + 2 * NG * STATE_DIM  # 42


def _space(n):
    return gymnasium.spaces.Box(low=-float("inf"), high=float("inf"), shape=(n,), dtype="float32")


def _make_belief_factory(device="cpu"):
    backend = StubBackend(n_joints=N_JOINTS, device=device, seed=0)

    def factory():
        return ParticleFilter(
            num_particles=NP,
            state_dim=STATE_DIM,
            num_gmm_components=NG,
            motion_model_fn=lambda particles, control: backend.step(particles, control, n_substeps=1),
            measurement_model_fn=lambda particles, measurement: backend.get_measurement(particles, sigma_enc=1.0),
            device=device,
        )

    return factory


def _make_agent(device="cpu", **cfg_overrides):
    belief_space = _space(BELIEF_DIM)
    action_space = gymnasium.spaces.Discrete(NC)
    models = {
        "policy": ResNetCategoricalPolicy(belief_space, action_space, device, resnet_blocks=2, width=32),
        "value": ResNetValue(belief_space, action_space, device, resnet_blocks=2, width=32),
    }
    cfg = dict(
        belief_factory=_make_belief_factory(device),
        belief_prior_fn=lambda obs: torch.randn(NP, STATE_DIM) * 0.1,
        rollouts=ROLLOUTS,
        learning_epochs=2,
        mini_batches=2,
        learning_rate=1e-3,
    )
    cfg.update(cfg_overrides)
    agent = BeliefPPO(
        models=models,
        memory=RandomMemory(memory_size=ROLLOUTS, num_envs=1, device=device),
        observation_space=_space(N_JOINTS),
        action_space=action_space,
        device=device,
        cfg=BeliefPPO_CFG(**cfg),
    )
    return agent


def _init(agent):
    agent.init()
    agent.enable_training_mode(True, apply_to_models=False)  # what SequentialTrainer.train() does
    return agent


@torch.no_grad()  # SequentialTrainer runs act/record_transition under no_grad
def _step(agent, t, timesteps=100, done=False):
    obs = torch.randn(1, N_JOINTS)
    actions, outputs = agent.act(obs, None, timestep=t, timesteps=timesteps)
    agent.record_transition(
        observations=obs,
        states=None,
        actions=actions,
        rewards=torch.tensor([[1.0]]),
        next_observations=torch.randn(1, N_JOINTS),
        next_states=None,
        terminated=torch.tensor([[done]]),
        truncated=torch.tensor([[False]]),
        infos={},
        timestep=t,
        timesteps=timesteps,
    )
    with torch.enable_grad():
        agent.post_interaction(timestep=t, timesteps=timesteps)
    return actions, outputs


def test_requires_belief_factory() -> None:
    with pytest.raises(ValueError):
        _make_agent(belief_factory=None)


def test_init_and_belief_dim() -> None:
    agent = _make_agent()
    _init(agent)
    assert agent.policy is not None and agent.value is not None
    assert agent.belief_dim == BELIEF_DIM


def test_act_shapes() -> None:
    agent = _make_agent()
    _init(agent)
    actions, outputs = agent.act(torch.randn(1, N_JOINTS), None, timestep=0, timesteps=10)
    assert actions.shape == (1, 1)
    assert 0 <= int(actions) < NC
    assert outputs["log_prob"].shape == (1, 1)
    assert agent._b_tilde.shape == (1, BELIEF_DIM)


def test_memory_stores_beliefs_not_obs() -> None:
    agent = _make_agent()
    _init(agent)
    _step(agent, 0)
    assert len(agent.memory) == 1
    beliefs = agent.memory.get_tensor_by_name("beliefs")
    assert beliefs.shape[-1] == BELIEF_DIM
    assert torch.isfinite(beliefs[0]).all()


def test_update_changes_parameters() -> None:
    agent = _make_agent()
    _init(agent)
    before = [p.detach().clone() for p in agent.policy.parameters()]
    for t in range(ROLLOUTS):  # update fires after the ROLLOUTS-th step
        _step(agent, t)
    after = list(agent.policy.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after))
    assert all(torch.isfinite(p).all() for p in after)


def test_training_loop_no_crash() -> None:
    agent = _make_agent()
    _init(agent)
    for t in range(3 * ROLLOUTS):
        _step(agent, t, done=(t % 5 == 4))


def test_prior_reset_on_episode_end() -> None:
    calls = []

    def prior(obs):
        calls.append(obs.shape)
        return torch.randn(NP, STATE_DIM) * 0.1

    agent = _make_agent(belief_prior_fn=prior)
    _init(agent)
    _step(agent, 0)               # first act seeds the filter
    _step(agent, 1, done=True)    # episode ends -> next act re-seeds
    _step(agent, 2)
    assert calls == [(N_JOINTS,), (N_JOINTS,)]


def test_rejects_multiple_envs() -> None:
    agent = _make_agent()
    _init(agent)
    with pytest.raises(ValueError):
        agent.act(torch.randn(2, N_JOINTS), None, timestep=0, timesteps=10)


def test_belief_markers_in_outputs_and_stats_tracked() -> None:
    agent = _init(
        _make_agent(
            return_belief_markers=True,
            belief_true_state_fn=lambda infos: torch.zeros(STATE_DIM),
            belief_state_to_position=lambda s: s[..., :3],
        )
    )
    _, outputs = _step(agent, 0)
    markers = outputs["belief_markers"]
    assert markers["particles"].shape == (NP, STATE_DIM) and markers["cluster_positions"].shape == (NG, 3)
    _step(agent, 1)
    for key in ["spread", "ess", "num_clusters", "cluster_std", "mean_error", "cluster_error_min", "cluster_error 0"]:
        assert f"Belief / {key}" in agent.tracking_data
    assert agent.get_belief_markers()["stats"]["num_clusters"] >= 1
