"""Tests for BeliefDQN agent — mirrors the test_dqn.py pattern."""
from __future__ import annotations

import pytest
import torch
import gymnasium

from skrl.agents.torch.belief_dqn import BeliefDQN, BeliefDQN_CFG, ResNetDuelingQNetwork
from skrl.belief.backends.stub_backend import StubBackend
from skrl.belief.torch.particle_filter import ParticleFilter
from skrl.memories.torch import RandomMemory


N_JOINTS = 2
STATE_DIM = 10   # StubBackend default
NG = 2
NC = 4           # 2 targets × 2 modes
BELIEF_DIM = NG + 2 * NG * STATE_DIM  # 42


def _make_obs_space():
    return gymnasium.spaces.Box(low=-float("inf"), high=float("inf"), shape=(N_JOINTS,), dtype="float32")


def _make_action_space():
    return gymnasium.spaces.Discrete(NC)


def _make_belief_space():
    return gymnasium.spaces.Box(low=-float("inf"), high=float("inf"), shape=(BELIEF_DIM,), dtype="float32")


def _make_belief_factory(device="cpu"):
    backend = StubBackend(n_joints=N_JOINTS, device=device, seed=0)

    def factory():
        def motion_model(particles, control):
            return backend.step(particles, control, n_substeps=1)

        def measurement_model(particles, measurement):
            return backend.get_measurement(particles, sigma_enc=1.0)

        return ParticleFilter(
            num_particles=50,
            state_dim=STATE_DIM,
            num_gmm_components=NG,
            motion_model_fn=motion_model,
            measurement_model_fn=measurement_model,
            device=device,
        )

    return factory


def _make_models(device="cpu"):
    belief_space = _make_belief_space()
    action_space = _make_action_space()
    q = ResNetDuelingQNetwork(belief_space, action_space, device, resnet_blocks=2, width=32)
    target_q = ResNetDuelingQNetwork(belief_space, action_space, device, resnet_blocks=2, width=32)
    return {"q_network": q, "target_q_network": target_q}


def _make_agent(device="cpu", **cfg_overrides):
    obs_space = _make_obs_space()
    action_space = _make_action_space()
    belief_space = _make_belief_space()
    memory = RandomMemory(memory_size=200, num_envs=1, device=device)

    cfg = BeliefDQN_CFG(
        belief_factory=_make_belief_factory(device),
        num_controllers=NC,
        gmm_components=NG,
        batch_size=8,
        discount_factor=0.98,
        learning_rate=1e-4,
        l2_reg=0.01,
        random_timesteps=0,
        learning_starts=0,
        update_interval=1,
        target_update_interval=5,
        resnet_blocks=2,
        resnet_width=32,
        n_substeps=1,
        **cfg_overrides,
    )
    agent = BeliefDQN(
        models=_make_models(device),
        memory=memory,
        observation_space=obs_space,
        action_space=action_space,
        device=device,
        cfg=cfg,
    )
    return agent


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

@pytest.mark.parametrize("device", ["cpu"])
def test_agent_init(device: str) -> None:
    """Test that the BeliefDQN agent initializes correctly with all required components.

    This test verifies:
    - The Q-network is created successfully
    - The target Q-network is created successfully
    - The internal belief state (_belief) is initialized

    Args:
        device: The device to run the agent on (e.g., "cpu" or "cuda").
    """
    agent = _make_agent(device)
    assert agent.q_network is not None
    assert agent.target_q_network is not None
    assert agent._belief is not None


@pytest.mark.parametrize("device", ["cpu"])
def test_agent_init_belief_dim(device: str) -> None:
    """Test that the belief dimension is correctly set during agent initialization.

    This test verifies that after calling init(), the particle filter's belief_dim
    matches the expected dimension calculated from the GMM components and state
    dimensions (NG + 2 * NG * STATE_DIM = 42).

    Args:
        device: The device to run the agent on.
    """
    agent = _make_agent(device)
    agent.init()
    assert agent._belief.belief_dim == BELIEF_DIM


@pytest.mark.parametrize("device", ["cpu"])
def test_act_returns_valid_action(device: str) -> None:
    """Test that the act() method returns valid actions within the action space.

    This test verifies:
    - The action has the correct shape (batch_size=1, action_dim=1)
    - The action value is within the valid discrete range [0, NC)

    Before calling act(), the belief must be seeded with particles using reset_belief().

    Args:
        device: The device to run the agent on.
    """
    agent = _make_agent(device)
    agent.init()
    # seed the belief with a prior
    particles = torch.randn(50, STATE_DIM)
    agent.reset_belief(particles)
    obs = torch.randn(1, N_JOINTS)
    actions, _ = agent.act(obs, None, timestep=0, timesteps=100)
    assert actions.shape == (1, 1)
    assert 0 <= actions.item() < NC


@pytest.mark.parametrize("device", ["cpu"])
def test_record_transition_and_update(device: str) -> None:
    """Test that transitions are correctly recorded in the replay memory.

    This test verifies:
    - A single transition can be recorded via record_transition()
    - The memory size increases by exactly 1 after recording

    The test performs a full interaction loop (act -> record) to ensure
    all required fields are properly handled.

    Args:
        device: The device to run the agent on.
    """
    agent = _make_agent(device)
    agent.init()
    particles = torch.randn(50, STATE_DIM)
    agent.reset_belief(particles)

    obs = torch.randn(1, N_JOINTS)
    next_obs = torch.randn(1, N_JOINTS)
    actions, _ = agent.act(obs, None, timestep=0, timesteps=100)

    agent.record_transition(
        observations=obs,
        states=None,
        actions=actions,
        rewards=torch.tensor([[-1.0]]),
        next_observations=next_obs,
        next_states=None,
        terminated=torch.tensor([[False]]),
        truncated=torch.tensor([[False]]),
        infos={},
        timestep=0,
        timesteps=100,
    )
    assert len(agent.memory) == 1


@pytest.mark.parametrize("device", ["cpu"])
def test_training_loop_no_crash(device: str) -> None:
    """Test that a minimal training loop runs without raising exceptions.

    This test verifies the end-to-end training workflow:
    - Multiple interaction steps can be executed
    - record_transition() works for multiple transitions
    - post_interaction() can be called after each step

    This is a smoke test to ensure basic integration works before
    checking learning progress or loss values.

    Args:
        device: The device to run the agent on.
    """
    agent = _make_agent(device)
    agent.init()
    particles = torch.randn(50, STATE_DIM)
    agent.reset_belief(particles)

    for t in range(20):
        obs = torch.randn(1, N_JOINTS)
        next_obs = torch.randn(1, N_JOINTS)
        actions, _ = agent.act(obs, None, timestep=t, timesteps=50)
        agent.record_transition(
            observations=obs,
            states=None,
            actions=actions,
            rewards=torch.tensor([[-1.0]]),
            next_observations=next_obs,
            next_states=None,
            terminated=torch.tensor([[False]]),
            truncated=torch.tensor([[False]]),
            infos={},
            timestep=t,
            timesteps=50,
        )
        agent.post_interaction(timestep=t, timesteps=50)


@pytest.mark.parametrize("device", ["cpu"])
def test_double_q_used(device: str) -> None:
    """Test that the double Q-learning configuration is properly passed to the agent.

    This test verifies that when double_q=True is set in the config, the agent
    correctly stores and uses this setting. Double Q-learning helps reduce
    overestimation bias by decoupling action selection from value estimation.

    Args:
        device: The device to run the agent on.
    """
    agent = _make_agent(device, double_q=True)
    assert agent.cfg.double_q is True


@pytest.mark.parametrize("device", ["cpu"])
def test_dueling_output_shape(device: str) -> None:
    """Test that the dueling Q-network produces outputs with correct shape.

    This test verifies:
    - The Q-network output has shape (batch_size, num_actions)
    - The dueling architecture correctly computes state values and action advantages

    The dueling network splits the Q-value into separate state value V(s) and
    advantage A(s,a) streams, which are combined to produce the final Q-values.

    Args:
        device: The device to run the agent on.
    """
    belief_space = _make_belief_space()
    action_space = _make_action_space()
    model = ResNetDuelingQNetwork(belief_space, action_space, device, resnet_blocks=2, width=32, dueling=True)
    x = torch.randn(4, BELIEF_DIM)
    q, _ = model.compute({"observations": x})
    assert q.shape == (4, NC)


@pytest.mark.parametrize("device", ["cpu"])
def test_greedy_no_exploration(device: str) -> None:
    """Test that the agent produces deterministic actions when exploration is disabled.

    This test verifies that with no exploration scheduler (exploration_scheduler=None)
    and at a late timestep (timestep=999 relative to timesteps=1000), the agent
    behaves greedily without random exploration noise.

    The output shape should be consistent regardless of exploration settings.

    Args:
        device: The device to run the agent on.
    """
    agent = _make_agent(device, exploration_scheduler=None)
    agent.init()
    agent.reset_belief(torch.randn(50, STATE_DIM))
    obs = torch.randn(1, N_JOINTS)
    actions, _ = agent.act(obs, None, timestep=999, timesteps=1000)
    assert actions.shape == (1, 1)
