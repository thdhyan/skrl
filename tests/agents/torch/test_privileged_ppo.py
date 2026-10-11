"""Tests for the privileged-curriculum PPO: curriculum, wrapper layout, gated actor, agent update."""
from __future__ import annotations

import numpy as np
import pytest
import torch
import gymnasium

from skrl.agents.torch.privileged_ppo import (
    ObsLayout,
    PrivilegedCategoricalPolicy,
    PrivilegedPPO,
    PrivilegedPPO_CFG,
    PrivilegedValue,
    SoftmaxMaskCurriculum,
)
from skrl.envs.wrappers.torch.privileged_history_wrapper import PrivilegedHistoryWrapper
from skrl.memories.torch import RandomMemory


D, H, P, NA, ROLLOUTS = 4, 3, 2, 2, 8
LAYOUT = ObsLayout(obs_dim=D, history_length=H, privileged_dim=P)


def test_curriculum_endpoints_and_monotone() -> None:
    c = SoftmaxMaskCurriculum(start=0.1, end=0.8, sharpness=10.0)
    ps = [c.mask_probability(x) for x in np.linspace(0, 1, 101)]
    assert ps[0] == 0.0 and ps[5] == 0.0
    assert ps[-1] == pytest.approx(1.0) and ps[90] == pytest.approx(1.0)
    assert all(b >= a for a, b in zip(ps, ps[1:]))
    assert 0.3 < c.mask_probability(0.45) < 0.7


def test_curriculum_rejects_bad_range() -> None:
    with pytest.raises(ValueError):
        SoftmaxMaskCurriculum(start=0.8, end=0.2)


def _wrapped_cartpole(curriculum=None, seed=0):
    env = gymnasium.make("CartPole-v1")
    return PrivilegedHistoryWrapper(
        env,
        privileged_fn=lambda e: np.asarray(e.unwrapped.state, dtype=np.float32)[[1, 3]],
        privileged_dim=P,
        history_length=H,
        curriculum=curriculum,
        seed=seed,
    )


def test_wrapper_layout_and_history() -> None:
    env = _wrapped_cartpole()
    obs, _ = env.reset(seed=0)
    assert obs.shape == (LAYOUT.size,) == env.observation_space.shape
    x = torch.tensor(obs)[None]
    assert torch.all(x[:, D : D * (1 + H)] == 0)  # history is zero-padded at reset
    assert LAYOUT.mask(x).item() == 0.0
    obs1, *_ = env.step(0)
    x1 = torch.tensor(obs1)[None]
    assert torch.allclose(x1[:, D * H : D * (1 + H)], x[:, :D])  # newest history slot = previous o_t
    assert torch.allclose(LAYOUT.privileged(x1), torch.tensor(env.unwrapped.state, dtype=torch.float32)[[1, 3]])


def test_wrapper_mask_follows_curriculum() -> None:
    c = SoftmaxMaskCurriculum()
    env = _wrapped_cartpole(c)
    masks = [env.reset()[0][-1] for _ in range(50)]
    assert set(masks) == {0.0}
    c.set_progress(1.0)
    masks = [env.reset()[0][-1] for _ in range(50)]
    assert set(masks) == {1.0}


def _space(n):
    return gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(n,), dtype=np.float32)


def _policy():
    return PrivilegedCategoricalPolicy(_space(LAYOUT.size), gymnasium.spaces.Discrete(NA), "cpu", layout=LAYOUT)


def test_masked_actor_ignores_privileged() -> None:
    policy = _policy()
    x = torch.randn(5, LAYOUT.size)
    y = x.clone()
    y[:, -P - 1 : -1] += 10.0  # change only the privileged values
    for mask, same in ((1.0, True), (0.0, False)):
        x[:, -1], y[:, -1] = mask, mask
        a = policy.compute({"observations": x})[0]
        b = policy.compute({"observations": y})[0]
        assert torch.allclose(a, b) == same


def _agent(curriculum=None):
    obs_space, act_space = _space(LAYOUT.size), gymnasium.spaces.Discrete(NA)
    models = {
        "policy": _policy(),
        "value": PrivilegedValue(obs_space, act_space, "cpu", layout=LAYOUT),
    }
    agent = PrivilegedPPO(
        models=models,
        memory=RandomMemory(memory_size=ROLLOUTS, num_envs=2, device="cpu"),
        observation_space=obs_space,
        action_space=act_space,
        device="cpu",
        cfg=PrivilegedPPO_CFG(rollouts=ROLLOUTS, learning_epochs=2, mini_batches=2, curriculum=curriculum),
    )
    agent.init()
    agent.enable_training_mode(True, apply_to_models=False)
    return agent


def test_rejects_observation_preprocessor() -> None:
    from skrl.resources.preprocessors.torch import RunningStandardScaler

    obs_space = _space(LAYOUT.size)
    with pytest.raises(ValueError):
        PrivilegedPPO(
            models={"policy": _policy(), "value": PrivilegedValue(obs_space, None, "cpu", layout=LAYOUT)},
            observation_space=obs_space,
            action_space=gymnasium.spaces.Discrete(NA),
            device="cpu",
            cfg={"observation_preprocessor": RunningStandardScaler},
        )


def test_update_trains_estimator_towards_encoder() -> None:
    torch.manual_seed(0)
    c = SoftmaxMaskCurriculum()
    agent = _agent(c)
    trunk = agent.policy.trunk

    # fixed batch where the privileged values are a function of the history, so ẑ can match z
    def obs_batch():
        x = torch.randn(2, LAYOUT.size)
        x[:, -P - 1 : -1] = x[:, D : D + P]
        x[:, -1] = 0.0
        return x

    probe = torch.cat([obs_batch() for _ in range(32)])
    with torch.no_grad():
        z, z_hat = trunk.latents(probe)
        before = torch.mean((z - z_hat) ** 2).item()

    timesteps = 40 * ROLLOUTS
    for t in range(timesteps):
        obs = obs_batch()
        with torch.no_grad():
            actions, _ = agent.act(obs, None, timestep=t, timesteps=timesteps)
            agent.record_transition(
                observations=obs, states=None, actions=actions, rewards=torch.ones(2, 1),
                next_observations=obs_batch(), next_states=None,
                terminated=torch.zeros(2, 1, dtype=torch.bool), truncated=torch.zeros(2, 1, dtype=torch.bool),
                infos={}, timestep=t, timesteps=timesteps,
            )
        agent.post_interaction(timestep=t, timesteps=timesteps)

    with torch.no_grad():
        z, z_hat = trunk.latents(probe)
        after = torch.mean((z - z_hat) ** 2).item()
    assert after < 0.5 * before
    assert c.progress == pytest.approx((timesteps - 1) / timesteps)
