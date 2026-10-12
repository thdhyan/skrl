"""Tests for the privileged-curriculum PPO: curriculum, wrapper layout, gated actor, agent update."""
from __future__ import annotations

import numpy as np
import pytest
import torch
import gymnasium

from skrl.agents.torch.privileged_ppo import (
    ESTIMATORS,
    PrivilegedActorTrunk,
    ObsLayout,
    PrivilegedCategoricalPolicy,
    PrivilegedPPO,
    PrivilegedPPO_CFG,
    PrivilegedValue,
    RewardGatedMaskCurriculum,
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


def test_wrapper_include_last_action() -> None:
    env = PrivilegedHistoryWrapper(
        gymnasium.make("Pendulum-v1"),
        privileged_fn=lambda e: np.zeros(1),
        privileged_dim=1,
        history_length=2,
        include_last_action=True,
    )
    layout = ObsLayout(obs_dim=3 + 1, history_length=2, privileged_dim=1)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (layout.size,) == env.observation_space.shape
    assert obs[3] == 0.0  # no previous action at episode start
    obs, *_ = env.step(np.array([1.5], dtype=np.float32))
    assert obs[3] == 1.5
    assert obs[4 + 4 + 3] == 0.0  # newest history slot holds the reset step's (zero) action
    obs, *_ = env.step(np.array([-0.5], dtype=np.float32))
    assert obs[3] == -0.5 and obs[4 + 4 + 3] == 1.5
    with pytest.raises(ValueError):
        PrivilegedHistoryWrapper(
            gymnasium.make("CartPole-v1"), privileged_fn=lambda e: np.zeros(1), privileged_dim=1,
            history_length=2, include_last_action=True,
        )


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


def test_reward_gated_curriculum_steps_on_plateau_and_backs_off() -> None:
    c = RewardGatedMaskCurriculum(step=0.25, tolerance=0.05, patience=2, smoothing=1.0, deadline=None)
    for r in [10, 10, 10, 10]:  # flat return of the initial policy: not armed
        c.observe_return(r)
    assert not c.armed and c.mask_probability() == 0.0
    for r in [50, 100, 150]:  # still rising: no masking
        c.observe_return(r)
    assert c.armed and c.mask_probability() == 0.0
    for _ in range(2):  # constant return: one step
        c.observe_return(150)
    assert c.mask_probability() == pytest.approx(0.25)
    for _ in range(2):  # the return holds its level with less information: another step
        c.observe_return(146)
    assert c.mask_probability() == pytest.approx(0.5)
    c.observe_return(100)  # large drop: back off
    assert c.mask_probability() == pytest.approx(0.25) and c.steps_down == 1


def test_reward_gated_curriculum_deadline_floor() -> None:
    c = RewardGatedMaskCurriculum(deadline=0.5)
    assert c.mask_probability(0.25) == pytest.approx(0.5)
    assert c.mask_probability(0.9) == pytest.approx(1.0)


def test_probes_report_r2_per_group() -> None:
    torch.manual_seed(0)
    c = RewardGatedMaskCurriculum(patience=1)
    agent = _agent(c)
    agent.cfg.privileged_groups = {"a": [0], "b": [1]}
    agent.privileged_groups = agent.cfg.privileged_groups
    agent.write_interval = 1000  # track episode returns without writing
    timesteps = 3 * ROLLOUTS
    for t in range(timesteps):
        obs = torch.randn(2, LAYOUT.size)
        obs[:, -1] = 1.0
        with torch.no_grad():
            actions, _ = agent.act(obs, None, timestep=t, timesteps=timesteps)
            agent.record_transition(
                observations=obs, states=None, actions=actions, rewards=torch.ones(2, 1),
                next_observations=obs, next_states=None,
                terminated=torch.ones(2, 1, dtype=torch.bool), truncated=torch.zeros(2, 1, dtype=torch.bool),
                infos={}, timestep=t, timesteps=timesteps,
            )
        agent.post_interaction(timestep=t, timesteps=timesteps)
    assert set(agent.probe_metrics) == {"z R2 a", "z R2 b", "ẑ R2 a", "ẑ R2 b"}
    assert c.smoothed is not None


def test_layout_sequence_is_oldest_first() -> None:
    env = _wrapped_cartpole()
    obs0, _ = env.reset(seed=0)
    obs1, *_ = env.step(0)
    seq = LAYOUT.sequence(torch.tensor(obs1)[None])
    assert seq.shape == (1, 1 + H, D)
    assert torch.allclose(seq[0, -1], torch.tensor(obs1[:D]))  # newest: o_t
    assert torch.allclose(seq[0, -2], torch.tensor(obs0[:D]))  # then o_{t-1}
    assert torch.all(seq[0, :-2] == 0)  # zero padding from the reset


@pytest.mark.parametrize("arch", ESTIMATORS)
def test_estimator_architectures(arch) -> None:
    torch.manual_seed(0)
    trunk = PrivilegedActorTrunk(LAYOUT, latent_dim=5, hidden=16, estimator=arch, sysid_head=True)
    x = torch.randn(7, LAYOUT.size)
    z, z_hat = trunk.latents(x)
    assert z.shape == z_hat.shape == (7, 5)
    assert trunk.sysid(z_hat).shape == (7, P)
    y = x.clone()
    y[:, -P - 1 :] += 10.0  # privileged values and mask: the estimator must ignore them
    assert torch.allclose(trunk.latents(y)[1], z_hat, atol=1e-6)
    if arch == "transformer":  # causal over time; the newest token sees everything, so check the oldest frame matters
        y = x.clone()
        y[:, D : 2 * D] += 1.0
        assert not torch.allclose(trunk.latents(y)[1], z_hat)


def test_sysid_head_regresses_privileged_values() -> None:
    torch.manual_seed(0)
    obs_space, act_space = _space(LAYOUT.size), gymnasium.spaces.Discrete(NA)
    policy = PrivilegedCategoricalPolicy(obs_space, act_space, "cpu", layout=LAYOUT, estimator="lstm", sysid_head=True)
    agent = PrivilegedPPO(
        models={"policy": policy, "value": PrivilegedValue(obs_space, act_space, "cpu", layout=LAYOUT)},
        memory=RandomMemory(memory_size=ROLLOUTS, num_envs=2, device="cpu"),
        observation_space=obs_space,
        action_space=act_space,
        device="cpu",
        cfg=PrivilegedPPO_CFG(
            rollouts=ROLLOUTS, learning_epochs=2, mini_batches=2, encoder_loss_scale=0.0, sysid_loss_scale=1.0,
            privileged_groups={"a": [0], "b": [1]},
        ),
    )
    agent.init()
    agent.enable_training_mode(True, apply_to_models=False)

    def obs_batch():  # privileged values are a function of the oldest history frame
        x = torch.randn(2, LAYOUT.size)
        x[:, -P - 1 : -1] = x[:, D : D + P]
        x[:, -1] = 1.0
        return x

    timesteps = 60 * ROLLOUTS
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
    assert agent.probe_metrics["ψ(ẑ) R2 a"] > 0.5 and agent.probe_metrics["ψ(ẑ) R2 b"] > 0.5


def test_sysid_needs_head() -> None:
    obs_space = _space(LAYOUT.size)
    with pytest.raises(ValueError):
        PrivilegedPPO(
            models={"policy": _policy(), "value": PrivilegedValue(obs_space, None, "cpu", layout=LAYOUT)},
            observation_space=obs_space,
            action_space=gymnasium.spaces.Discrete(NA),
            device="cpu",
            cfg={"sysid_loss_scale": 1.0},
        )
