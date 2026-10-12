import math

import numpy as np
import pytest

pytest.importorskip("mujoco")

from skrl.belief.grid_belief import GridBelief, entropy, expected_information_gain
from skrl.envs.cube_push import CubePushEnv, pose_error
from skrl.envs.cube_push.sim import FORCE, cube_pose, rollout, substep_controls
from skrl.envs.cube_push.wafr08 import WAFR08CubePlanner


OPTIONS = {"cube_pose": [0.06, 0.0, 0.2], "physics": [0.5, 0.4, 1.0], "goal": [0.25, 0.05, 0.0]}


def test_env_reset_step():
    env = CubePushEnv()
    obs, info = env.reset(seed=0, options=OPTIONS)
    assert obs.shape == (10,) and env.observation_space.contains(obs)
    np.testing.assert_allclose(info["cube_pose"], OPTIONS["cube_pose"], atol=1e-6)
    np.testing.assert_allclose(obs[6:], [0.25, 0.05, 1.0, 0.0], atol=1e-6)
    obs, reward, terminated, truncated, info = env.step([0.5, 0.0, env.cfg.z_high])
    # the target moves at most max_speed * dt per step
    assert env.target[0] == pytest.approx(env.cfg.pusher_start[0] + env.cfg.max_speed * env.cfg.control_dt)
    assert not terminated and not truncated and not info["success"]
    assert reward == pytest.approx(-pose_error(info["cube_pose"], env.goal)[0] - 0.05 * pose_error(
        info["cube_pose"], env.goal)[1])


def test_rollout_matches_env():
    env = CubePushEnv()
    env.reset(seed=0, options=OPTIONS)
    start = np.array([-0.02, 0.0, env.cfg.z_low])
    env.sim.set_state(env.data, start, OPTIONS["cube_pose"])
    env.target = start.copy()
    targets = np.stack([start + [0.002 * (k + 1), 0.0, 0.0] for k in range(60)])
    trace = []
    for t in targets:
        env.step(t)
        trace.append(env.data.sensordata[FORCE].copy())
    model = env.sim.variant(*OPTIONS["physics"])
    sd = rollout([model], env.sim.initial_states(start, np.array([OPTIONS["cube_pose"]])),
                 substep_controls(start, targets, env.cfg.substeps))
    steps = sd[0, env.cfg.substeps - 1 :: env.cfg.substeps]
    np.testing.assert_allclose(steps[:, FORCE], np.array(trace), atol=1e-6)
    np.testing.assert_allclose(cube_pose(steps[-1]), env.cube_pose(), atol=1e-6)
    assert np.abs(np.array(trace)[:, 0]).max() > 0.5  # the pusher touched the cube


def test_grid_belief_update_and_blur():
    b = GridBelief([np.arange(5.0), np.arange(4.0), np.array([0.0, 1.0])], moving=(0, 1), periods={1: 4.0})
    flat, w = b.support(1.0)
    assert len(flat) == 40 and w.sum() == pytest.approx(1.0)
    values = b.values(flat)
    next_values = values.copy()
    next_values[:, 0] += 0.5  # half a cell: split between two cells, x = 4 drops half its mass off the grid
    next_values[:, 1] += 1.0  # periodic shift
    ll = np.where(values[:, 2] == 1.0, 0.0, -np.inf)  # the observation rules out the static level 0
    b.update(flat, ll, next_values)
    assert b.p.sum() == pytest.approx(1.0)
    assert b.marginal((2,))[0] == 0.0
    np.testing.assert_allclose(b.marginal((1,)), 0.25)
    # x cells keep [1/2, 1, 1, 1, 1] of their mass (half of x = 4 left the grid), renormalized
    np.testing.assert_allclose(b.marginal((0,)), np.array([0.5, 1, 1, 1, 1]) / 4.5)
    before = b.entropy()
    b.blur()
    assert b.p.sum() == pytest.approx(1.0) and b.entropy() >= before
    np.testing.assert_array_equal(b.nearest(b.values(np.arange(40))), np.arange(40))


def test_information_gain():
    w = np.full(4, 0.25)
    assert expected_information_gain(w, np.where(np.eye(4) > 0, 0.0, -50.0)) == pytest.approx(math.log(4))
    assert expected_information_gain(w, np.zeros((4, 4))) == pytest.approx(0.0, abs=1e-12)
    # perfectly informative, but the action maps every state to one next cell: nothing left to gain
    assert expected_information_gain(w, np.where(np.eye(4) > 0, 0.0, -50.0), np.zeros(4)) == pytest.approx(
        math.log(4))
    assert entropy(w) == pytest.approx(math.log(4))


def test_planner_predicts_execution_and_decides():
    env = CubePushEnv()
    _, info = env.reset(seed=0, options={**OPTIONS, "cube_pose": [0.06, 0.0, 0.0], "physics": [0.55, 0.4, 1.5]})
    planner = WAFR08CubePlanner(env.cfg)
    planner.reset(info["goal"])
    true_state = np.array([0.06, 0.0, math.radians(5.0), 0.55, 0.4, 1.5])  # a grid cell next to the truth
    for wrt in (planner.touch(true_state, 2, 0.0), planner.probe(true_state, 2)):
        predicted = planner.predict(wrt, true_state[None])
        observed = planner.execute(env, wrt)
        assert observed[0] == predicted.obs[0, 0] == 1.0
        np.testing.assert_allclose(observed[1], predicted.obs[0, 1], atol=0.01)
    assert predicted.obs[0, 2] > 0.5  # pushing force at v1 ≈ μ m g + k_d v
    # a belief concentrated on one pose: the next decision is a goal push or a probe at that pose
    b = planner.belief
    p = np.zeros(b.shape)
    flat = b.nearest(np.concatenate([env.cube_pose(), [0.55, 0.4, 1.5]])[None])
    p.ravel()[flat] = 1.0
    b.set(p)
    wrt, report = planner.decide()
    assert wrt is not None and wrt.kind in ("goal", "probe") and report["p_goal"] == 0.0
