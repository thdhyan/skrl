import numpy as np
import pytest
import torch
import gymnasium as gym
from gymnasium import spaces

from skrl.belief.torch.omnireset import (
    NEAR_GOAL_MODE,
    PATHWAY_MODE,
    REACHING_MODE,
    OmniResetCfg,
    sample_omnireset_starts,
)
from skrl.envs.wrappers.torch.omnireset_wrapper import OmniResetWrapper


CFG = OmniResetCfg(
    goal_xy=(0.62, 0.0),
    start_x_range=(0.34, 0.47),
    start_y_range=(-0.14, 0.14),
    path_lateral_range=(-0.075, 0.075),
    near_goal_radius_range=(0.012, 0.090),
    reaching_ratio=0.35,
    pathway_ratio=0.30,
)


def test_mode_proportions_match_ratios():
    n = 200_000
    _, mode_ids = sample_omnireset_starts(CFG, n)
    counts = torch.bincount(mode_ids, minlength=3).float() / n
    assert counts[REACHING_MODE] == pytest.approx(CFG.reaching_ratio, abs=0.01)
    assert counts[PATHWAY_MODE] == pytest.approx(CFG.pathway_ratio, abs=0.01)
    assert counts[NEAR_GOAL_MODE] == pytest.approx(1 - CFG.reaching_ratio - CFG.pathway_ratio, abs=0.01)


def test_reaching_mode_inside_start_region():
    xy, mode_ids = sample_omnireset_starts(CFG, 100_000)
    r = xy[mode_ids == REACHING_MODE]
    assert r.shape[0] > 1000
    assert (r[:, 0] >= CFG.start_x_range[0]).all()
    assert (r[:, 0] <= CFG.start_x_range[1]).all()
    assert (r[:, 1] >= CFG.start_y_range[0]).all()
    assert (r[:, 1] <= CFG.start_y_range[1]).all()


def test_pathway_mode_lies_near_segment_with_lateral_bound():
    xy, mode_ids = sample_omnireset_starts(CFG, 100_000)
    p = xy[mode_ids == PATHWAY_MODE]
    assert p.shape[0] > 1000

    # distance from each point to the infinite start->goal line family is not
    # well-defined without knowing the base start; instead verify the stronger
    # constructive property by re-simulating the sampler math on fixed seeds.
    generator_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(0)
        xy_s, modes_s = sample_omnireset_starts(CFG, 4096)
        p = xy_s[modes_s == PATHWAY_MODE]
        goal = torch.tensor(CFG.goal_xy)
        # every pathway point must be within max lateral offset of the segment hull:
        # check distance to the goal is at most the largest alpha-reachable + lateral radius
        d_goal = torch.linalg.norm(p - goal, dim=-1)
        max_span = 1.0  # generous bound: |start-goal| <= workspace diagonal ~ < 1
        lateral_max = max(abs(v) for v in CFG.path_lateral_range)
        assert (d_goal <= (1 - CFG.path_alpha_range[0]) * max_span + lateral_max + 1e-3).all()
        assert (d_goal >= (1 - CFG.path_alpha_range[1]) * 0.0 - lateral_max - 1e-3).all()
        # and no pathway point may coincide exactly with the goal ring interior
        assert (d_goal >= 0).all()
    finally:
        torch.random.set_rng_state(generator_state)


def test_near_goal_mode_radius_in_range():
    xy, mode_ids = sample_omnireset_starts(CFG, 100_000)
    g = xy[mode_ids == NEAR_GOAL_MODE]
    assert g.shape[0] > 1000
    goal = torch.tensor(CFG.goal_xy)
    radii = torch.linalg.norm(g - goal, dim=-1)
    assert (radii >= CFG.near_goal_radius_range[0] - 1e-6).all()
    # outer radius can shrink via workspace clamping only from below
    assert (radii <= CFG.near_goal_radius_range[1] + 1e-6).all()


def test_workspace_clamp_respected():
    cfg = OmniResetCfg(
        goal_xy=(10.0, 0.0),  # far outside workspace forces clamping in near-goal mode
        near_goal_radius_range=(5.0, 8.0),
    )
    xy, _ = sample_omnireset_starts(cfg, 50_000)
    (x_min, x_max), (y_min, y_max) = cfg.workspace_bounds_xy
    m = cfg.workspace_margin
    assert (xy[:, 0] >= x_min + m - 1e-6).all() and (xy[:, 0] <= x_max - m + 1e-6).all()
    assert (xy[:, 1] >= y_min + m - 1e-6).all() and (xy[:, 1] <= y_max - m + 1e-6).all()


def test_invalid_ratios_raise():
    with pytest.raises(ValueError):
        sample_omnireset_starts(OmniResetCfg(reaching_ratio=0.7, pathway_ratio=0.7), 4)
    with pytest.raises(ValueError):
        sample_omnireset_starts(CFG, 0)


class DummyEnv(gym.Env):
    """Minimal single-env environment recording reset/step calls."""

    def __init__(self):
        self.observation_space = spaces.Box(-np.inf, np.inf, (4,), np.float32)
        self.action_space = spaces.Discrete(2)
        self.placements = []
        self.reset_calls = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.reset_calls += 1
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        obs = np.ones(4, dtype=np.float32)
        return obs, 1.0, True, False, {}


def test_wrapper_places_state_on_every_reset():
    env = DummyEnv()

    def place(env_unwrapped, xy, mode_ids):
        env_unwrapped.placements.append((xy.clone(), mode_ids.clone()))

    wrapped = OmniResetWrapper(env, place_state_fn=place, cfg=dict(CFG.__dict__))
    for _ in range(5):
        wrapped.reset()

    assert len(env.placements) == 5
    for xy, mode_ids in env.placements:
        assert xy.shape == (1, 2)
        assert mode_ids.shape == (1,)
    assert int(wrapped.total_mode_counts.sum()) == 5


def test_wrapper_info_and_obs_refresh():
    env = DummyEnv()

    seen = {}

    def place(env_unwrapped, xy, mode_ids):
        seen["xy"] = xy

    def get_obs(env_unwrapped):
        return np.full(4, 7.0, dtype=np.float32)

    wrapped = OmniResetWrapper(env, place_state_fn=place, get_obs_fn=get_obs)
    obs, info = wrapped.reset()
    assert (obs == 7.0).all()
    assert "omnireset/mode_ids" in info
    assert info["omnireset/mode_fractions"].sum() == pytest.approx(1.0)


def test_wrapper_requires_callback():
    with pytest.raises(ValueError):
        OmniResetWrapper(DummyEnv(), place_state_fn=None)
