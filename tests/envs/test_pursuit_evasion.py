import pytest

import torch

from skrl.envs.pursuit_evasion import (
    OBSERVATION_BUILDERS,
    ObservationBuilder,
    PursuitEvasionCfg,
    PursuitEvasionEnv,
    register_observation,
)
from skrl.envs.pursuit_evasion.potential_fields import compute_agent_fields, grid_coordinates, khatib_profile
from skrl.envs.wrappers.torch import MultiAgentEnvWrapper, wrap_env

OBS_MODES = ["full_state", "global_potential", "local_potential"]


def make_env(**kwargs):
    cfg = dict(num_envs=4, height=8, width=9, num_pursuers=3, max_steps=20, device="cpu", seed=0)
    cfg.update(kwargs)
    return PursuitEvasionEnv(**cfg)


def expected_obs_size(env: PursuitEvasionEnv) -> int:
    a, h, w = env.num_agents, env.height, env.width
    mode = env.cfg.obs_mode
    if mode == "full_state":
        return 5 * a + 3 + (h * w if env.obstacles.any() else 0)
    if mode == "global_potential":
        return 5 * h * w
    k = 2 * env.cfg.local_radius + 1
    return 4 * k * k + 4


@pytest.mark.parametrize("obs_mode", OBS_MODES)
@pytest.mark.parametrize("obstacle_density", [0.0, 0.15])
def test_reset_step_shapes(obs_mode, obstacle_density):
    env = wrap_env(make_env(obs_mode=obs_mode, obstacle_density=obstacle_density), wrapper="pursuit-evasion")
    assert isinstance(env, MultiAgentEnvWrapper)
    assert env.possible_agents == ["pursuer_0", "pursuer_1", "pursuer_2", "evader"]
    assert env.num_envs == 4 and env.max_num_agents == 4
    size = expected_obs_size(env._unwrapped)

    observations, infos = env.reset()
    states = env.state()
    for uid in env.possible_agents:
        assert env.observation_space(uid).shape == (size,)
        assert env.action_space(uid).n == 5
        assert observations[uid].shape == (4, size)
        assert states[uid].shape == (4, env.state_space(uid).shape[0])

    for _ in range(50):
        actions = {uid: torch.randint(5, (4, 1)) for uid in env.possible_agents}
        observations, rewards, terminated, truncated, infos = env.step(actions)
        for uid in env.possible_agents:
            assert observations[uid].shape == (4, size)
            assert observations[uid].dtype == torch.float32
            assert observations[uid].min() >= -1 and observations[uid].max() <= 1
            assert rewards[uid].shape == terminated[uid].shape == truncated[uid].shape == (4, 1)
            assert terminated[uid].dtype == truncated[uid].dtype == torch.bool
        # zero-sum, shared team reward
        assert torch.allclose(rewards["pursuer_0"], rewards["pursuer_2"])
        assert torch.allclose(rewards["pursuer_0"], -rewards["evader"])
        assert not (terminated["evader"] & truncated["evader"]).any()
        # agents never stand on obstacles nor leave the grid
        positions = env._unwrapped.positions
        assert positions.min() >= 0 and (positions[..., 0] < 8).all() and (positions[..., 1] < 9).all()
        assert not env._unwrapped.obstacles[positions[..., 0], positions[..., 1]].any()


def test_image_shape():
    env = make_env(obs_mode="global_potential")
    assert env.observation_image_shape == (5, 8, 9)
    env = make_env(obs_mode="local_potential", local_radius=3)
    assert env.observation_image_shape == (4, 7, 7)
    assert make_env(obs_mode="full_state").observation_image_shape is None


def test_capture_and_truncation():
    env = make_env(num_envs=2, num_pursuers=1, capture_radius=0, max_steps=5)
    env.reset()
    # env 0: pursuer moves right onto the evader -> capture; env 1: far apart -> truncation after max_steps
    env.positions[0] = torch.tensor([[0, 0], [0, 1]])
    env.positions[1] = torch.tensor([[0, 0], [7, 8]])
    stay = torch.zeros((2, 1), dtype=torch.long)
    actions = {"pursuer_0": torch.tensor([[4], [0]]), "evader": stay}
    _, rewards, terminated, truncated, infos = env.step(actions)
    assert terminated["evader"].view(-1).tolist() == [True, False]
    assert rewards["pursuer_0"][0, 0] > env.cfg.capture_reward - 1
    for _ in range(4):
        _, _, terminated, truncated, infos = env.step({"pursuer_0": stay, "evader": stay})
    assert truncated["evader"].view(-1).tolist() == [False, True]


def test_swap_counts_as_capture():
    env = make_env(num_envs=1, num_pursuers=1, capture_radius=0)
    env.reset()
    env.positions[0] = torch.tensor([[3, 3], [3, 4]])
    actions = {"pursuer_0": torch.tensor([[4]]), "evader": torch.tensor([[3]])}  # right / left: pass through
    _, _, terminated, _, _ = env.step(actions)
    assert terminated["evader"].item()


def test_obstacles_block_movement():
    env = make_env(num_envs=1, num_pursuers=1, obstacles=[(2, 3)], spawn_min_distance=1)
    env.reset()
    env.positions[0] = torch.tensor([[2, 2], [7, 8]])
    env.step({"pursuer_0": torch.tensor([[4]]), "evader": torch.tensor([[0]])})
    assert env.positions[0, 0].tolist() == [2, 2]


def test_attractive_field_minimized_at_evader():
    env = make_env(num_envs=3, obs_mode="global_potential")
    env.reset()
    fields = env.potential_fields()
    evader = env.positions[:, -1]
    h, w = env.height, env.width
    for e in range(env.num_envs):
        for i in range(env.num_pursuers):
            att = fields["attractive"][e, i]
            flat_argmin = int(att.view(-1).argmin())
            assert divmod(flat_argmin, w) == tuple(evader[e].tolist())
            assert att[evader[e, 0], evader[e, 1]] == 0
            assert (att >= 0).all()
    # the evader's channel of the self-position plane is set on its own cell
    obs = env._observations()["evader"].view(env.num_envs, 5, h, w)
    for e in range(env.num_envs):
        assert obs[e, 4].sum() == 1 and obs[e, 4, evader[e, 0], evader[e, 1]] == 1


def test_repulsive_fields():
    # Khatib profile: maximal at the source, zero beyond d0, monotonically decreasing
    d = torch.linspace(0, 5, 51)
    g = khatib_profile(d, d0=3.0)
    assert g[0] == 1 and (g[d >= 3.0] == 0).all() and (g[1:] <= g[:-1]).all()

    coords = grid_coordinates(9, 9)
    pursuers = torch.tensor([[[4, 4], [4, 5]]])
    evader = torch.tensor([[0, 0]])
    fields = compute_agent_fields(
        pursuers=pursuers,
        evader=evader,
        coords=coords,
        obstacle_potential=torch.zeros(9, 9),
        k_att=1.0,
        k_pp=0.5,
        d0_pp=2.5,
        k_ep=1.0,
        d0_ep=4.0,
        k_center=0.0,
    )
    # evader's repulsive field peaks at the pursuers and vanishes far away
    rep_e = fields["repulsive"][0, -1]
    assert rep_e[4, 4] == rep_e.max() and rep_e[0, 0] == 0
    # pursuer 0 is repelled by pursuer 1 but not by itself
    rep_0 = fields["repulsive"][0, 0]
    assert rep_0[4, 5] == rep_0.max() and rep_0[4, 5] > rep_0[4, 4]
    assert torch.allclose(fields["total"], fields["attractive"] + fields["repulsive"] + fields["obstacle"])


def test_local_window_is_egocentric():
    env = make_env(num_envs=2, obs_mode="local_potential", local_radius=1)
    env.reset()
    fields = env.potential_fields()
    obs = env._observations()
    k = 3
    for idx, uid in enumerate(env.possible_agents):
        window = obs[uid][:, : 4 * k * k].view(2, 4, k, k)
        for e in range(2):
            r, c = env.positions[e, idx].tolist()
            centre_total = (fields["total"][e, idx, r, c] / env.cfg.potential_clip).clamp(0, 1)
            assert torch.isclose(window[e, 0, 1, 1], centre_total)
            # out-of-bounds cells are flagged as obstacles
            if r == 0:
                assert (window[e, 3, 0] == 1).all()


def test_observation_registry_is_pluggable():
    name = "_test_positions_only"
    if name not in OBSERVATION_BUILDERS:

        @register_observation(name)
        class PositionsOnly(ObservationBuilder):
            def size(self):
                return 2

            def build(self):
                return self.normalized_positions()

    env = make_env(obs_mode=name)
    observations, _ = env.reset()
    assert observations["evader"].shape == (4, 2)
    with pytest.raises(KeyError):
        make_env(obs_mode="does_not_exist")


def test_cfg_object_and_render():
    env = PursuitEvasionEnv(PursuitEvasionCfg(num_envs=2, height=5, width=6, num_pursuers=2, device="cpu", seed=1))
    env.reset()
    text = env.render()
    assert len(text.splitlines()) == 5 and "E" in text
