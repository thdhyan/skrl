import pytest

import gymnasium
import numpy as np
import torch

from skrl.envs.wrappers.torch import wrap_env
from skrl.envs.wrappers.torch.pomdp_wrapper import PomdpWrapper


class _BoxEnv(gymnasium.Env):
    """Deterministic env with a multi-dimensional observation (2 x 3) for masking checks."""

    def __init__(self, shape=(2, 3)):
        self.observation_space = gymnasium.spaces.Box(-10.0, 10.0, shape=shape, dtype=np.float32)
        self.action_space = gymnasium.spaces.Discrete(2)
        self._t = 0

    def _obs(self):
        size = int(np.prod(self.observation_space.shape))
        return (np.arange(size, dtype=np.float32) + 1.0 + self._t).reshape(self.observation_space.shape)

    def reset(self, *, seed=None, options=None):
        self._t = 0
        return self._obs(), {}

    def step(self, action):
        self._t += 1
        return self._obs(), 1.0, False, self._t >= 5, {}


def test_cartpole_hidden_dims_are_zero_and_visible_dims_untouched():
    env = PomdpWrapper(gymnasium.make("CartPole-v1"), hidden_obs_indices=[1, 3])
    obs, info = env.reset(seed=0)
    full = env.get_full_obs()
    assert np.all(obs[[1, 3]] == 0.0)
    np.testing.assert_array_equal(obs[[0, 2]], full[[0, 2]])
    np.testing.assert_array_equal(info["full_obs"], full)
    assert env.hidden_indices == [1, 3] and env.visible_indices == [0, 2]
    for _ in range(10):
        obs, _, terminated, truncated, info = env.step(env.action_space.sample())
        assert np.all(obs[[1, 3]] == 0.0)
        np.testing.assert_array_equal(obs[[0, 2]], info["full_obs"][[0, 2]])
        assert obs.dtype == np.float32 and obs.shape == env.observation_space.shape
        if terminated or truncated:
            break


def test_noise_only_on_visible_dims():
    env = PomdpWrapper(gymnasium.make("CartPole-v1"), hidden_obs_indices=[1, 3], obs_noise_std=0.1, seed=0)
    obs, _ = env.reset(seed=0)
    diffs = [obs - env.get_full_obs()]
    for _ in range(20):
        obs, _, terminated, truncated, _ = env.step(0)
        diffs.append(obs - env.get_full_obs() * np.array([1, 0, 1, 0], dtype=np.float32))
        if terminated or truncated:
            break
    diffs = np.stack(diffs[1:])
    assert np.all(diffs[:, [1, 3]] == 0.0)  # hidden dims: exactly zero, no noise leaked
    assert np.all(np.abs(diffs[:, [0, 2]]) > 0.0)  # visible dims: noisy
    assert 0.03 < diffs[:, [0, 2]].std() < 0.3


def test_full_observability_is_identity():
    env = PomdpWrapper(gymnasium.make("CartPole-v1"))
    obs, _ = env.reset(seed=1)
    np.testing.assert_array_equal(obs, env.get_full_obs())


def test_multidimensional_observation_mask_uses_flat_indices():
    env = PomdpWrapper(_BoxEnv(), hidden_obs_indices=[1, 5])
    obs, _ = env.reset()
    assert obs.shape == (2, 3)
    expected = np.arange(1.0, 7.0, dtype=np.float32)
    expected[[1, 5]] = 0.0
    np.testing.assert_array_equal(obs.reshape(-1), expected)
    assert env.visible_indices == [0, 2, 3, 4]


@pytest.mark.parametrize("indices", [[4], [-5]])
def test_out_of_range_hidden_indices_raise(indices):
    with pytest.raises(ValueError):
        PomdpWrapper(gymnasium.make("CartPole-v1"), hidden_obs_indices=indices)


def test_negative_and_duplicate_indices_are_normalized():
    env = PomdpWrapper(gymnasium.make("CartPole-v1"), hidden_obs_indices=[-1, 3, 1, 1])
    assert env.hidden_indices == [1, 3] and env.visible_indices == [0, 2]
    obs, _ = env.reset(seed=0)
    assert np.all(obs[[1, 3]] == 0.0)


def test_masked_observations_reach_the_skrl_agent_tensors():
    env = wrap_env(PomdpWrapper(gymnasium.make("CartPole-v1"), hidden_obs_indices=[1, 3]), verbose=False)
    observations, _ = env.reset()
    assert observations.shape == (1, 4)
    assert (observations[:, [1, 3]] == 0).all()
    observations, *_ = env.step(torch.zeros((1, 1), dtype=torch.long))
    assert (observations[:, [1, 3]] == 0).all()
    env.close()
