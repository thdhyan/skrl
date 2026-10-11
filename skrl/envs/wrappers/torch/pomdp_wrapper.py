from __future__ import annotations

from typing import Sequence

import numpy as np
import gymnasium


class PomdpWrapper(gymnasium.Wrapper):
    """Partial-observability wrapper for gymnasium environments.

    Hides a subset of observation dimensions and optionally adds Gaussian
    noise to the visible ones.  The hidden dimensions are replaced with
    zeros in the returned observation — the agent must infer them via
    belief tracking (particle filter).

    Used to turn fully-observable gymnasium envs (CartPole, Pendulum,
    MountainCar) into POMDP benchmarks for the BeliefDQN.

    :param env: Base gymnasium environment.
    :param hidden_obs_indices: Indices of observation dimensions to zero out
        (i.e. not shown to the agent). Indices refer to the flattened observation
        (negative indices count from the end; duplicates are ignored).
        Example for CartPole: [1, 3] hides cart velocity and pole angular velocity.
    :param obs_noise_std: Gaussian noise std added to visible observations.
        Simulates sensor noise (paper uses 1 deg ≈ 0.0175 rad for encoders).
    :param seed: Optional RNG seed.
    """

    def __init__(
        self,
        env: gymnasium.Env,
        *,
        hidden_obs_indices: Sequence[int] = (),
        obs_noise_std: float = 0.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(env)
        self._noise_std = obs_noise_std
        self._rng = np.random.default_rng(seed)

        # normalize the (flat) hidden indices: negative -> positive, sorted, unique
        n = int(np.prod(env.observation_space.shape))
        hidden = set()
        for i in hidden_obs_indices:
            i = int(i)
            if not -n <= i < n:
                raise ValueError(f"Hidden observation index {i} out of range for an observation of size {n}")
            hidden.add(i % n)
        self._hidden = sorted(hidden)

        # mask: 1 = visible, 0 = hidden (same shape as the observation)
        self._mask = np.ones(n, dtype=np.float32)
        self._mask[self._hidden] = 0.0
        self._mask = self._mask.reshape(env.observation_space.shape)

        # observation space unchanged in shape (zeros for hidden dims)
        self.observation_space = env.observation_space

        # store full last obs for ground-truth logging
        self._last_full_obs: np.ndarray | None = None

    # ------------------------------------------------------------------
    # gymnasium.Wrapper overrides
    # ------------------------------------------------------------------

    def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
        obs, info = self.env.reset(**kwargs)
        self._last_full_obs = obs.copy()
        info["full_obs"] = obs.copy()
        return self._apply_mask(obs), info

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_full_obs = obs.copy()
        info["full_obs"] = obs.copy()
        return self._apply_mask(obs), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Helpers for the belief system
    # ------------------------------------------------------------------

    def get_full_obs(self) -> np.ndarray | None:
        """Return the last full (ground-truth) observation before masking."""
        return self._last_full_obs

    def _apply_mask(self, obs: np.ndarray) -> np.ndarray:
        dtype = obs.dtype
        obs = obs * self._mask
        if self._noise_std > 0.0:
            noise = self._rng.normal(0.0, self._noise_std, size=obs.shape)
            obs = obs + noise * self._mask
        return obs.astype(dtype, copy=False)

    @property
    def hidden_indices(self) -> list[int]:
        return self._hidden

    @property
    def visible_indices(self) -> list[int]:
        return [int(i) for i in np.flatnonzero(self._mask.reshape(-1))]
