from __future__ import annotations

from typing import Callable

import collections

import numpy as np
import gymnasium


class PrivilegedHistoryWrapper(gymnasium.Wrapper):
    """Packs observation history, privileged values and a mask flag into one flat observation.

    Layout of the returned observation (see :class:`skrl.agents.torch.privileged_ppo.ObsLayout`)::

        [ o_t (D) | o_{t-H} .. o_{t-1} (H*D) | privileged (P) | mask (1) ]

    The privileged values are always present, so an asymmetric critic can use them. The mask flag
    (1 = masked) tells the actor to ignore them and use its history estimator instead. The flag is
    drawn once per episode with probability ``curriculum.mask_probability()``, so the stored rollout
    is consistent between ``act()`` and the PPO update.

    The curriculum object is shared with the agent, which advances it. This only works when the
    wrapper runs in the same process (single env or ``SyncVectorEnv``).

    :param env: Environment whose observation is the (partial) observation ``o_t``.
    :param privileged_fn: ``privileged_fn(env) -> np.ndarray[P]``, called after every reset and step
        with the wrapped env (e.g. hidden velocities, randomized masses).
    :param privileged_dim: ``P``.
    :param history_length: ``H``, number of past observations to stack (zero-padded at episode start).
    :param curriculum: Object with a ``mask_probability()`` method. ``None`` never masks.
    :param randomize_fn: Optional ``randomize_fn(env, rng)`` called before every reset (domain randomization).
    :param seed: RNG seed for the mask draws and ``randomize_fn``.
    :param include_last_action: Append the previous action ``a_{t-1}`` (zeros at episode start) to ``o_t``, so the
        history holds observation-action pairs and the estimator can identify dynamics (e.g. mass from the
        response to a known force). ``D`` then includes the action size. Box action spaces only.
    """

    def __init__(
        self,
        env: gymnasium.Env,
        *,
        privileged_fn: Callable[[gymnasium.Env], np.ndarray],
        privileged_dim: int,
        history_length: int,
        curriculum=None,
        randomize_fn: Callable[[gymnasium.Env, np.random.Generator], None] | None = None,
        seed: int | None = None,
        include_last_action: bool = False,
    ) -> None:
        super().__init__(env)
        self._privileged_fn = privileged_fn
        self._privileged_dim = privileged_dim
        self._curriculum = curriculum
        self._randomize_fn = randomize_fn
        self._rng = np.random.default_rng(seed)

        self._obs_dim = int(np.prod(env.observation_space.shape))
        self._last_action = None
        if include_last_action:
            if not isinstance(env.action_space, gymnasium.spaces.Box):
                raise ValueError("include_last_action needs a Box action space")
            self._last_action = np.zeros(int(np.prod(env.action_space.shape)), dtype=np.float32)
            self._obs_dim += self._last_action.size
        self._history = collections.deque(maxlen=history_length)
        self._history_length = history_length
        self._mask = 0.0

        size = self._obs_dim * (1 + history_length) + privileged_dim + 1
        self.observation_space = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(size,), dtype=np.float32)

    def reset(self, **kwargs):
        if self._randomize_fn is not None:
            self._randomize_fn(self.env, self._rng)
        obs, info = self.env.reset(**kwargs)
        if self._last_action is not None:
            self._last_action[:] = 0.0
        p = self._curriculum.mask_probability() if self._curriculum is not None else 0.0
        self._mask = float(self._rng.random() < p)
        self._history.clear()
        self._history.extend(np.zeros(self._obs_dim, dtype=np.float32) for _ in range(self._history_length))
        info["privileged_mask"] = self._mask
        return self._pack(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self._last_action is not None:
            self._last_action[:] = np.asarray(action, dtype=np.float32).reshape(-1)
        packed = self._pack(obs)
        info["privileged_mask"] = self._mask
        return packed, reward, terminated, truncated, info

    def _pack(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        if self._last_action is not None:
            obs = np.concatenate([obs, self._last_action])
        privileged = np.asarray(self._privileged_fn(self.env), dtype=np.float32).reshape(self._privileged_dim)
        packed = np.concatenate([obs, *self._history, privileged, [self._mask]]).astype(np.float32)
        if self._history_length:
            self._history.append(obs)  # history for the next step: o_{t-H+1} .. o_t
        return packed
