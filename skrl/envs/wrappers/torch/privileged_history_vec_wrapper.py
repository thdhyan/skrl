from __future__ import annotations

from typing import Any, Callable

import gymnasium
import numpy as np

import torch

from skrl.envs.wrappers.torch.base import Wrapper


class PrivilegedHistoryVecWrapper(Wrapper):
    """Batched (GPU) counterpart of :class:`PrivilegedHistoryWrapper` for vectorized skrl environments (Isaac Lab).

    Packs, per env, the observation history, the privileged values and a mask flag into one flat observation::

        [ o_t (D) | o_{t-H} .. o_{t-1} (H*D) | privileged (P) | mask (1) ]

    (see :class:`skrl.agents.torch.privileged_ppo.ObsLayout`). The mask flag of each env is drawn with probability
    ``curriculum.mask_probability()`` at the start of each of its episodes. The wrapped environment must reset done
    envs in the same step (Isaac Lab auto-reset): the observation returned for a done env is the first one of its
    next episode, so its history and last action are cleared and a new mask is drawn.

    :param env: skrl-wrapped vectorized environment (e.g. ``IsaacLabWrapper``).
    :param history_length: ``H``, number of past observations to stack (zeros at episode start).
    :param privileged_fn: ``privileged_fn(env) -> Tensor[num_envs, P]``, called after every reset and step with the
        skrl-wrapped env. ``None``: the env state (``env.state()``, e.g. the Isaac Lab ``"critic"`` group).
    :param privileged_indices: Subset of the dimensions of the privileged values to keep. ``None``: all.
    :param curriculum: Object with a ``mask_probability()`` method, shared with the agent. ``None`` never masks.
    :param include_last_action: Append the previous action ``a_{t-1}`` to ``o_t`` (Box action spaces only).
    :param seed: Seed of the mask draws.
    """

    def __init__(
        self,
        env: Wrapper,
        *,
        history_length: int,
        privileged_fn: Callable[[Wrapper], torch.Tensor] | None = None,
        privileged_indices: list[int] | None = None,
        curriculum=None,
        include_last_action: bool = False,
        seed: int | None = None,
    ) -> None:
        super().__init__(env)
        self._device = env.device
        self._history_length = history_length
        self._privileged_fn = privileged_fn or (lambda e: e.state())
        self._privileged_indices = privileged_indices
        self._curriculum = curriculum
        self._generator = torch.Generator(device=self._device)
        if seed is not None:
            self._generator.manual_seed(seed)

        self._obs_dim = int(np.prod(env.observation_space.shape))
        self._action_dim = 0
        if include_last_action:
            if not isinstance(env.action_space, gymnasium.spaces.Box):
                raise ValueError("include_last_action needs a Box action space")
            self._action_dim = int(np.prod(env.action_space.shape))
        self._privileged_dim = None  # known after the first reset (from the privileged values)
        self._history = self._last_action = self._mask = None

    # --- spaces and layout ------------------------------------------------------------------------------------

    @property
    def obs_dim(self) -> int:
        """``D``: observation size, including the last action if appended."""
        return self._obs_dim + self._action_dim

    @property
    def privileged_dim(self) -> int:
        if self._privileged_dim is None:
            self.reset()
        return self._privileged_dim

    @property
    def history_length(self) -> int:
        return self._history_length

    @property
    def layout(self):
        from skrl.agents.torch.privileged_ppo.models import ObsLayout

        return ObsLayout(obs_dim=self.obs_dim, history_length=self._history_length, privileged_dim=self.privileged_dim)

    @property
    def observation_space(self) -> gymnasium.Space:
        size = self.obs_dim * (1 + self._history_length) + self.privileged_dim + 1
        return gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(size,), dtype=np.float32)

    @property
    def state_space(self) -> gymnasium.Space | None:
        return self._env.state_space

    @property
    def action_space(self) -> gymnasium.Space:
        return self._env.action_space

    @property
    def num_envs(self) -> int:
        return self._env.num_envs

    # --- env interface ----------------------------------------------------------------------------------------

    def reset(self) -> tuple[torch.Tensor, Any]:
        observations, info = self._env.reset()
        if self._history is None:  # Isaac Lab resets only once: start every env's episode here
            n = observations.shape[0]
            privileged = self._privileged()
            self._privileged_dim = privileged.shape[-1]
            self._history = torch.zeros(n, self._history_length, self.obs_dim, device=self._device)
            self._last_action = torch.zeros(n, self._action_dim, device=self._device)
            self._mask = torch.zeros(n, 1, device=self._device)
            self._new_episodes(torch.ones(n, dtype=torch.bool, device=self._device))
            self._packed = self._pack(observations, privileged)
        return self._packed, info

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        observations, rewards, terminated, truncated, info = self._env.step(actions)
        if self._action_dim:
            self._last_action = actions.reshape(actions.shape[0], -1).to(self._last_action)
        done = (terminated | truncated).view(-1)
        self._new_episodes(done)
        self._packed = self._pack(observations, self._privileged())
        return self._packed, rewards, terminated, truncated, info

    def state(self) -> torch.Tensor | None:
        return self._env.state()

    def render(self, *args, **kwargs) -> Any:
        return self._env.render(*args, **kwargs)

    def close(self) -> None:
        self._env.close()

    # --- internals --------------------------------------------------------------------------------------------

    def _privileged(self) -> torch.Tensor:
        privileged = self._privileged_fn(self._env)
        if privileged is None:
            raise ValueError("No privileged values: the env has no state (e.g. Isaac Lab 'critic' group); set privileged_fn")
        privileged = privileged.reshape(privileged.shape[0], -1).to(device=self._device, dtype=torch.float32)
        return privileged if self._privileged_indices is None else privileged[:, self._privileged_indices]

    def _new_episodes(self, envs: torch.Tensor) -> None:
        """Clear the history and last action of ``envs`` (bool [N]) and draw their mask flags."""
        if not envs.any():
            return
        self._history[envs] = 0.0
        self._last_action[envs] = 0.0
        p = self._curriculum.mask_probability() if self._curriculum is not None else 0.0
        draw = torch.rand(envs.shape[0], 1, device=self._device, generator=self._generator) < p
        self._mask[envs] = draw[envs].float()

    def _pack(self, observations: torch.Tensor, privileged: torch.Tensor) -> torch.Tensor:
        obs = observations.reshape(observations.shape[0], -1).float()
        if self._action_dim:
            obs = torch.cat([obs, self._last_action], dim=-1)
        packed = torch.cat([obs, self._history.flatten(1), privileged, self._mask], dim=-1)
        if self._history_length:  # history for the next step: o_{t-H+1} .. o_t
            self._history = torch.cat([self._history[:, 1:], obs[:, None]], dim=1)
        return packed
