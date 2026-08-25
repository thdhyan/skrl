from __future__ import annotations

from typing import Callable, Sequence

import gymnasium
import torch

from skrl.belief.torch.omnireset import OmniResetCfg, sample_omnireset_starts


class OmniResetWrapper(gymnasium.Wrapper):
    """OmniReset-style diverse episode starts for gymnasium environments.

    On every :meth:`reset` the environment resets normally, then start-state
    positions are drawn from an OmniReset mixture (reaching / transport
    corridor / near-goal — see :class:`skrl.belief.torch.omnireset.OmniResetCfg`)
    and handed to a user callback that writes them into the simulator.

    This wrapper only sees resets performed through the Python API. Environments
    that auto-reset internally (e.g. vectorized Isaac Lab environments) must use
    the engine's reset-event mechanism instead — see
    ``skrl/belief/torch/isaaclab_omnireset.py``.

    :param env: Base gymnasium environment.
    :param place_state_fn: Callback ``place_state_fn(env_unwrapped, xy, mode_ids)``
        writing the sampled env-local XY starts into the task state (engine
        specific). For vectorized environments ``xy`` has shape
        ``[num_envs, 2]``; otherwise ``[1, 2]``.
    :param cfg: Mixture configuration. Defaults are tuned for the Franka cube
        push task from the source repository.
    :param get_obs_fn: Optional callback ``get_obs_fn(env_unwrapped) -> obs``
        re-reading the observation after placement (e.g. after teleporting an
        object). When omitted, the post-reset observation is returned unchanged.
    :param seed: Unused placeholder for API symmetry with other wrappers;
        sampling uses torch's global RNG (seed it via
        ``skrl.utils.set_seed``).

    Example::

        >>> def place(env_unwrapped, xy, mode_ids):
        ...     cube = env_unwrapped.scene["red_cube"]
        ...     pos = torch.cat((env_unwrapped.scene.env_origins[:, :2] + xy,
        ...                      torch.full((xy.shape[0], 1), 0.0203, device=xy.device)), dim=-1)
        ...     cube.write_root_pose_to_sim(torch.cat((pos, default_quat(pos.shape[0], xy.device)), dim=-1))
        >>>
        >>> env = OmniResetWrapper(env, place_state_fn=place)
    """

    def __init__(
        self,
        env: gymnasium.Env,
        *,
        place_state_fn: Callable[[gymnasium.Env, torch.Tensor, torch.Tensor], None],
        cfg: OmniResetCfg | dict | None = None,
        get_obs_fn: Callable[[gymnasium.Env], object] | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(env)
        if place_state_fn is None:
            raise ValueError("OmniResetWrapper requires a place_state_fn callback")
        if isinstance(cfg, dict):
            self._cfg = OmniResetCfg(**cfg)
        else:
            self._cfg = cfg if cfg is not None else OmniResetCfg()
        self._place_state_fn = place_state_fn
        self._get_obs_fn = get_obs_fn
        self._seed = seed
        self._total_mode_counts = torch.zeros(3, dtype=torch.long)

    @property
    def omnireset_cfg(self) -> OmniResetCfg:
        """Active mixture configuration."""
        return self._cfg

    @property
    def total_mode_counts(self) -> torch.Tensor:
        """Cumulative per-mode sample counts across all resets."""
        return self._total_mode_counts.clone()

    def _num_envs(self) -> int:
        return int(getattr(self.env.unwrapped, "num_envs", 1))

    def _device(self) -> torch.device:
        device = getattr(self.env.unwrapped, "device", None)
        try:
            return torch.device(device) if device is not None else torch.device("cpu")
        except Exception:
            return torch.device("cpu")

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        num_envs = self._num_envs()
        xy, mode_ids = sample_omnireset_starts(self._cfg, num_envs, device=self._device())
        self._place_state_fn(self.env.unwrapped, xy, mode_ids)

        self._total_mode_counts += torch.bincount(mode_ids.cpu(), minlength=3)

        info["omnireset/mode_ids"] = mode_ids.cpu()
        counts = torch.bincount(mode_ids.cpu(), minlength=3).float() / max(1, num_envs)
        info["omnireset/mode_fractions"] = counts

        if self._get_obs_fn is not None:
            obs = self._get_obs_fn(self.env.unwrapped)
        return obs, info
