from __future__ import annotations

from typing import Sequence

import torch

from skrl.belief.torch.omnireset import OmniResetCfg


def reset_object_pose_omnireset(
    env,
    env_ids: torch.Tensor,
    *,
    object_cfg,
    cfg: OmniResetCfg | None = None,
    z: float = 0.0203,
    yaw_range: tuple[float, float] = (-3.14159, 3.14159),
) -> None:
    """Isaac Lab reset event: OmniReset-style object start placement.

    Drop-in event term for manager-based environments, replacing a plain
    uniform object-reset event. Positions are drawn from the shared mixture
    sampler (:func:`skrl.belief.torch.omnireset.sample_omnireset_starts`) —
    reaching / transport-corridor / near-goal around the fixed goal — then
    written into the simulation. The goal itself is NOT touched; keep your
    existing goal-placement event.

    Usage in an ``EventTerm`` (mode ``"reset"``)::

        reset_object_omnireset = EventTerm(
            func=reset_object_pose_omnireset,
            mode="reset",
            params={
                "object_cfg": SceneEntityCfg("red_cube"),
                "cfg": OmniResetCfg(goal_xy=(0.62, 0.0)),
                "z": 0.0203,
            },
        )

    The sampled mode ids are stored on the environment as
    ``env._skrl_omnireset_mode_ids`` for logging/visualization.

    This module imports Isaac Lab lazily inside the call so that ``skrl``
    remains importable without an Isaac Lab installation.

    :param env: The vectorized environment (manager-based).
    :param env_ids: Environment ids being reset.
    :param object_cfg: ``SceneEntityCfg`` of the rigid object to place.
    :param cfg: Mixture configuration (defaults tuned for the source
        repository's Franka cube push task).
    :param z: Fixed object height at placement.
    :param yaw_range: Uniform yaw range applied to the placed object.
    """
    from isaaclab.utils.math import quat_from_euler_xyz

    if env_ids is None or len(env_ids) == 0:
        return

    mixture_cfg = cfg if cfg is not None else OmniResetCfg()
    xy, mode_ids = sample_omnireset_starts(mixture_cfg, len(env_ids), device=env.device)

    obj = env.scene[object_cfg.name]
    origins = env.scene.env_origins[env_ids]
    num = len(env_ids)
    device = env.device

    pos = torch.cat(
        (
            origins[:, :2] + xy.to(device),
            torch.full((num, 1), z, device=device),
        ),
        dim=-1,
    )
    yaw = torch.empty(num, device=device).uniform_(*yaw_range)
    quat = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
    obj.write_root_pose_to_sim(torch.cat((pos, quat), dim=-1), env_ids=env_ids)
    obj.write_root_velocity_to_sim(torch.zeros(num, 6, device=device), env_ids=env_ids)

    setattr(env, "_skrl_omnireset_mode_ids", mode_ids.detach().clone())


def omnireset_mode_fractions(env) -> torch.Tensor:
    """Return per-mode fractions sampled at the last reset (for logging).

    :param env: Environment previously reset through
        :func:`reset_object_pose_omnireset`.
    :return: Fractions ``[3]`` for (reaching, pathway, near-goal), or zeros
        before the first reset.
    """
    mode_ids: Sequence[int] | None = getattr(env, "_skrl_omnireset_mode_ids", None)
    if mode_ids is None:
        return torch.zeros(3)
    ids = torch.as_tensor(mode_ids, dtype=torch.long).reshape(-1)
    return torch.bincount(ids.cpu(), minlength=3).float() / max(1, ids.numel())
