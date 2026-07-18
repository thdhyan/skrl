from __future__ import annotations

import math
from typing import Optional

import torch

from skrl.belief.backends.base import DynamicsBackend


class MjlabBackend(DynamicsBackend):
    """mjlab backend for the particle filter.

    mjlab wraps MuJoCo with a Gym-compatible interface and vectorised
    env support.  We exploit that to run NP parallel rollouts — one per
    particle — using the same model.

    :param env_id: mjlab environment ID (e.g. 'mjlab/PegInHole-v1').
    :param n_joints: Number of robot joints n_r.
    :param n_obj_dof: Object free-joint DoF (default 6).
    :param num_particles: NP — number of parallel envs to create.
    :param sigma_enc_deg: Encoder noise std in degrees.
    :param device: Torch device.
    :param seed: Optional RNG seed.
    """

    def __init__(
        self,
        *,
        env_id: str,
        n_joints: int,
        n_obj_dof: int = 6,
        num_particles: int = 300,
        sigma_enc_deg: float = 1.0,
        device: str | torch.device = "cpu",
        seed: Optional[int] = None,
    ) -> None:
        try:
            import mjlab
        except ImportError as e:
            raise ImportError(
                "MjlabBackend requires 'mjlab'. Install from your mjlab source."
            ) from e

        self._n_joints = n_joints
        self._n_obj_dof = n_obj_dof
        self._sigma_enc = math.radians(sigma_enc_deg)
        self._device = torch.device(device)
        self._NP = num_particles
        self._state_dim = 2 * n_joints + 2 * n_obj_dof

        # vectorised env: NP envs for particles + 1 canonical for ground truth
        self._venv = mjlab.make_vec(env_id, num_envs=num_particles, seed=seed)
        self._gt_env = mjlab.make(env_id, seed=seed)
        self._last_gt_obs: Optional[torch.Tensor] = None

    @property
    def state_dim(self) -> int:
        return self._state_dim

    def _obs_to_state(self, qpos: torch.Tensor, qvel: torch.Tensor) -> torch.Tensor:
        """Pack qpos/qvel into state layout [q_r, qd_r, q_obj, qd_obj]."""
        return torch.cat([
            qpos[:, :self._n_joints],
            qvel[:, :self._n_joints],
            qpos[:, self._n_joints:self._n_joints + self._n_obj_dof],
            qvel[:, self._n_joints:self._n_joints + self._n_obj_dof],
        ], dim=-1)

    def step(
        self,
        particles: torch.Tensor,
        control: torch.Tensor,
        n_substeps: int = 1,
    ) -> torch.Tensor:
        """Drive all NP envs from particles for n_substeps."""
        # set state of all envs
        qpos = torch.cat([
            particles[:, :self._n_joints],
            particles[:, 2 * self._n_joints:2 * self._n_joints + self._n_obj_dof],
        ], dim=-1).cpu().numpy()
        qvel = torch.cat([
            particles[:, self._n_joints:2 * self._n_joints],
            particles[:, 2 * self._n_joints + self._n_obj_dof:],
        ], dim=-1).cpu().numpy()
        self._venv.set_state(qpos, qvel)

        action = control[:self._n_joints].cpu().numpy()
        for _ in range(n_substeps):
            obs, *_ = self._venv.step_with_action(action)

        qpos_t = torch.tensor(self._venv.get_qpos(), dtype=torch.float32)
        qvel_t = torch.tensor(self._venv.get_qvel(), dtype=torch.float32)
        return self._obs_to_state(qpos_t, qvel_t).to(self._device)

    def get_measurement(
        self,
        particles: torch.Tensor,
        sigma_enc: float = 1.0,
    ) -> torch.Tensor:
        sigma = math.radians(sigma_enc)
        z = self._last_gt_obs[:self._n_joints] if self._last_gt_obs is not None else torch.zeros(self._n_joints)
        z = z.to(self._device)
        q_r = particles[:, :self._n_joints]
        diff = q_r - z.unsqueeze(0)
        log_ll = -0.5 * (diff / sigma).pow(2).sum(-1)
        log_ll -= self._n_joints * math.log(sigma * math.sqrt(2 * math.pi))
        return log_ll

    def get_object_pose(self) -> torch.Tensor:
        import numpy as np
        qpos = self._gt_env.get_qpos()
        pos = qpos[self._n_joints:self._n_joints + 3]
        quat_wxyz = qpos[self._n_joints + 3:self._n_joints + 7]
        pose = torch.tensor([*pos, quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=torch.float32)
        return pose.to(self._device)

    def get_robot_state(self) -> torch.Tensor:
        import numpy as np
        qpos = torch.tensor(self._gt_env.get_qpos()[:self._n_joints], dtype=torch.float32)
        qvel = torch.tensor(self._gt_env.get_qvel()[:self._n_joints], dtype=torch.float32)
        return torch.cat([qpos, qvel]).to(self._device)

    def reset(self, seed: Optional[int] = None) -> torch.Tensor:
        obs, _ = self._gt_env.reset(seed=seed)
        self._venv.reset()
        self._last_gt_obs = torch.tensor(obs, dtype=torch.float32, device=self._device)
        return self._last_gt_obs[:self._n_joints]
