from __future__ import annotations

import math
from typing import Optional

import torch

from skrl.belief.backends.base import DynamicsBackend


class MuJoCoBackend(DynamicsBackend):
    """MuJoCo physics backend for the particle filter.

    Wraps a MuJoCo model and implements the impedance control law from
    paper eq. 3-4.  Each particle gets its own mjData so that NP rollouts
    run in parallel without state aliasing.

    :param model_path: Path to the MuJoCo XML/MJCF model file.
    :param n_joints: Number of robot joints n_r.
    :param n_obj_dof: Object DoF count (6 for free joint).
    :param dt: Physics substep size in seconds (paper uses 0.005 s = 200 Hz).
    :param K_stiffness: Joint stiffness diagonal ``[n_joints]``.
        Paper: 200 Nm/rad revolute, 800 N/m prismatic.
    :param D_damping: Joint damping diagonal ``[n_joints]``.
        Set automatically via critical-damping design if None.
    :param sigma_enc_deg: Encoder noise std in degrees (paper: 1.0 deg).
    :param ou_sigma: Ornstein-Uhlenbeck noise std on object torques.
    :param ou_theta: OU mean-reversion rate.
    :param device: Torch device (CPU only for MuJoCo; use stub on GPU).
    :param num_particles: NP used to pre-allocate mjData pool.
    :param seed: Optional RNG seed.
    """

    def __init__(
        self,
        *,
        model_path: str,
        n_joints: int,
        n_obj_dof: int = 6,
        dt: float = 0.005,
        K_stiffness: Optional[torch.Tensor] = None,
        D_damping: Optional[torch.Tensor] = None,
        sigma_enc_deg: float = 1.0,
        ou_sigma: float = 0.1,
        ou_theta: float = 0.15,
        device: str | torch.device = "cpu",
        num_particles: int = 300,
        seed: Optional[int] = None,
    ) -> None:
        try:
            import mujoco
        except ImportError as e:
            raise ImportError(
                "MuJoCoBackend requires the 'mujoco' Python package. "
                "Install via: pip install mujoco"
            ) from e

        self._mujoco = mujoco
        self._n_joints = n_joints
        self._n_obj_dof = n_obj_dof
        self._dt = dt
        self._sigma_enc = math.radians(sigma_enc_deg)
        self._ou_sigma = ou_sigma
        self._ou_theta = ou_theta
        self._device = torch.device(device)
        self._NP = num_particles

        # state layout: q_r, qd_r, q_obj, qd_obj
        self._state_dim = 2 * n_joints + 2 * n_obj_dof

        # load model once; create NP data objects
        self._model = mujoco.MjModel.from_xml_path(model_path)
        self._model.opt.timestep = dt
        self._datas = [mujoco.MjData(self._model) for _ in range(num_particles)]
        self._gt_data = mujoco.MjData(self._model)

        # stiffness / damping
        import numpy as np
        if K_stiffness is None:
            K_stiffness = torch.ones(n_joints) * 200.0
        if D_damping is None:
            # critical damping: D = 2 * sqrt(K * M_diag); approx M_diag = 1
            D_damping = 2.0 * K_stiffness.sqrt()
        self._K = K_stiffness.to(self._device)
        self._D = D_damping.to(self._device)

        # OU noise state per particle [NP, n_obj_dof]
        self._ou_state = torch.zeros(num_particles, n_obj_dof, device=self._device)

        self._rng = torch.Generator(device=self._device)
        if seed is not None:
            self._rng.manual_seed(seed)

        # object qpos block width: 7 for a free joint (3 pos + wxyz quat),
        # else == n_obj_dof (e.g. a 3-slide+3-hinge chain has no quat).
        self._obj_qpos_dim = self._model.nq - n_joints

    @property
    def state_dim(self) -> int:
        return self._state_dim

    def _pack_state(self, d) -> torch.Tensor:
        """Extract state tensor from a single mjData."""
        import numpy as np
        q_r = torch.tensor(d.qpos[:self._n_joints], dtype=torch.float32)
        qd_r = torch.tensor(d.qvel[:self._n_joints], dtype=torch.float32)
        q_obj = torch.tensor(d.qpos[self._n_joints:self._n_joints + self._n_obj_dof], dtype=torch.float32)
        qd_obj = torch.tensor(d.qvel[self._n_joints:self._n_joints + self._n_obj_dof], dtype=torch.float32)
        return torch.cat([q_r, qd_r, q_obj, qd_obj]).to(self._device)

    def _unpack_state(self, d, state: torch.Tensor) -> None:
        """Write a state tensor back into a mjData."""
        s = state.cpu().numpy()
        d.qpos[:self._n_joints] = s[:self._n_joints]
        d.qvel[:self._n_joints] = s[self._n_joints:2 * self._n_joints]
        d.qpos[self._n_joints:self._n_joints + self._n_obj_dof] = s[2 * self._n_joints:2 * self._n_joints + self._n_obj_dof]
        d.qvel[self._n_joints:self._n_joints + self._n_obj_dof] = s[2 * self._n_joints + self._n_obj_dof:]

    def _ou_step(self) -> torch.Tensor:
        """Step OU process and return object torque noise [NP, n_obj_dof]."""
        noise = torch.randn_like(self._ou_state, generator=self._rng)
        self._ou_state = (
            self._ou_state
            - self._ou_theta * self._ou_state * self._dt
            + self._ou_sigma * math.sqrt(self._dt) * noise
        )
        return self._ou_state

    def step(
        self,
        particles: torch.Tensor,
        control: torch.Tensor,
        n_substeps: int = 1,
    ) -> torch.Tensor:
        """Step all NP particles using MuJoCo forward dynamics.

        Applies the compliance controller (eq. 3) then mj_step for each
        substep.  OU force noise is injected on the object DoFs.

        :param particles: ``[NP, state_dim]``
        :param control: ``[n_joints]`` desired joint positions q_r^d.
        :param n_substeps: Number of 5ms substeps (100 for 2 Hz → 200 Hz).
        :return: ``[NP, state_dim]``
        """
        q_des = control[:self._n_joints].cpu()
        next_states = []

        for i, d in enumerate(self._datas[:particles.shape[0]]):
            self._unpack_state(d, particles[i])
            for _ in range(n_substeps):
                # impedance control torque (eq. 3)
                q_r = torch.tensor(d.qpos[:self._n_joints], dtype=torch.float32)
                qd_r = torch.tensor(d.qvel[:self._n_joints], dtype=torch.float32)
                tau = (self._K.cpu() * (q_des - q_r) - self._D.cpu() * qd_r).numpy()
                d.ctrl[:self._n_joints] = tau

                # OU object force (per substep; use slice for particle i)
                ou_torque = self._ou_state[i].cpu().numpy()
                d.xfrc_applied[self._n_joints:, :] = 0.0
                try:
                    d.qfrc_applied[self._n_joints:self._n_joints + self._n_obj_dof] = ou_torque
                except Exception:
                    pass

                self._mujoco.mj_step(self._model, d)

            # step OU for this particle
            ou_noise = torch.randn(self._n_obj_dof, generator=self._rng)
            self._ou_state[i] = (
                self._ou_state[i]
                - self._ou_theta * self._ou_state[i] * self._dt
                + self._ou_sigma * math.sqrt(self._dt) * ou_noise
            )
            next_states.append(self._pack_state(d))

        return torch.stack(next_states, dim=0)

    def step_ground_truth(self, control: torch.Tensor, n_substeps: int = 1) -> torch.Tensor:
        """Step the single ground-truth ``mjData`` forward (for training rollouts).

        Mirrors the impedance control law in :meth:`step` but applies it to
        ``self._gt_data`` directly instead of the NP-particle data pool, and
        does not inject OU force noise (the ground-truth trajectory is the
        sim's real dynamics, not a filter hypothesis).

        :param control: ``[n_joints]`` desired joint positions q_r^d.
        :param n_substeps: Number of 5ms substeps.
        :return: Joint encoder reading ``[n_joints]`` after stepping.
        """
        q_des = control[:self._n_joints].cpu()
        d = self._gt_data
        for _ in range(n_substeps):
            q_r = torch.tensor(d.qpos[:self._n_joints], dtype=torch.float32)
            qd_r = torch.tensor(d.qvel[:self._n_joints], dtype=torch.float32)
            tau = (self._K.cpu() * (q_des - q_r) - self._D.cpu() * qd_r).numpy()
            d.ctrl[:self._n_joints] = tau
            self._mujoco.mj_step(self._model, d)
        return torch.tensor(d.qpos[:self._n_joints], dtype=torch.float32, device=self._device)

    def get_measurement(
        self,
        particles: torch.Tensor,
        sigma_enc: float = 1.0,
    ) -> torch.Tensor:
        """Log-likelihood log N(z; q_r^[i], sigma^2 I) per particle."""
        sigma = math.radians(sigma_enc)
        z = self._pack_state(self._gt_data)[:self._n_joints]  # ground-truth q_r
        q_r = particles[:, :self._n_joints]
        diff = q_r - z.unsqueeze(0)
        log_ll = -0.5 * (diff / sigma).pow(2).sum(-1)
        log_ll -= self._n_joints * math.log(sigma * math.sqrt(2 * math.pi))
        return log_ll

    def get_object_pose(self) -> torch.Tensor:
        """Return ground-truth object pose [x,y,z,qx,qy,qz,qw].

        For a free-joint object (qpos block width 7: 3 pos + wxyz quat) this
        returns the true orientation. For a non-quaternion joint chain (e.g.
        3 slide + 3 hinge, qpos block width == n_obj_dof) only xyz position
        is meaningful; the quaternion is returned as identity.
        """
        pos = self._gt_data.qpos[self._n_joints:self._n_joints + 3]
        if self._obj_qpos_dim == 7:
            quat = self._gt_data.qpos[self._n_joints + 3:self._n_joints + 7]  # wxyz in MuJoCo
            # convert MuJoCo wxyz → xyzw
            return torch.tensor([*pos, quat[1], quat[2], quat[3], quat[0]], dtype=torch.float32, device=self._device)
        return torch.tensor([*pos, 0.0, 0.0, 0.0, 1.0], dtype=torch.float32, device=self._device)

    def get_robot_state(self) -> torch.Tensor:
        s = self._pack_state(self._gt_data)
        return s[:2 * self._n_joints]

    def reset(self, seed: Optional[int] = None) -> torch.Tensor:
        if seed is not None:
            self._rng.manual_seed(seed)
        self._mujoco.mj_resetData(self._model, self._gt_data)
        self._ou_state.zero_()
        return torch.tensor(self._gt_data.qpos[:self._n_joints], dtype=torch.float32, device=self._device)
