from __future__ import annotations

import math
from typing import Optional

import torch

from skrl.belief.backends.base import DynamicsBackend


class StubBackend(DynamicsBackend):
    """Pure-torch stub backend for unit tests and CI.

    Models a 2-DoF planar robot arm and a free-sliding box on a table.
    No MuJoCo or mjlab dependency required.

    State layout: [q_r0, q_r1, q_dot_r0, q_dot_r1, x_obj, y_obj, theta_obj,
                   xdot_obj, ydot_obj, thetadot_obj]
    state_dim = 10 (2 robot joints + 3 object planar pose + 5 velocities)

    The dynamics are simple linear + spring-damper — realistic enough to
    test the filter pipeline, not intended for policy training.

    :param n_joints: Number of robot joints.  Default 2.
    :param dt: Substep duration in seconds.  Default 0.005 (200 Hz).
    :param process_noise_std: Std of additive Gaussian process noise.
    :param sigma_enc_deg: Encoder noise std in degrees.
    :param device: Torch device.
    :param seed: Optional RNG seed.
    """

    N_ROBOT = 2   # joint count
    N_OBJ_POS = 3  # planar: x, y, theta
    N_OBJ_VEL = 3

    def __init__(
        self,
        *,
        n_joints: int = 2,
        dt: float = 0.005,
        process_noise_std: float = 0.01,
        sigma_enc_deg: float = 1.0,
        device: str | torch.device = "cpu",
        seed: Optional[int] = None,
    ) -> None:
        self._n_joints = n_joints
        self._dt = dt
        self._noise_std = process_noise_std
        self._sigma_enc = math.radians(sigma_enc_deg)
        self._device = torch.device(device)

        # single ground-truth state used by get_object_pose / get_robot_state
        self._state_dim = 2 * n_joints + self.N_OBJ_POS + self.N_OBJ_VEL
        self._gt_state: torch.Tensor = torch.zeros(self._state_dim, device=self._device)

        self._rng = torch.Generator(device=self._device)
        if seed is not None:
            self._rng.manual_seed(seed)

    @property
    def state_dim(self) -> int:
        return self._state_dim

    # ------------------------------------------------------------------
    # DynamicsBackend interface
    # ------------------------------------------------------------------

    def step(
        self,
        particles: torch.Tensor,
        control: torch.Tensor,
        n_substeps: int = 1,
    ) -> torch.Tensor:
        """Simple Euler integration with spring-damper robot + sliding object.

        :param particles: ``[NP, state_dim]``
        :param control: ``[n_joints]`` — desired joint positions (q_r^d)
        :param n_substeps: Number of integration substeps.
        :return: ``[NP, state_dim]``
        """
        p = particles.clone()
        kp, kd = 50.0, 10.0  # spring, damper gains
        friction = 0.1

        for _ in range(n_substeps):
            q_r = p[:, :self._n_joints]
            qd_r = p[:, self._n_joints:2 * self._n_joints]
            obj_pos = p[:, 2 * self._n_joints:2 * self._n_joints + self.N_OBJ_POS]
            obj_vel = p[:, 2 * self._n_joints + self.N_OBJ_POS:]

            # robot spring-damper towards control target
            q_des = control[:self._n_joints].to(self._device)
            acc_r = kp * (q_des - q_r) - kd * qd_r
            qd_r_new = qd_r + acc_r * self._dt
            q_r_new = q_r + qd_r_new * self._dt

            # object slides with friction
            acc_obj = -friction * obj_vel
            obj_vel_new = obj_vel + acc_obj * self._dt
            obj_pos_new = obj_pos + obj_vel_new * self._dt

            # add process noise
            noise = torch.randn_like(p, generator=self._rng) * self._noise_std
            p = torch.cat([q_r_new, qd_r_new, obj_pos_new, obj_vel_new], dim=-1) + noise

        return p

    def get_measurement(
        self,
        particles: torch.Tensor,
        sigma_enc: float = 1.0,
    ) -> torch.Tensor:
        """Log-likelihood of joint encoder reading given each particle.

        Measurement model: z ~ N(q_r, sigma^2 I) (paper eq. 2).
        The ground-truth reading is q_r from self._gt_state.

        :param particles: ``[NP, state_dim]``
        :param sigma_enc: std in degrees, converted to radians internally.
        :return: log-likelihoods ``[NP]``
        """
        sigma = math.radians(sigma_enc)
        z = self._gt_state[:self._n_joints]  # ground-truth joint positions
        q_r = particles[:, :self._n_joints]  # [NP, n_joints]
        diff = q_r - z.unsqueeze(0)  # [NP, n_joints]
        log_ll = -0.5 * (diff / sigma).pow(2).sum(-1)  # [NP]
        log_ll -= self._n_joints * math.log(sigma * math.sqrt(2 * math.pi))
        return log_ll

    def get_object_pose(self) -> torch.Tensor:
        """Return ground-truth object pose [x, y, theta, 0, 0, 0, 1] (stub)."""
        pos = self._gt_state[2 * self._n_joints:2 * self._n_joints + self.N_OBJ_POS]
        # convert planar [x, y, theta] to 7-vec [x, y, 0, qx, qy, qz, qw]
        theta = pos[2]
        qw = torch.cos(theta / 2)
        qz = torch.sin(theta / 2)
        return torch.tensor([pos[0], pos[1], 0.0, 0.0, 0.0, qz, qw], device=self._device)

    def get_robot_state(self) -> torch.Tensor:
        """Return robot joint positions and velocities ``[2 * n_joints]``."""
        return self._gt_state[:2 * self._n_joints].clone()

    def reset(self, seed: Optional[int] = None) -> torch.Tensor:
        """Reset ground-truth state and return initial joint encoder reading."""
        if seed is not None:
            self._rng.manual_seed(seed)
        self._gt_state = torch.randn(self._state_dim, device=self._device, generator=self._rng) * 0.1
        return self._gt_state[:self._n_joints].clone()

    def step_ground_truth(self, control: torch.Tensor, n_substeps: int = 1) -> torch.Tensor:
        """Step the single ground-truth state (for training rollouts).

        :return: Joint encoder reading ``[n_joints]`` after stepping.
        """
        self._gt_state = self.step(
            self._gt_state.unsqueeze(0), control, n_substeps
        ).squeeze(0)
        return self._gt_state[:self._n_joints].clone()
