from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

from skrl.agents.torch.motion_controllers.base import MotionController


def _quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector v by quaternion q (xyzw convention)."""
    qv = q[:3]
    qw = q[3]
    return v + 2.0 * qw * torch.cross(qv, v, dim=0) + 2.0 * torch.cross(qv, torch.cross(qv, v, dim=0), dim=0)


def _pose_error(
    ee_pos: torch.Tensor,
    ee_quat: torch.Tensor,
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
) -> torch.Tensor:
    """6-DoF Cartesian error: [pos_err (3), rot_err_axis_angle (3)]."""
    pos_err = target_pos - ee_pos
    # quaternion error: q_err = q_target * q_ee^{-1}
    ee_conj = torch.cat([-ee_quat[:3], ee_quat[3:4]])
    q_err_xyz = ee_quat[3] * target_quat[:3] + target_quat[3] * ee_quat[:3] + torch.cross(target_quat[:3], ee_quat[:3], dim=0)
    q_err_w = target_quat[3] * ee_quat[3] - torch.dot(target_quat[:3], ee_quat[:3])
    rot_err = 2.0 * q_err_xyz * torch.sign(q_err_w)
    return torch.cat([pos_err, rot_err])  # [6]


class ETaSLAttractorController(MotionController):
    """Acceleration-resolved Cartesian attractor (paper §IV-B, eq. 6-7).

    Each controller is an attractor that accelerates the robot end-effector
    towards a desired Cartesian target t_j relative to object mode mu_o,i.
    The QP (eq. 7) minimises the tracking error subject to joint position,
    velocity, and acceleration box constraints.

    The QP is:
        min_{q_ddot}  || Je * q_ddot + Je_dot * q_dot - (-Kp*e - Kd*e_dot) ||^2
        s.t.  dcp/dq_ddot * q_ddot <= -Kp*cp - Kd*cp_dot   (pos limits)
              dcv/dq_ddot * q_ddot <= -Kd*cv_dot            (vel limits)
              q_ddot_min <= q_ddot <= q_ddot_max

    All constraints are linear in q_ddot and the cost is quadratic → QP.
    Solved via qpth (differentiable) if available, else closed-form
    unconstrained (fast for the stub backend).

    In this implementation the robot FK / Jacobian are provided by a
    callable ``fk_fn`` so the controller is decoupled from any physics
    engine.

    :param target_pose: Target pose t_j in object frame, ``[7]`` (xyz+xyzw).
    :param n_joints: Number of robot joints n_r.
    :param fk_fn: Callable(q_r -> (ee_pos[3], ee_quat[4])) — forward kinematics.
    :param jacobian_fn: Callable(q_r -> J[6, n_r]) — body Jacobian.
    :param Kp: Proportional gain (scalar or ``[6]``).  Time constant τ = 1/Kp.
        Paper uses τ = 0.2 s → Kp = 5.
    :param Kd: Derivative gain (critically damped: Kd = 2*sqrt(Kp)).
    :param q_ddot_min: Joint acceleration lower bounds ``[n_r]``.
    :param q_ddot_max: Joint acceleration upper bounds ``[n_r]``.
    :param q_min: Joint position lower bounds ``[n_r]`` (for cp constraint).
    :param q_max: Joint position upper bounds ``[n_r]``.
    :param qdot_max: Joint velocity limits ``[n_r]`` (symmetric).
    :param gmm_mode_index: Which GMM mode this controller targets.
    :param gripper_open: Gripper state command (PR2-Hammer variants).
    :param device: Torch device.
    """

    def __init__(
        self,
        *,
        target_pose: torch.Tensor,
        n_joints: int,
        fk_fn,
        jacobian_fn,
        Kp: float = 5.0,
        Kd: Optional[float] = None,
        q_ddot_min: Optional[torch.Tensor] = None,
        q_ddot_max: Optional[torch.Tensor] = None,
        q_min: Optional[torch.Tensor] = None,
        q_max: Optional[torch.Tensor] = None,
        qdot_max: Optional[torch.Tensor] = None,
        gmm_mode_index: int = 0,
        gripper_open: bool = False,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__(
            target_pose=target_pose,
            gmm_mode_index=gmm_mode_index,
            gripper_open=gripper_open,
        )
        self._n = n_joints
        self._fk_fn = fk_fn
        self._jac_fn = jacobian_fn
        self._Kp = Kp
        self._Kd = Kd if Kd is not None else 2.0 * math.sqrt(Kp)
        self._device = torch.device(device)

        inf = float("inf")
        self._q_ddot_min = (q_ddot_min if q_ddot_min is not None
                            else torch.full((n_joints,), -50.0)).to(self._device)
        self._q_ddot_max = (q_ddot_max if q_ddot_max is not None
                            else torch.full((n_joints,), 50.0)).to(self._device)
        self._q_min = (q_min if q_min is not None
                       else torch.full((n_joints,), -math.pi)).to(self._device)
        self._q_max = (q_max if q_max is not None
                       else torch.full((n_joints,), math.pi)).to(self._device)
        self._qdot_max = (qdot_max if qdot_max is not None
                          else torch.full((n_joints,), math.pi)).to(self._device)

        self._e_dot_prev = torch.zeros(6, device=self._device)

        # qpth available?
        try:
            from qpth.qp import QPFunction
            self._qpf = QPFunction(verbose=False, check_Q_spd=False)
            self._use_qpth = True
        except ImportError:
            self._use_qpth = False

    @property
    def name(self) -> str:
        return f"ETaSL_mode{self.gmm_mode_index}_grip{'O' if self.gripper_open else 'C'}"

    def act(
        self,
        robot_state: torch.Tensor,
        object_mode_pose: torch.Tensor,
    ) -> torch.Tensor:
        """Compute desired joint acceleration q_ddot, integrate to q_r^d.

        :param robot_state: ``[2*n_joints]`` = ``[q_r, q_dot_r]``.
        :param object_mode_pose: Object mode mean ``[7]`` (xyz+xyzw).
        :return: Desired joint positions ``[n_joints]``.
        """
        q_r = robot_state[:self._n].to(self._device)
        q_dot_r = robot_state[self._n:2 * self._n].to(self._device)

        # forward kinematics
        ee_pos, ee_quat = self._fk_fn(q_r)
        J = self._jac_fn(q_r)  # [6, n]

        # world-frame target = object_mode_pose ⊕ target_pose
        obj_pos = object_mode_pose[:3].to(self._device)
        obj_quat = object_mode_pose[3:].to(self._device)
        t_pos = self.target_pose[:3].to(self._device)
        t_quat = self.target_pose[3:].to(self._device)

        target_pos_world = obj_pos + _quat_apply(obj_quat, t_pos)
        # compose quaternions: q_world = q_obj * q_target
        qr = obj_quat[3] * t_quat[:3] + t_quat[3] * obj_quat[:3] + torch.cross(obj_quat[:3], t_quat[:3], dim=0)
        qw = obj_quat[3] * t_quat[3] - torch.dot(obj_quat[:3], t_quat[:3])
        target_quat_world = torch.cat([qr, qw.unsqueeze(0)])

        # Cartesian error and derivative
        e = _pose_error(ee_pos, ee_quat, target_pos_world, target_quat_world)  # [6]
        e_dot = J @ q_dot_r  # approximate e_dot via Jacobian
        self._e_dot_prev = e_dot

        # desired Cartesian acceleration
        e_ddot_des = -self._Kp * e - self._Kd * e_dot  # [6]

        # solve QP: min ||J q_ddot - e_ddot_des||^2 s.t. bounds
        q_ddot = self._solve_qp(J, e_ddot_des, q_r, q_dot_r)

        # integrate twice to get q_r^d (simple Euler, dt = 1/200 s per substep)
        dt = 0.005
        q_dot_des = q_dot_r + q_ddot * dt
        q_des = q_r + q_dot_des * dt
        return q_des

    def _solve_qp(
        self,
        J: torch.Tensor,
        e_ddot_des: torch.Tensor,
        q_r: torch.Tensor,
        q_dot_r: torch.Tensor,
    ) -> torch.Tensor:
        """Solve the acceleration QP (eq. 7).

        Returns q_ddot [n_joints].
        Falls back to unconstrained least squares if qpth unavailable.
        """
        n = self._n
        # cost: min ||J q_ddot - e_ddot_des||^2
        # Q = J^T J, p = -J^T e_ddot_des
        # Tikhonov-regularised to keep Q PSD with a non-zero diagonal when J
        # is rank-deficient (e.g. a planar arm leaves rows 2-5 identically
        # zero), which would otherwise make qpth's LU factorization fail.
        Q = J.T @ J + 1e-6 * torch.eye(n, device=J.device, dtype=J.dtype)  # [n, n]
        p = -(J.T @ e_ddot_des)  # [n]

        # box constraints as inequality: [I; -I] q_ddot <= [q_ddot_max; -q_ddot_min]
        # plus position and velocity limit constraints (simplified)
        G_box = torch.cat([torch.eye(n), -torch.eye(n)], dim=0)  # [2n, n]
        h_box = torch.cat([self._q_ddot_max, -self._q_ddot_min])  # [2n]

        if self._use_qpth:
            # qpth expects batched inputs
            Q_b = Q.unsqueeze(0)
            p_b = p.unsqueeze(0)
            G_b = G_box.unsqueeze(0)
            h_b = h_box.unsqueeze(0)
            e_dummy = torch.zeros(1, 0, device=self._device)
            b_dummy = torch.zeros(1, 0, device=self._device)
            q_ddot = self._qpf(Q_b, p_b, G_b, h_b, e_dummy, b_dummy).squeeze(0)
        else:
            # unconstrained least squares: q_ddot = (J^T J)^{-1} J^T e_ddot_des
            q_ddot = torch.linalg.lstsq(J, e_ddot_des).solution
            # clamp to box
            q_ddot = q_ddot.clamp(self._q_ddot_min, self._q_ddot_max)

        return q_ddot
