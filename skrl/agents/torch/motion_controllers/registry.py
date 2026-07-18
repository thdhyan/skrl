from __future__ import annotations

from typing import Callable, List, Optional

import torch

from skrl.agents.torch.motion_controllers.etasl_attractor import ETaSLAttractorController


def build_controller_suite(
    *,
    gmm_means: torch.Tensor,
    target_templates: List[torch.Tensor],
    n_joints: int,
    fk_fn: Callable,
    jacobian_fn: Callable,
    Kp: float = 5.0,
    Kd: Optional[float] = None,
    q_ddot_min: Optional[torch.Tensor] = None,
    q_ddot_max: Optional[torch.Tensor] = None,
    q_min: Optional[torch.Tensor] = None,
    q_max: Optional[torch.Tensor] = None,
    qdot_max: Optional[torch.Tensor] = None,
    gripper_variants: bool = False,
    device: str | torch.device = "cpu",
) -> List[ETaSLAttractorController]:
    """Build the full NC-controller set (paper §IV-B).

    NC = len(target_templates) * NG controllers are created.
    For each GMM mode i and each target template t_j, one controller is
    instantiated that attracts the end-effector to ``mu_o,i ⊕ t_j``.

    If ``gripper_variants=True``, two controllers are built per
    (mode, template) pair — one with gripper open, one closed — giving
    NC = 2 * n * NG (used for the PR2-Hammer task).

    :param gmm_means: Object GMM mode means, shape ``[NG, state_dim]``.
        Only the first 7 entries (xyz + quat) are used as the object pose.
    :param target_templates: List of ``n`` target poses in object frame,
        each ``[7]`` (xyz + quaternion xyzw).
    :param n_joints: n_r.
    :param fk_fn: Forward kinematics callable (q_r) -> (pos[3], quat[4]).
    :param jacobian_fn: Jacobian callable (q_r) -> J[6, n_r].
    :param Kp: Proportional gain (time constant 0.2 s → Kp=5, paper §V-A).
    :param Kd: Derivative gain.  Auto-computed (critically damped) if None.
    :param q_ddot_min: Joint accel lower bound ``[n_r]``.
    :param q_ddot_max: Joint accel upper bound ``[n_r]``.
    :param q_min: Joint position lower bound ``[n_r]``.
    :param q_max: Joint position upper bound ``[n_r]``.
    :param qdot_max: Joint velocity limit ``[n_r]`` (symmetric).
    :param gripper_variants: If True, add open/close gripper variants (2× NC).
    :param device: Torch device.
    :return: List of NC (or 2·NC) ETaSLAttractorController instances.
    """
    NG = gmm_means.shape[0]
    controllers = []
    gripper_states = [False, True] if gripper_variants else [False]

    for mode_idx in range(NG):
        for t_j in target_templates:
            for gripper_open in gripper_states:
                ctrl = ETaSLAttractorController(
                    target_pose=t_j,
                    n_joints=n_joints,
                    fk_fn=fk_fn,
                    jacobian_fn=jacobian_fn,
                    Kp=Kp,
                    Kd=Kd,
                    q_ddot_min=q_ddot_min,
                    q_ddot_max=q_ddot_max,
                    q_min=q_min,
                    q_max=q_max,
                    qdot_max=qdot_max,
                    gmm_mode_index=mode_idx,
                    gripper_open=gripper_open,
                    device=device,
                )
                controllers.append(ctrl)

    return controllers
