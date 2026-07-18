from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch


class MotionController(ABC):
    """Abstract low-level goal-directed motion controller.

    Each controller is an attractor for the robot end-effector towards a
    Cartesian target t_j expressed relative to one GMM mode mu_o,i.
    At 200 Hz the controller produces an acceleration-limited reference
    (q_r^d, q_dot_r^d, q_ddot_r^d) consumed by the compliance controller.

    NC = n * NG controllers are instantiated per task (paper §IV-B).
    The high-level BeliefDQN selects which controller is active at 2 Hz.

    :param target_pose: Target end-effector pose t_j in object-relative frame,
        shape ``[7]`` (xyz + quaternion xyzw).
    :param gmm_mode_index: Which GMM mode index i this controller is tied to.
    :param gripper_open: Whether this controller opens the gripper (PR2 variants).
    """

    def __init__(
        self,
        *,
        target_pose: torch.Tensor,
        gmm_mode_index: int = 0,
        gripper_open: bool = False,
    ) -> None:
        self.target_pose = target_pose
        self.gmm_mode_index = gmm_mode_index
        self.gripper_open = gripper_open

    @abstractmethod
    def act(
        self,
        robot_state: torch.Tensor,
        object_mode_pose: torch.Tensor,
    ) -> torch.Tensor:
        """Compute a reference command for the compliance controller.

        :param robot_state: Current robot joint positions + velocities,
            shape ``[2 * n_joints]``.
        :param object_mode_pose: Current GMM mode mean for the object,
            shape ``[7]`` (xyz + quaternion xyzw).
        :return: Reference command ``[n_joints]`` — desired joint positions.
            The compliance controller converts this to torques.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable label for TensorBoard / logging."""
