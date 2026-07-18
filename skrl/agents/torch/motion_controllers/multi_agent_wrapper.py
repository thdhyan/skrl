from __future__ import annotations

from typing import Any, List

import torch

from skrl.agents.torch.motion_controllers.base import MotionController
from skrl.belief.torch.base import Belief


class MotionControllerSuite:
    """Manages NC low-level controllers and dispatches the active one.

    The BeliefDQN high-level policy selects a controller index a* at 2 Hz.
    This suite runs that controller for n_substeps at 200 Hz, collecting
    the sequence of joint references delivered to the compliance controller.

    Usage in a training loop::

        suite = MotionControllerSuite(controllers, belief, n_substeps=100)
        ref = suite.act(action_idx, robot_state, belief)

    :param controllers: List of NC MotionController instances, ordered
        consistently with the DQN output indices.
    :param n_substeps: Inner-loop substeps (200 Hz / 2 Hz = 100).
    """

    def __init__(
        self,
        controllers: List[MotionController],
        n_substeps: int = 100,
    ) -> None:
        self._controllers = controllers
        self._n_substeps = n_substeps
        self._nc = len(controllers)

    @property
    def num_controllers(self) -> int:
        return self._nc

    def act(
        self,
        action_idx: int,
        robot_state: torch.Tensor,
        belief: Belief,
    ) -> torch.Tensor:
        """Run the selected controller and return the final reference.

        For visualisation purposes the calling code should call
        ``belief.get_particles()`` and ``belief.gmm`` before/after.

        :param action_idx: Discrete action a* in {0, ..., NC-1}.
        :param robot_state: Current robot joint positions + velocities,
            shape ``[2 * n_joints]``.
        :param belief: Current Belief instance (used for the object mode pose).
        :return: Desired joint positions ``[n_joints]`` after n_substeps.
        """
        if action_idx < 0 or action_idx >= self._nc:
            raise IndexError(f"Controller index {action_idx} out of range [0, {self._nc})")

        ctrl = self._controllers[action_idx]
        mode_idx = ctrl.gmm_mode_index

        # extract object pose from belief GMM mode
        gmm = belief.gmm
        if gmm is None:
            belief.compress()
            gmm = belief.gmm

        # object mode mean — first 7 entries are pose (xyz + quaternion)
        # state_dim may include velocities; we take only config part
        obj_mean = gmm.means[mode_idx][:7]  # [7]: xyz + quat xyzw

        n_joints = robot_state.shape[0] // 2
        q_r = robot_state[:n_joints]
        q_dot_r = robot_state[n_joints:]
        for _ in range(self._n_substeps):
            ref = ctrl.act(torch.cat([q_r, q_dot_r]), obj_mean)
            q_r = ref

        return ref

    def controller_names(self) -> List[str]:
        return [c.name for c in self._controllers]
