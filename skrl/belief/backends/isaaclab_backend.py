from __future__ import annotations

import math
from typing import Callable, Optional

import torch

from skrl.belief.backends.base import DynamicsBackend


class IsaacLabBackend(DynamicsBackend):
    """IsaacLab / IsaacSim backend for the particle filter.

    This backend is an interface stub.  The user provides two callables
    that bridge the gap between skrl's belief system and the IsaacLab
    environment running in a separate process or the same process.

    IsaacLab runs GPU-parallel environments internally.  The particle
    filter needs NP independent rollouts of the contact dynamics, which
    maps naturally to IsaacLab's vectorised env API (num_envs = NP).

    Integration pattern
    -------------------
    1.  Create an IsaacLab env with num_envs = NP.
    2.  Pass ``step_fn`` and ``obs_fn`` to this backend at construction.
    3.  Pass this backend's ``step`` / ``get_measurement`` methods as
        ``motion_model_fn`` / ``measurement_model_fn`` to ParticleFilter.
    4.  At visualisation time call ``filter.get_particles()`` and overlay
        the particle cloud and GMM ellipsoids in the Isaac viewport.

    Required callables
    ------------------
    step_fn(particles, control, n_substeps) -> particles_next
        Drives the IsaacLab envs from states ``particles`` for
        ``n_substeps`` physics steps and returns the resulting states.
        Shape contract: same as DynamicsBackend.step.

    obs_fn(particles) -> log_likelihoods
        Returns log p(z | x^[i]) for each particle given the current
        ground-truth measurement obtained from the *real* env (the
        canonical env index 0 or a separate held-out env).
        Shape: [NP].

    pose_fn() -> tensor[7]
        Returns current ground-truth object pose [x,y,z,qx,qy,qz,qw]
        from the canonical env.

    robot_state_fn() -> tensor[2 * n_joints]
        Returns joint positions + velocities from the canonical env.

    reset_fn(seed) -> tensor[n_joints]
        Resets all envs and returns the initial joint encoder reading.

    :param step_fn: Callable implementing forward dynamics for NP envs.
    :param obs_fn: Callable returning log-likelihoods per particle.
    :param pose_fn: Callable returning ground-truth object pose.
    :param robot_state_fn: Callable returning robot joint state.
    :param reset_fn: Callable resetting envs and returning first obs.
    :param state_dim: Full state dimensionality.
    :param device: Torch device (typically 'cuda:0' for IsaacLab).
    """

    def __init__(
        self,
        *,
        step_fn: Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor],
        obs_fn: Callable[[torch.Tensor], torch.Tensor],
        pose_fn: Callable[[], torch.Tensor],
        robot_state_fn: Callable[[], torch.Tensor],
        reset_fn: Callable[[Optional[int]], torch.Tensor],
        state_dim: int,
        device: str | torch.device = "cuda:0",
    ) -> None:
        self._step_fn = step_fn
        self._obs_fn = obs_fn
        self._pose_fn = pose_fn
        self._robot_state_fn = robot_state_fn
        self._reset_fn = reset_fn
        self._state_dim = state_dim
        self._device = torch.device(device)

    @property
    def state_dim(self) -> int:
        return self._state_dim

    def step(
        self,
        particles: torch.Tensor,
        control: torch.Tensor,
        n_substeps: int = 1,
    ) -> torch.Tensor:
        return self._step_fn(particles.to(self._device), control.to(self._device), n_substeps)

    def get_measurement(
        self,
        particles: torch.Tensor,
        sigma_enc: float = 1.0,
    ) -> torch.Tensor:
        return self._obs_fn(particles.to(self._device))

    def get_object_pose(self) -> torch.Tensor:
        return self._pose_fn().to(self._device)

    def get_robot_state(self) -> torch.Tensor:
        return self._robot_state_fn().to(self._device)

    def reset(self, seed: Optional[int] = None) -> torch.Tensor:
        return self._reset_fn(seed).to(self._device)
