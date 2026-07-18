from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class DynamicsBackend(ABC):
    """Abstract physics backend interface for the particle filter.

    The particle filter holds a reference to one DynamicsBackend.
    Each call to predict() invokes step() on the backend once per
    high-level action, propagating all NP particles in a batch.

    The simulation environment (MuJoCo, mjlab, IsaacLab) provides a
    concrete subclass.  The agent is completely decoupled from the
    engine — it only holds a DynamicsBackend reference.

    All tensors are on the device specified at construction.
    """

    @abstractmethod
    def step(
        self,
        particles: torch.Tensor,
        control: torch.Tensor,
        n_substeps: int = 1,
    ) -> torch.Tensor:
        """Propagate a batch of particles forward by n_substeps.

        Each particle is an independent copy of the system state.
        The backend applies the compliance-interfaced contact dynamics
        (paper eq. 4) and injects Ornstein-Uhlenbeck force noise on
        the object (paper §III-B) independently per particle.

        :param particles: Current particle states, shape ``[NP, state_dim]``.
        :param control: Reference command for the compliance controller,
            shape ``[action_dim]``.  Interpreted as (q_r^d, q_dot_r^d, q_ddot_r^d).
        :param n_substeps: Number of physics substeps.  At 200 Hz inner
            loop and 2 Hz high-level policy, this is 100.
        :return: Next particle states, shape ``[NP, state_dim]``.
        """

    @abstractmethod
    def get_measurement(
        self,
        particles: torch.Tensor,
        sigma_enc: float = 1.0,
    ) -> torch.Tensor:
        """Compute log-likelihood log p(z | x^[i]) for each particle.

        The paper uses a Gaussian measurement model on joint positions
        only (eq. 2): p(z | x) = N(q_r; sigma_enc^2 * I).

        :param particles: Shape ``[NP, state_dim]``.
        :param sigma_enc: Encoder noise std in degrees (default 1.0 deg).
        :return: Log-likelihoods, shape ``[NP]``.
        """

    @abstractmethod
    def get_object_pose(self) -> torch.Tensor:
        """Return the current ground-truth object pose from the sim.

        Used only by training rollouts to sample the ground-truth
        candidate that drives the particle filter's measurement update
        (paper §IV-C).  Not called during inference.

        :return: Object pose, shape ``[7]`` (xyz + quaternion xyzw).
        """

    @abstractmethod
    def get_robot_state(self) -> torch.Tensor:
        """Return current robot joint positions and velocities.

        :return: Shape ``[n_r + n_r]`` = ``[2 * n_joints]``.
        """

    @abstractmethod
    def reset(self, seed: int | None = None) -> torch.Tensor:
        """Reset the simulation and return the initial observation.

        :param seed: Optional random seed.
        :return: Initial joint encoder reading, shape ``[n_joints]``.
        """
