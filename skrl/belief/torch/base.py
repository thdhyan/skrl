from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch


class Belief(ABC):
    """Abstract base class for belief tracking over partially observable states.

    Belief tracks a distribution over the hidden system state given a history
    of actions and observations.  Concrete implementations (e.g. ParticleFilter)
    maintain the internal representation and expose a compressed fixed-size
    tensor suitable as input to a neural network policy.

    The agent owns the Belief instance.  The simulation backend is accessed
    only through the motion_model and measurement_model callables so that the
    filter is decoupled from any specific physics engine.
    """

    @abstractmethod
    def reset(self, initial_particles: torch.Tensor, initial_weights: torch.Tensor | None = None) -> None:
        """Reset belief to a new prior.

        :param initial_particles: Initial particle set, shape ``[NP, state_dim]``.
        :param initial_weights: Optional initial weights, shape ``[NP]``.
            If None, uniform weights are used.
        """

    @abstractmethod
    def predict(self, control: torch.Tensor) -> None:
        """Propagate particles forward through the motion model.

        Calls ``backend.step(control, n_substeps)`` for each particle and
        updates the particle set in place.  Does not resample.

        :param control: Control input applied over the high-level timestep,
            shape ``[action_dim]``.
        """

    @abstractmethod
    def update(self, measurement: torch.Tensor) -> None:
        """Weight particles using the measurement likelihood and resample.

        Applies the measurement model ``p(z | x)`` to reweight particles,
        then performs systematic resampling when N_eff / NP < 0.5.

        :param measurement: Sensor reading (e.g. joint encoder positions),
            shape ``[obs_dim]``.
        """

    @abstractmethod
    def compress(self) -> None:
        """Fit a GMM to the current particle set and cache the result.

        Updates the internal ``GMMState`` used by ``get_compressed()``.
        Should be called once per high-level step after ``update()``.
        """

    @abstractmethod
    def get_particles(self) -> torch.Tensor:
        """Return current particle states, shape ``[NP, state_dim]``."""

    @abstractmethod
    def get_weights(self) -> torch.Tensor:
        """Return current particle log-weights (unnormalised), shape ``[NP]``."""

    @abstractmethod
    def get_compressed(self) -> torch.Tensor:
        """Return the flattened GMM belief vector b̃.

        Layout: ``[h_1..h_NG, mu_1..mu_NG, log_diag_cov_1..log_diag_cov_NG]``
        where each ``mu_i`` and ``log_diag_cov_i`` has ``state_dim`` entries.

        Shape: ``[NG + 2 * NG * state_dim]``
        """

    @abstractmethod
    def get_entropy(self) -> torch.Tensor:
        """Return GMM upper-bound differential entropy (scalar tensor)."""

    @property
    @abstractmethod
    def belief_dim(self) -> int:
        """Dimensionality of the compressed belief vector."""

    @property
    @abstractmethod
    def state_dim(self) -> int:
        """Dimensionality of a single particle state vector."""

    @property
    @abstractmethod
    def num_particles(self) -> int:
        """Number of particles NP."""

    @property
    @abstractmethod
    def num_components(self) -> int:
        """Number of GMM components NG."""
