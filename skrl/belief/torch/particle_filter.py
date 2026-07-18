from __future__ import annotations

from typing import Callable, Optional

import torch

from skrl.belief.torch.base import Belief
from skrl.belief.torch.compression import GMMState, flatten_gmm, gmm_fit
from skrl.belief.torch.entropy import gmm_entropy_upper_bound


class ParticleFilter(Belief):
    """Sequential importance resampling particle filter.

    Architecture (Option A from the plan):
    - The filter owns the backend reference via motion_model_fn and
      measurement_model_fn callables — no direct engine dep in this file.
    - The agent owns this filter instance; the sim backend is injected at
      construction via the two callables.
    - Particles are torch.Tensor [NP, state_dim] on the configured device.
    - After each high-level step: predict → update → compress is called by
      the agent.  get_particles() is always available for sim visualisation.

    Motion model (paper eq. 4):
        motion_model_fn(particles, control) -> particles_next
        Input:  [NP, state_dim], [action_dim]
        Output: [NP, state_dim]
        The callable is responsible for all substeps and noise injection
        (paper uses Ornstein-Uhlenbeck process on object torques).

    Measurement model (paper eq. 2):
        measurement_model_fn(particles, measurement) -> log_likelihoods
        Input:  [NP, state_dim], [obs_dim]
        Output: [NP]  (log p(z | x^[i]))
        The paper uses N(q_r; sigma_enc) with sigma_nom = 1 deg.

    Resampling: systematic resampling when N_eff / NP < resample_threshold.

    :param num_particles: NP.  Paper uses 300.
    :param state_dim: Dimensionality of the full state x = (q, q_dot).
    :param num_gmm_components: NG.  Paper uses 2.
    :param motion_model_fn: Callable propagating all particles one step.
    :param measurement_model_fn: Callable returning log-likelihoods.
    :param device: Torch device.
    :param resample_threshold: N_eff / NP below which resampling triggers.
    :param gmm_max_iter: EM iterations for GMM fitting.
    :param gmm_tol: EM convergence threshold.
    """

    def __init__(
        self,
        *,
        num_particles: int = 300,
        state_dim: int,
        num_gmm_components: int = 2,
        motion_model_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        measurement_model_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        device: str | torch.device = "cpu",
        resample_threshold: float = 0.5,
        gmm_max_iter: int = 50,
        gmm_tol: float = 1e-4,
    ) -> None:
        self._NP = num_particles
        self._state_dim = state_dim
        self._NG = num_gmm_components
        self._motion_model_fn = motion_model_fn
        self._measurement_model_fn = measurement_model_fn
        self._device = torch.device(device)
        self._resample_threshold = resample_threshold
        self._gmm_max_iter = gmm_max_iter
        self._gmm_tol = gmm_tol

        # particle state: [NP, state_dim], log-weights: [NP]
        self._particles: torch.Tensor = torch.zeros(num_particles, state_dim, device=self._device)
        self._log_weights: torch.Tensor = torch.full((num_particles,), -torch.log(torch.tensor(float(num_particles))),
                                                      device=self._device)

        # cached compressed belief
        self._gmm: Optional[GMMState] = None
        self._compressed: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Belief interface
    # ------------------------------------------------------------------

    def reset(self, initial_particles: torch.Tensor, initial_weights: Optional[torch.Tensor] = None) -> None:
        """Reset belief to a new prior particle set.

        :param initial_particles: Shape ``[NP, state_dim]``.
        :param initial_weights: Optional normalised weights ``[NP]``.
            Uniform if None.
        """
        self._particles = initial_particles.to(self._device)
        if initial_weights is not None:
            log_w = initial_weights.to(self._device).log()
        else:
            log_w = torch.full((self._NP,), -torch.log(torch.tensor(float(self._NP))), device=self._device)
        self._log_weights = log_w - torch.logsumexp(log_w, dim=0)
        self._gmm = None
        self._compressed = None

    def predict(self, control: torch.Tensor) -> None:
        """Propagate particles through the motion model (no weight update).

        :param control: Control input, shape ``[action_dim]``.
        """
        control = control.to(self._device)
        self._particles = self._motion_model_fn(self._particles, control)
        # weights unchanged during prediction
        self._gmm = None
        self._compressed = None

    def update(self, measurement: torch.Tensor) -> None:
        """Bayesian weight update + systematic resampling.

        :param measurement: Sensor observation, shape ``[obs_dim]``.
        """
        measurement = measurement.to(self._device)
        log_likelihoods = self._measurement_model_fn(self._particles, measurement)  # [NP]
        self._log_weights = self._log_weights + log_likelihoods
        # normalise
        self._log_weights = self._log_weights - torch.logsumexp(self._log_weights, dim=0)

        # resample if effective particle count is too low
        weights = self._log_weights.exp()
        n_eff = 1.0 / (weights ** 2).sum()
        if n_eff / self._NP < self._resample_threshold:
            self._systematic_resample(weights)

        self._gmm = None
        self._compressed = None

    def compress(self) -> None:
        """Fit GMM to current particles and cache b̃."""
        weights = self._log_weights.exp()
        self._gmm = gmm_fit(
            self._particles,
            weights,
            ng=self._NG,
            max_iter=self._gmm_max_iter,
            tol=self._gmm_tol,
        )
        self._compressed = flatten_gmm(self._gmm)

    def get_particles(self) -> torch.Tensor:
        """Return particle states, shape ``[NP, state_dim]``."""
        return self._particles

    def get_weights(self) -> torch.Tensor:
        """Return log-weights ``[NP]`` (normalised, sum-to-one in prob space)."""
        return self._log_weights

    def get_compressed(self) -> torch.Tensor:
        """Return cached b̃.  Call compress() first."""
        if self._compressed is None:
            self.compress()
        return self._compressed

    def get_entropy(self) -> torch.Tensor:
        """Return GMM entropy upper bound."""
        if self._gmm is None:
            self.compress()
        return gmm_entropy_upper_bound(self._gmm)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def belief_dim(self) -> int:
        return self._NG + 2 * self._NG * self._state_dim

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def num_particles(self) -> int:
        return self._NP

    @property
    def num_components(self) -> int:
        return self._NG

    @property
    def gmm(self) -> Optional[GMMState]:
        """Current fitted GMM, or None if compress() hasn't been called yet."""
        return self._gmm

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _systematic_resample(self, weights: torch.Tensor) -> None:
        """Systematic (low-variance) resampling, O(NP).

        Resets log-weights to uniform after resampling.

        :param weights: Normalised importance weights, shape ``[NP]``.
        """
        NP = self._NP
        cumsum = weights.cumsum(0)
        # NP evenly-spaced positions with a single random offset
        u = (torch.rand(1, device=self._device) + torch.arange(NP, device=self._device, dtype=weights.dtype)) / NP
        # for each u[j] find the first cumsum >= u[j]
        indices = torch.searchsorted(cumsum, u)
        indices = indices.clamp(0, NP - 1)
        self._particles = self._particles[indices]
        self._log_weights = torch.full((NP,), -torch.log(torch.tensor(float(NP), device=self._device)), device=self._device)
