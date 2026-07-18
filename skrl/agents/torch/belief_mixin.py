from __future__ import annotations

import time
from typing import Any, Callable, Optional

import torch

from skrl.belief.torch.base import Belief


class BeliefAgentMixin:
    """Shared belief-tracking lifecycle for belief-conditioned agents.

    Factors out the particle-filter plumbing common to BeliefDQN and
    BeliefPPO: instantiating the Belief from a factory, compressing it
    into b̃ at action-selection time, driving predict/update/compress
    after each env step, and resetting the filter at episode start.

    A consuming Agent calls, in order:
    - ``self._belief_setup(belief_factory)`` once, inside ``init()``.
    - ``self._belief_observe()`` at the start of ``act()`` to get the
      current compressed belief b̃ (and track compression time).
    - ``self._belief_advance(action, next_observation)`` inside
      ``record_transition()`` to push the filter forward with the
      taken action and the next raw observation, returning b̃_{t+1}
      and tracking entropy.
    - ``self.reset_belief(initial_particles, initial_weights)`` at the
      start of each episode.

    Tracking data is reported via ``self.track_data(...)``, so the
    consuming class must be a ``skrl.agents.torch.Agent`` subclass.
    """

    _belief: Belief
    _b_tilde: Optional[torch.Tensor]
    track_data: Callable[[str, float], None]

    def _belief_setup(self, belief_factory: Callable[[], Belief]) -> None:
        self._belief = belief_factory()
        self._b_tilde = None

    def _belief_observe(self) -> torch.Tensor:
        """Compress the current particle set and return b̃, shape ``[1, belief_dim]``."""
        t0 = time.perf_counter()
        self._belief.compress()
        compress_ms = (time.perf_counter() - t0) * 1000.0
        self.track_data("Belief / compress_ms", compress_ms)

        b_tilde = self._belief.get_compressed().unsqueeze(0)
        self._b_tilde = b_tilde.clone()
        return b_tilde

    def _belief_advance(self, action: torch.Tensor, next_observation: torch.Tensor) -> torch.Tensor:
        """Predict with ``action``, update with ``next_observation``, and return b̃_{t+1}.

        :param action: Control input passed to the motion model, shape ``[action_dim]`` or ``[1]``.
        :param next_observation: Raw measurement, shape ``[obs_dim]`` (batch dim squeezed off).
        :return: Compressed belief b̃_{t+1}, shape ``[1, belief_dim]``.
        """
        self._belief.predict(action.squeeze(-1))
        self._belief.update(next_observation.squeeze(0))

        t0 = time.perf_counter()
        self._belief.compress()
        compress_ms = (time.perf_counter() - t0) * 1000.0

        b_t1 = self._belief.get_compressed().unsqueeze(0)

        self.track_data("Belief / entropy", self._belief.get_entropy().item())
        self.track_data("Belief / compress_ms_record", compress_ms)

        self._b_tilde = b_t1.clone()
        return b_t1

    @property
    def belief_dim(self) -> int:
        return self._belief.belief_dim

    def reset_belief(self, initial_particles: torch.Tensor, initial_weights: torch.Tensor | None = None) -> None:
        """Reset the particle filter to a new prior at the start of an episode.

        :param initial_particles: ``[NP, state_dim]``
        :param initial_weights: Optional ``[NP]``
        """
        self._belief.reset(initial_particles, initial_weights)
        self._b_tilde = None
