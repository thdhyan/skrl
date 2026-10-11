from __future__ import annotations

import time
from typing import Any, Callable, Optional

import torch

from skrl.belief.torch.base import Belief
from skrl.belief.torch.markers import belief_markers, belief_stats


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

    For visualization, ``get_belief_markers()`` returns the particles, weights and
    GMM clusters (see :func:`skrl.belief.torch.markers.belief_markers`). If the agent
    config sets ``belief_true_state_fn(infos) -> [state_dim]``, every step also tracks
    the belief statistics against the true state (``Belief / ...``).

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

    def _belief_advance(
        self, action: torch.Tensor, next_observation: torch.Tensor, infos: Any = None
    ) -> torch.Tensor:
        """Predict with ``action``, update with ``next_observation``, and return b̃_{t+1}.

        :param action: Control input passed to the motion model, shape ``[action_dim]`` or ``[1]``.
        :param next_observation: Raw measurement, shape ``[obs_dim]`` (batch dim squeezed off).
        :param infos: Environment infos, passed to ``cfg.belief_true_state_fn`` for the statistics.
        :return: Compressed belief b̃_{t+1}, shape ``[1, belief_dim]``.
        """
        self._belief.predict(action.squeeze(-1))
        self._belief.update(next_observation.squeeze(0))

        t0 = time.perf_counter()
        self._belief.compress()
        compress_ms = (time.perf_counter() - t0) * 1000.0

        b_t1 = self._belief.get_compressed().unsqueeze(0)

        self.track_data("Belief / compress_ms_record", compress_ms)
        self._track_belief_stats(infos)

        self._b_tilde = b_t1.clone()
        return b_t1

    def _belief_true_state(self, infos: Any) -> torch.Tensor | None:
        fn = getattr(self.cfg, "belief_true_state_fn", None)
        return None if fn is None or infos is None else fn(infos)

    def _track_belief_stats(self, infos: Any) -> None:
        """Track spread, ESS, clusters (count, hefts, stds) and, with a true state, the cluster errors."""
        stats = belief_stats(self._belief, self._belief_true_state(infos))
        for key, value in stats.items():
            if isinstance(value, list):  # per cluster, sorted by decreasing heft
                for k, v in enumerate(value):
                    self.track_data(f"Belief / {key[:-1]} {k}", v)
            else:
                self.track_data(f"Belief / {key}", value)

    def get_belief_markers(
        self,
        true_state: torch.Tensor | None = None,
        state_to_position: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> dict[str, Any]:
        """Current particles, weights and GMM clusters for drawing in a simulator.

        See :func:`skrl.belief.torch.markers.belief_markers`. ``state_to_position`` defaults to
        ``cfg.belief_state_to_position``. Publish the result to another process with
        :class:`skrl.belief.torch.markers.BeliefMarkerServer`.
        """
        if state_to_position is None:
            state_to_position = getattr(self.cfg, "belief_state_to_position", None)
        return belief_markers(self._belief, true_state, state_to_position=state_to_position)

    def _with_belief_markers(self, outputs: dict[str, Any]) -> dict[str, Any]:
        """Add the belief markers to the ``act()`` outputs when ``cfg.return_belief_markers`` is set."""
        if getattr(self.cfg, "return_belief_markers", False):
            outputs["belief_markers"] = self.get_belief_markers()
        return outputs

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
