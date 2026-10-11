from __future__ import annotations

import math


class SoftmaxMaskCurriculum:
    """Probability of masking the privileged observations, as a function of training progress.

    Each episode picks one of two modes, *visible* (actor uses the privileged encoder) or *masked*
    (actor uses the history estimator). Their probability is a 2-way softmax over the logits
    ``[0, sharpness * (progress - center)]``, i.e. a sigmoid in training progress. The sigmoid is
    rescaled so that the mask probability is exactly ``p_start`` at ``progress = start`` and ``p_end``
    at ``progress = end``: training starts fully observable and ends fully masked.

    :param start: Progress (fraction of total timesteps) where masking starts to ramp up.
    :param end: Progress where the ramp is complete.
    :param sharpness: Softmax inverse temperature. Small → nearly linear ramp, large → step at ``center``.
    :param p_start: Mask probability before ``start``.
    :param p_end: Mask probability after ``end``.
    """

    def __init__(
        self,
        *,
        start: float = 0.1,
        end: float = 0.8,
        sharpness: float = 10.0,
        p_start: float = 0.0,
        p_end: float = 1.0,
    ) -> None:
        if not 0.0 <= start < end <= 1.0:
            raise ValueError(f"Expected 0 <= start < end <= 1, got start={start}, end={end}")
        self.start, self.end, self.sharpness = start, end, sharpness
        self.p_start, self.p_end = p_start, p_end
        self.progress = 0.0

    def _sigmoid(self, x: float) -> float:
        center = 0.5 * (self.start + self.end)
        return 1.0 / (1.0 + math.exp(-self.sharpness * (x - center)))

    def mask_probability(self, progress: float | None = None) -> float:
        """Mask probability at ``progress`` (defaults to the last value set with :meth:`set_progress`)."""
        x = min(max(self.progress if progress is None else progress, self.start), self.end)
        lo, hi = self._sigmoid(self.start), self._sigmoid(self.end)
        ramp = (self._sigmoid(x) - lo) / (hi - lo)
        return self.p_start + (self.p_end - self.p_start) * ramp

    def set_progress(self, progress: float) -> None:
        self.progress = progress
