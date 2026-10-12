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


class RewardGatedMaskCurriculum:
    """Mask probability that grows only while the training return stays constant.

    Information is removed in steps of ``step``. A step is taken when the smoothed training return
    has been *constant at its reference level* for ``patience`` updates: within ``tolerance * scale``
    of the reference (``scale = max(|reference|, return_scale)``) over the whole window. After every
    step the reference is reset to the current return, so the next step waits until the policy holds
    that level again with less information. If the return falls more than ``2 * tolerance * scale``
    below the reference, the probability backs off by ``step`` and the reference is reset. So each
    unit of information loss is paid for by a constant (not falling) return.

    The gate is armed only once the return has risen by ``2 * tolerance * scale`` above the first
    observed return, so the flat return of the initial random policy does not start the masking.

    ``deadline`` adds a linear floor ``p >= p_end * progress / deadline`` so masking always finishes
    by that fraction of training even if the gate stalls (``None`` disables the floor).

    The agent calls :meth:`observe_return` once per update with the mean training return.

    :param step: Increase (and back-off) of the mask probability per gate decision.
    :param tolerance: Relative tolerance of the plateau, arming and drop tests.
    :param patience: Updates in the plateau window (and minimum updates between two changes).
    :param smoothing: EMA coefficient of the observed returns (1: no smoothing).
    :param return_scale: Lower bound of the scale used by the relative tests (returns near zero).
    :param deadline: Progress at which the linear floor reaches ``p_end``.
    :param p_start: Initial mask probability.
    :param p_end: Maximum mask probability.
    """

    def __init__(
        self,
        *,
        step: float = 0.1,
        tolerance: float = 0.1,
        patience: int = 3,
        smoothing: float = 0.3,
        return_scale: float = 1.0,
        deadline: float | None = 0.9,
        p_start: float = 0.0,
        p_end: float = 1.0,
    ) -> None:
        if not 0.0 < step <= 1.0:
            raise ValueError(f"Expected 0 < step <= 1, got {step}")
        if patience < 1:
            raise ValueError(f"Expected patience >= 1, got {patience}")
        self.step, self.tolerance, self.patience = step, tolerance, patience
        self.smoothing, self.return_scale, self.deadline = smoothing, return_scale, deadline
        self.p_start, self.p_end = p_start, p_end
        self.gated = p_start
        self.progress = 0.0
        self.smoothed: float | None = None
        self.initial: float | None = None
        self.armed = False
        self.reference = -math.inf
        self.window: list[float] = []
        self.steps_up = 0
        self.steps_down = 0

    def mask_probability(self, progress: float | None = None) -> float:
        progress = self.progress if progress is None else progress
        floor = self.p_end * min(progress / self.deadline, 1.0) if self.deadline else self.p_start
        return min(max(self.gated, floor), self.p_end)

    def set_progress(self, progress: float) -> None:
        self.progress = progress

    def _margin(self, value: float) -> float:
        return self.tolerance * max(abs(value), self.return_scale)

    def observe_return(self, mean_return: float) -> None:
        """Feed the mean training return of the last rollout and apply the gate."""
        r = mean_return if self.smoothed is None else self.smoothing * mean_return + (1 - self.smoothing) * self.smoothed
        self.smoothed = r
        if self.initial is None:
            self.initial = r
        if not self.armed:
            self.armed = r >= self.initial + 2 * self._margin(self.initial)
            if not self.armed:
                return
        if self.reference == -math.inf:
            self.reference = r
        self.window = (self.window + [r])[-self.patience :]
        margin = self._margin(self.reference)
        if r < self.reference - 2 * margin and self.gated > self.p_start:
            self.gated = max(self.p_start, self.gated - self.step)
            self.steps_down += 1
            self._rebase(r)
        elif len(self.window) == self.patience and all(abs(v - self.reference) <= margin for v in self.window):
            if self.gated < self.p_end:
                self.gated = min(self.p_end, self.gated + self.step)
                self.steps_up += 1
            self._rebase(r)
        elif r > self.reference + margin:  # still improving at this masking level: follow the return up
            self.reference = r
            self.window = [r]

    def _rebase(self, r: float) -> None:
        self.reference = r
        self.window = []


class FixedMaskCurriculum:
    """Constant mask probability ``p`` (``0``: always visible, ``1``: always masked)."""

    def __init__(self, p: float = 1.0) -> None:
        self.p = p

    def mask_probability(self, progress: float | None = None) -> float:
        return self.p

    def set_progress(self, progress: float) -> None:
        pass


class TwoPhaseMaskCurriculum:
    """RMA: no episode masked before ``switch`` (fraction of training), every episode masked after it."""

    def __init__(self, switch: float = 0.75) -> None:
        self.switch, self.progress = switch, 0.0

    def mask_probability(self, progress: float | None = None) -> float:
        return float((self.progress if progress is None else progress) >= self.switch)

    def set_progress(self, progress: float) -> None:
        self.progress = progress


CURRICULA = {
    "SoftmaxMaskCurriculum": SoftmaxMaskCurriculum,
    "RewardGatedMaskCurriculum": RewardGatedMaskCurriculum,
    "FixedMaskCurriculum": FixedMaskCurriculum,
    "TwoPhaseMaskCurriculum": TwoPhaseMaskCurriculum,
}


def make_curriculum(spec):
    """Build a curriculum from a config entry: ``None``, an instance, or ``{"class": <name>, **kwargs}`` where
    ``<name>`` is a key of :data:`CURRICULA` (case-insensitive)."""
    if spec is None or not isinstance(spec, dict):
        return spec
    kwargs = dict(spec)
    name = kwargs.pop("class", "SoftmaxMaskCurriculum")
    classes = {k.lower(): v for k, v in CURRICULA.items()}
    if name.lower() not in classes:
        raise ValueError(f"unknown curriculum '{name}', expected one of {list(CURRICULA)}")
    return classes[name.lower()](**kwargs)
