"""Histogram (grid) belief and the expected information gain of Hsiao, Lozano-Pérez and Kaelbling, "Robust Belief-Based
Execution of Manipulation Programs" (WAFR 2008).

The belief is a dense array over a regular grid of hidden states. Some grid dimensions move under actions (e.g. an
object pose): after an action each grid state ``w`` goes to a continuous next state, whose mass is spread over the
neighbouring cells (multilinear splatting). The others (e.g. physics parameters) are static.
"""
from __future__ import annotations

import itertools

import numpy as np


def entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64).ravel()
    p = p[p > 0] / p.sum()
    return float(-(p * np.log(p)).sum())


def expected_information_gain(
    weights: np.ndarray, log_likelihood: np.ndarray, next_cells: np.ndarray | None = None
) -> float:
    """WAFR08 information gain of an action on a belief support:
    ``IG = H(b) - sum_j b(w_j) H(b | o_j)``, with ``o_j`` the observation expected if ``w_j`` is the true state and
    ``log_likelihood[i, j] = log P(o_j | w_i)``.

    :param next_cells: Optional cell index of each state after the action: the posterior entropy is then that of the
        next-state belief, so actions that scatter the states (e.g. pushes) lose the information they disturb.
    """
    b = np.asarray(weights, dtype=np.float64)
    b = b / b.sum()
    log_post = np.log(np.maximum(b, 1e-300))[:, None] + log_likelihood
    log_post -= log_post.max(axis=0, keepdims=True)
    post = np.exp(log_post)
    post /= post.sum(axis=0, keepdims=True)
    if next_cells is not None:
        _, inverse = np.unique(next_cells, return_inverse=True)
        onehot = np.zeros((len(b), inverse.max() + 1))
        onehot[np.arange(len(b)), inverse] = 1.0
        post = onehot.T @ post  # [cells, M]
    h = -(post * np.log(np.maximum(post, 1e-300))).sum(axis=0)
    return entropy(b) - float((b * h).sum())


class GridBelief:
    """Belief over the cells of a regular grid.

    :param axes: Grid values per dimension (each evenly spaced).
    :param moving: Dimensions that change under actions (splatted); the others keep their grid index.
    :param periods: Period per periodic moving dimension, e.g. ``{2: pi / 2}`` for a square's yaw.
    """

    def __init__(self, axes: list[np.ndarray], moving: tuple[int, ...] = (), periods: dict[int, float] | None = None):
        self.axes = [np.asarray(a, dtype=np.float64) for a in axes]
        self.shape = tuple(len(a) for a in self.axes)
        self.moving = tuple(moving)
        self.periods = periods or {}
        self.steps = [float(a[1] - a[0]) if len(a) > 1 else 1.0 for a in self.axes]
        self.p = np.full(self.shape, 1.0 / np.prod(self.shape))

    def set(self, p: np.ndarray) -> None:
        self.p = np.asarray(p, dtype=np.float64) / np.sum(p)

    # --- queries --------------------------------------------------------------------------------------------------

    def values(self, flat: np.ndarray) -> np.ndarray:
        """Grid values ``[K, D]`` of flat cell indices."""
        idx = np.unravel_index(flat, self.shape)
        return np.stack([a[i] for a, i in zip(self.axes, idx)], axis=-1)

    def support(self, mass: float = 0.999, max_cells: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Most likely cells holding ``mass`` of the belief (at most ``max_cells``): flat indices and weights."""
        flat = self.p.ravel()
        nz = np.flatnonzero(flat)
        order = nz[np.argsort(-flat[nz], kind="stable")]
        cum = np.cumsum(flat[order])
        n = int(np.searchsorted(cum, mass * cum[-1]) + 1)
        if max_cells is not None:
            n = min(n, max_cells)
        return order[:n], flat[order[:n]]

    def nearest(self, values: np.ndarray) -> np.ndarray:
        """Flat index of the cell nearest to each of the values ``[K, D]`` (periodic dims wrapped, others clipped)."""
        idx = []
        for d, a in enumerate(self.axes):
            i = np.rint((values[:, d] - a[0]) / self.steps[d]).astype(np.int64)
            idx.append(np.mod(i, self.shape[d]) if d in self.periods else np.clip(i, 0, self.shape[d] - 1))
        return np.ravel_multi_index(tuple(idx), self.shape)

    def map_cell(self) -> int:
        return int(np.argmax(self.p))

    def marginal(self, dims: tuple[int, ...]) -> np.ndarray:
        others = tuple(d for d in range(len(self.shape)) if d not in dims)
        return self.p.sum(axis=others)

    def entropy(self) -> float:
        return entropy(self.p)

    # --- update ---------------------------------------------------------------------------------------------------

    def update(
        self, flat: np.ndarray, log_likelihood: np.ndarray, next_values: np.ndarray, keep: np.ndarray | None = None
    ) -> float:
        """Bayes update on the cells ``flat`` (mass outside them is dropped): weight by the likelihood, move each cell
        to its next state (``next_values[:, moving]``) and splat it on the grid.

        :param keep: Cells whose mass survives the action (``False``: e.g. the object fell off the table).
        :return: The probability that the observation was explained (``sum_i b_i P(o | w_i)``), log-scale safe.
        """
        b = self.p.ravel()[flat]
        logw = np.log(np.maximum(b, 1e-300)) + log_likelihood
        if keep is not None:
            logw = np.where(keep, logw, -np.inf)
        top = logw.max()
        w = np.exp(logw - top)
        evidence = float(np.log(w.sum()) + top)
        w /= w.sum()
        idx = list(np.unravel_index(flat, self.shape))
        corners, weights = [], []
        for d in self.moving:
            f = (next_values[:, d] - self.axes[d][0]) / self.steps[d]
            lo = np.floor(f)
            frac = f - lo
            lo = lo.astype(np.int64)
            corners.append((lo, lo + 1))
            weights.append((1.0 - frac, frac))
        new = np.zeros(self.shape)
        for choice in itertools.product((0, 1), repeat=len(self.moving)):
            cell = list(idx)
            cw = w.copy()
            for k, d in enumerate(self.moving):
                i = corners[k][choice[k]]
                if d in self.periods:
                    i = np.mod(i, self.shape[d])
                else:
                    cw = np.where((i >= 0) & (i < self.shape[d]), cw, 0.0)  # mass leaving the grid is dropped
                    i = np.clip(i, 0, self.shape[d] - 1)
                cell[d] = i
                cw = cw * weights[k][choice[k]]
            np.add.at(new, tuple(cell), cw)
        if new.sum() <= 0:
            raise RuntimeError("the belief has no mass left on the grid")
        self.p = new / new.sum()
        return evidence

    def blur(self, kernel=(0.1, 0.8, 0.1)) -> None:
        """Diffuse the moving dimensions (process noise)."""
        k = np.asarray(kernel)
        for d in self.moving:
            p = np.moveaxis(self.p, d, 0)
            out = k[1] * p
            if d in self.periods:
                out += k[0] * np.roll(p, 1, axis=0) + k[2] * np.roll(p, -1, axis=0)
            else:
                out[1:] += k[0] * p[:-1]
                out[:-1] += k[2] * p[1:]
            self.p = np.moveaxis(out, 0, d)
        self.p /= self.p.sum()
