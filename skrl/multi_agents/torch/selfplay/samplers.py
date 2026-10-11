"""Opponent sampling strategies for self-play.

A sampler maps an :class:`~skrl.multi_agents.torch.selfplay.pool.OpponentPool` to a probability distribution over its
snapshots (:meth:`OpponentSampler.probabilities`) and draws opponent indices from it (:meth:`OpponentSampler.sample`).
Adaptive samplers additionally learn from finished games (:meth:`OpponentSampler.update`) and from the end-of-generation
summary (:meth:`OpponentSampler.on_generation_end`).

Registered strategies (``make_sampler(name, **kwargs)``):

* ``"last"``: latest snapshot only (naive self-play).
* ``"best"``: highest Elo-rated snapshot.
* ``"uniform"``: uniform over all snapshots (fictitious self-play).
* ``"hedge"``: Hedge / exponential weights (Exp3 flavour) that upweights opponents that beat the learner.
* ``"adaptive_pbt"``: adaptive prioritized fictitious self-play (PFSP) whose weighting adapts to the learner's
  progress (the runner can additionally perform population-based exploit/explore of the learner, see ``SelfPlayCfg``).
"""

from __future__ import annotations

from typing import Any, Callable

import math

import torch

from skrl.multi_agents.torch.selfplay.pool import OpponentPool

SAMPLERS: dict[str, type["OpponentSampler"]] = {}


def register_sampler(name: str) -> Callable[[type], type]:
    """Class decorator that registers an opponent sampler under ``name``."""

    def decorator(cls: type) -> type:
        if name in SAMPLERS:
            raise KeyError(f"Sampler '{name}' is already registered")
        SAMPLERS[name] = cls
        cls.name = name
        return cls

    return decorator


def make_sampler(name: str, **kwargs: Any) -> "OpponentSampler":
    """Instantiate the sampler registered under ``name``."""
    if name not in SAMPLERS:
        raise KeyError(f"Unknown sampler '{name}'. Available: {sorted(SAMPLERS)}")
    return SAMPLERS[name](**kwargs)


class OpponentSampler:
    """Base class of the opponent sampling strategies."""

    name: str = ""

    def probabilities(self, pool: OpponentPool) -> torch.Tensor:
        """Sampling distribution over the pool's snapshots (``float64`` tensor of size ``len(pool)``, sums to 1)."""
        raise NotImplementedError

    def sample(self, pool: OpponentPool, num_samples: int, generator: torch.Generator | None = None) -> torch.Tensor:
        """Draw ``num_samples`` opponent indices (with replacement).

        :return: ``int64`` CPU tensor of pool indices.
        """
        if not len(pool):
            raise ValueError("Cannot sample from an empty opponent pool")
        if num_samples <= 0:
            return torch.zeros(0, dtype=torch.long)
        probabilities = self.probabilities(pool)
        return torch.multinomial(probabilities, num_samples, replacement=True, generator=generator)

    def update(self, pool: OpponentPool, indices: torch.Tensor, learner_scores: torch.Tensor) -> None:
        """Learn from finished games (no-op by default).

        :param indices: Pool indices of the opponents (one per game).
        :param learner_scores: Learner score per game in ``[0, 1]`` (1: learner won).
        """

    def on_generation_end(self, pool: OpponentPool, learner_win_rate: float | None) -> None:
        """Hook called at the end of each generation (no-op by default).

        :param learner_win_rate: Mean learner score over the generation's training games (None if no game ended).
        """

    def state(self) -> dict[str, Any]:
        """Internal state (for logging)."""
        return {}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.state()})"


def _one_hot(size: int, index: int) -> torch.Tensor:
    probabilities = torch.zeros(size, dtype=torch.float64)
    probabilities[index] = 1.0
    return probabilities


@register_sampler("last")
class LastSampler(OpponentSampler):
    """(a) Play only against the latest generation."""

    def probabilities(self, pool: OpponentPool) -> torch.Tensor:
        return _one_hot(len(pool), len(pool) - 1)


@register_sampler("best")
class BestSampler(OpponentSampler):
    """(b) Play only against the best (highest Elo) generation."""

    def probabilities(self, pool: OpponentPool) -> torch.Tensor:
        return _one_hot(len(pool), pool.best_index())


@register_sampler("uniform")
class UniformSampler(OpponentSampler):
    """(c) Uniform sample of all past generations."""

    def probabilities(self, pool: OpponentPool) -> torch.Tensor:
        return torch.full((len(pool),), 1.0 / len(pool), dtype=torch.float64)


@register_sampler("hedge")
class HedgeSampler(OpponentSampler):
    def __init__(
        self,
        *,
        eta: float = 0.1,
        gamma: float = 0.05,
        top_k: int | None = None,
        importance_weighting: bool = True,
    ) -> None:
        """(d) Hedge / multiplicative-weights sampling focused on the strongest generations.

        Each snapshot ``i`` keeps a cumulative *gain* ``G_i`` = estimated total loss it inflicted on the learner.
        The sampling distribution is the exponential-weights distribution mixed with uniform exploration::

            p_i = (1 - gamma) * exp(eta * G_i) / sum_j exp(eta * G_j) + gamma / K

        After a batch of ``N`` finished games, with ``n_i`` games and mean learner score ``s_i`` against ``i``,
        the importance-weighted (Exp3) loss estimate ``l_i = (n_i / N) * (1 - s_i) / p_i`` is added to ``G_i``
        (an unbiased estimate of ``1 - s_i``, since ``E[n_i / N] = p_i``). Opponents that keep beating the learner
        therefore get exponentially more weight, while opponents the learner has mastered fade out.

        New snapshots are initialized optimistically with ``G = max_j G_j`` so they are explored immediately.

        :param eta: Learning rate of the exponential weights.
        :param gamma: Uniform exploration mixing coefficient.
        :param top_k: Restrict the support to the ``top_k`` highest-rated snapshots ("best generations").
            All snapshots if None.
        :param importance_weighting: Use the Exp3 importance-weighted estimate (otherwise the plain loss).
        """
        self.eta = eta
        self.gamma = gamma
        self.top_k = top_k
        self.importance_weighting = importance_weighting
        self.gains: dict[int, float] = {}
        self._last_probabilities: dict[int, float] = {}

    def _support(self, pool: OpponentPool) -> torch.Tensor:
        mask = torch.ones(len(pool), dtype=torch.bool)
        if self.top_k is not None and self.top_k < len(pool):
            mask[:] = False
            # sort by rating, ties broken towards recent snapshots
            order = sorted(range(len(pool)), key=lambda i: (pool[i].rating, i), reverse=True)
            mask[order[: self.top_k]] = True
        return mask

    def probabilities(self, pool: OpponentPool) -> torch.Tensor:
        known = [self.gains[entry.id] for entry in pool if entry.id in self.gains]
        initial = max(known) if known else 0.0
        for entry in pool:
            self.gains.setdefault(entry.id, initial)
        support = self._support(pool)
        gains = torch.tensor([self.gains[entry.id] for entry in pool], dtype=torch.float64)
        logits = torch.where(support, self.eta * gains, torch.full_like(gains, -math.inf))
        weights = torch.softmax(logits, dim=0)
        probabilities = (1.0 - self.gamma) * weights + self.gamma * support.double() / support.sum()
        self._last_probabilities = {entry.id: float(p) for entry, p in zip(pool, probabilities)}
        return probabilities

    def update(self, pool: OpponentPool, indices: torch.Tensor, learner_scores: torch.Tensor) -> None:
        if not indices.numel():
            return
        probabilities = self.probabilities(pool)
        total = indices.numel()
        for index in indices.unique().tolist():
            mask = indices == index
            loss = float((1.0 - learner_scores[mask].double()).mean())
            if self.importance_weighting:
                estimate = (mask.sum().item() / total) * loss / max(float(probabilities[index]), 1e-8)
            else:
                estimate = loss
            self.gains[pool[index].id] += estimate
        # keep the gains bounded (the distribution is invariant to a common shift)
        offset = min(self.gains[entry.id] for entry in pool)
        for key in self.gains:
            self.gains[key] -= offset

    def state(self) -> dict[str, Any]:
        return {"eta": self.eta, "gamma": self.gamma, "top_k": self.top_k}


@register_sampler("adaptive_pbt")
class AdaptivePBTSampler(OpponentSampler):
    def __init__(
        self,
        *,
        weighting: str = "hard",
        exponent: float = 2.0,
        epsilon: float = 0.05,
        upper_win_rate: float = 0.7,
        lower_win_rate: float = 0.3,
        adaptation_factor: float = 1.5,
        min_exponent: float = 1.0,
        max_exponent: float = 8.0,
        prior: float = 0.5,
    ) -> None:
        """(e) Adaptive population-based sampling: prioritized fictitious self-play (PFSP) with adaptive priority.

        Each snapshot ``i`` gets the weight ``f(x_i) + epsilon_floor`` where ``x_i`` is the learner's estimated win
        probability against it (exponential moving average of game outcomes kept by the pool, ``prior`` if unseen):

        * ``"hard"``: ``f(x) = (1 - x)^p`` -- focus on the opponents the learner struggles against.
        * ``"variance"``: ``f(x) = x * (1 - x)`` -- focus on evenly-matched opponents (useful when the learner is
          losing against most of the population and needs a curriculum).

        The weighting adapts at the end of every generation from the learner's overall training win rate ``w``:

        * ``w > upper_win_rate``: learner dominates -> switch to ``"hard"`` and sharpen it (``p *= adaptation_factor``).
        * ``w < lower_win_rate``: learner struggles -> switch to ``"variance"`` and soften ``p``.
        * otherwise keep the current setting.

        The final distribution is mixed with ``epsilon`` uniform exploration over the whole population. Together with
        ``SelfPlayCfg.pbt`` (population-based exploit/explore of the learner's weights and hyper-parameters) this is
        the "adaptive population-based" strategy.
        """
        if weighting not in ("hard", "variance"):
            raise ValueError(f"Unknown PFSP weighting: {weighting}")
        self.weighting = weighting
        self.exponent = exponent
        self.epsilon = epsilon
        self.upper_win_rate = upper_win_rate
        self.lower_win_rate = lower_win_rate
        self.adaptation_factor = adaptation_factor
        self.min_exponent = min_exponent
        self.max_exponent = max_exponent
        self.prior = prior

    def priority(self, win_rates: torch.Tensor) -> torch.Tensor:
        if self.weighting == "hard":
            return (1.0 - win_rates).clamp(min=0.0) ** self.exponent
        return win_rates * (1.0 - win_rates)

    def probabilities(self, pool: OpponentPool) -> torch.Tensor:
        win_rates = torch.tensor([entry.learner_win_rate(self.prior) for entry in pool], dtype=torch.float64)
        weights = self.priority(win_rates) + 1e-6
        weights = weights / weights.sum()
        return (1.0 - self.epsilon) * weights + self.epsilon / len(pool)

    def on_generation_end(self, pool: OpponentPool, learner_win_rate: float | None) -> None:
        if learner_win_rate is None:
            return
        if learner_win_rate > self.upper_win_rate:
            self.weighting = "hard"
            self.exponent = min(self.exponent * self.adaptation_factor, self.max_exponent)
        elif learner_win_rate < self.lower_win_rate:
            self.weighting = "variance"
            self.exponent = max(self.exponent / self.adaptation_factor, self.min_exponent)

    def state(self) -> dict[str, Any]:
        return {"weighting": self.weighting, "exponent": round(self.exponent, 3), "epsilon": self.epsilon}
