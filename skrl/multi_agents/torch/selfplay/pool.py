"""Frozen policy snapshots, opponent pools and Elo ratings for self-play."""

from __future__ import annotations

from typing import Any, Callable

import copy
import dataclasses
import itertools

import torch

_snapshot_ids = itertools.count()


def elo_expected(rating_a: float, rating_b: float) -> float:
    """Expected score of player A against player B under the Elo model."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def elo_update(rating_a: float, rating_b: float, score_a: float, k: float = 32.0) -> tuple[float, float]:
    """Zero-sum Elo update for one match-up.

    :param score_a: Mean score of A in ``[0, 1]`` over the games played against B (1: A always won).

    :return: New ratings of A and B.
    """
    delta = k * (score_a - elo_expected(rating_a, rating_b))
    return rating_a + delta, rating_b - delta


def clone_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Detached CPU copy of a module's state dict."""
    return {k: v.detach().to("cpu", copy=True) for k, v in module.state_dict().items()}


@dataclasses.dataclass
class PolicySnapshot:
    """Frozen copy of one team's policies at a given generation."""

    team: str
    """Team name (e.g. ``"pursuers"`` or ``"evader"``)."""
    generation: int
    """Generation at which the snapshot was taken (0: initial random policies)."""
    policies: dict[str, dict[str, torch.Tensor]]
    """Policy state dicts, keyed by agent name."""
    values: dict[str, dict[str, torch.Tensor]] = dataclasses.field(default_factory=dict)
    """Value state dicts (only used to restore a learner, e.g. by population-based exploit)."""
    rating: float = 1000.0
    """Elo rating."""
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)
    """Free-form metadata (evaluation results, hyper-parameters, ...)."""
    id: int = dataclasses.field(default_factory=lambda: next(_snapshot_ids))
    """Unique, stable identifier (pool indices may shift when entries are evicted)."""
    # statistics of the *current learner* against this snapshot (used by samplers such as PFSP)
    learner_games: int = 0
    """Number of finished games of the learner against this snapshot."""
    learner_wins: float = 0.0
    """Cumulative learner score against this snapshot."""
    learner_win_rate_ema: float | None = None
    """Exponential moving average of the learner's score against this snapshot (``None`` if never played)."""

    def learner_win_rate(self, prior: float = 0.5) -> float:
        """Estimated learner win probability against this snapshot."""
        return prior if self.learner_win_rate_ema is None else self.learner_win_rate_ema


class OpponentPool:
    def __init__(
        self,
        team: str,
        agent_names: list[str],
        *,
        max_size: int | None = None,
        ema_decay: float = 0.05,
    ) -> None:
        """Pool of frozen snapshots of one team, used as opponents of the other team's learner.

        :param team: Team name.
        :param agent_names: Agent names belonging to the team.
        :param max_size: Maximum number of snapshots (oldest non-best snapshot is evicted). Unbounded if None.
        :param ema_decay: Step size of the learner win-rate moving average (per finished game).
        """
        self.team = team
        self.agent_names = list(agent_names)
        self.max_size = max_size
        self.ema_decay = ema_decay
        self.entries: list[PolicySnapshot] = []
        self._model_cache: dict[int, dict[str, torch.nn.Module]] = {}

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> PolicySnapshot:
        return self.entries[index]

    def __iter__(self):
        return iter(self.entries)

    @property
    def ratings(self) -> torch.Tensor:
        return torch.tensor([entry.rating for entry in self.entries], dtype=torch.float64)

    @property
    def learner_win_rates(self) -> torch.Tensor:
        return torch.tensor([entry.learner_win_rate() for entry in self.entries], dtype=torch.float64)

    def best_index(self) -> int:
        """Index of the highest-rated snapshot (latest one on ties)."""
        if not self.entries:
            raise IndexError("Empty pool")
        ratings = self.ratings
        best = ratings.max()
        return int((ratings == best).nonzero().max())

    def add(self, snapshot: PolicySnapshot) -> int:
        """Add a snapshot (evicting the oldest non-best one if the pool is full).

        :return: Index of the new snapshot.
        """
        self.entries.append(snapshot)
        if self.max_size is not None and len(self.entries) > self.max_size:
            best_id = self.entries[self.best_index()].id
            for i, entry in enumerate(self.entries[:-1]):
                if entry.id != best_id:
                    self._model_cache.pop(entry.id, None)
                    del self.entries[i]
                    break
        return len(self.entries) - 1

    def reset_learner_stats(self) -> None:
        """Forget the learner-vs-snapshot statistics (e.g. when the learner changes)."""
        for entry in self.entries:
            entry.learner_games = 0
            entry.learner_wins = 0.0
            entry.learner_win_rate_ema = None

    def record(self, indices: torch.Tensor, learner_scores: torch.Tensor) -> None:
        """Record finished games of the learner against snapshots of this pool.

        :param indices: Pool indices of the opponents (one per game).
        :param learner_scores: Learner score per game in ``[0, 1]`` (1: learner won).
        """
        for index, score in zip(indices.tolist(), learner_scores.tolist()):
            entry = self.entries[index]
            entry.learner_games += 1
            entry.learner_wins += score
            if entry.learner_win_rate_ema is None:
                entry.learner_win_rate_ema = float(score)
            else:
                entry.learner_win_rate_ema += self.ema_decay * (score - entry.learner_win_rate_ema)

    def policies(self, index: int, policy_factory: Callable[[str], torch.nn.Module]) -> dict[str, torch.nn.Module]:
        """Instantiated (cached) frozen policies of a snapshot, keyed by agent name.

        :param policy_factory: Callable returning a fresh policy model for a given agent name.
        """
        entry = self.entries[index]
        if entry.id not in self._model_cache:
            models = {}
            for uid in self.agent_names:
                model = policy_factory(uid)
                model.load_state_dict(entry.policies[uid])
                model.eval()
                for parameter in model.parameters():
                    parameter.requires_grad_(False)
                models[uid] = model
            self._model_cache[entry.id] = models
        return self._model_cache[entry.id]

    def summary(self) -> list[dict[str, Any]]:
        return [
            {
                "generation": entry.generation,
                "rating": round(entry.rating, 1),
                "learner_games": entry.learner_games,
                "learner_win_rate": (
                    None if entry.learner_win_rate_ema is None else round(entry.learner_win_rate_ema, 3)
                ),
            }
            for entry in self.entries
        ]

    def state_dict(self) -> dict[str, Any]:
        return {
            "team": self.team,
            "agent_names": self.agent_names,
            "entries": [copy.copy(dataclasses.asdict(entry)) for entry in self.entries],
        }
