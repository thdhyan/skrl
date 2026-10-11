"""Generation-based self-play runner for two-team (zero-sum) multi-agent environments.

Each team (e.g. ``"pursuers"`` and ``"evader"``) has

* a **learner**: a skrl :class:`~skrl.multi_agents.torch.mappo.MAPPO` or :class:`~skrl.multi_agents.torch.ippo.IPPO`
  instance controlling the team's agents, and
* an **opponent pool**: frozen snapshots of the team's policies, one per generation (generation 0 being the initial,
  untrained policies), each with an Elo rating.

Every generation:

1. **Train** the learning team(s) for ``timesteps_per_generation`` environment steps. Each sub-environment plays
   against an opponent snapshot drawn from the other team's pool by the configured sampler; a new opponent is drawn
   for a sub-environment whenever its episode ends (per-episode opponent sampling). Game outcomes feed the pool
   statistics and adaptive samplers.
2. **Evaluate** the learner against (up to ``eval_max_opponents``) snapshots of the opposing pool -- the most recent
   ones plus the best-rated one -- and update the Elo ratings.
3. **Snapshot** the learner into its own pool (with its rating and evaluation metadata).
4. Optionally perform **population-based exploit/explore** (``pbt``): the best-rated snapshot of the learner's own
   pool is evaluated against the same opponents; if the new snapshot scores clearly worse (``pbt_margin``), the learner
   restarts from the best snapshot's weights (exploit) and its learning rate and entropy coefficient are randomly
   perturbed (explore).

Training schemes:

* ``"alternating"``: one team learns per generation (``first_learner`` first, then alternating) on all
  sub-environments while the other team is frozen.
* ``"simultaneous"``: both teams learn during every generation; the sub-environments are split in two halves, the
  first half trains the pursuers' learner against the evader pool and the second half trains the evader's learner
  against the pursuer pool.
"""

from __future__ import annotations

from typing import Any, Callable

import copy
import dataclasses
import json
import math
import os
import random
import time

import torch

from skrl import logger
from skrl.memories.torch import RandomMemory
from skrl.multi_agents.torch.selfplay.models import MLPCategoricalPolicy, MLPValue
from skrl.multi_agents.torch.selfplay.pool import OpponentPool, PolicySnapshot, clone_state_dict, elo_update
from skrl.multi_agents.torch.selfplay.samplers import OpponentSampler, make_sampler
from skrl.trainers.torch.sequential import SequentialTrainerCfg

DEFAULT_AGENT_CFG: dict[str, Any] = {
    "rollouts": 16,
    "learning_epochs": 4,
    "mini_batches": 2,
    "discount_factor": 0.99,
    "gae_lambda": 0.95,
    "learning_rate": 3e-4,
    "entropy_loss_scale": 0.01,
    "value_loss_scale": 1.0,
    "grad_norm_clip": 0.5,
    "experiment": {"write_interval": 0, "checkpoint_interval": 0},
}


@dataclasses.dataclass(kw_only=True)
class SelfPlayCfg:
    """Configuration of the self-play runner."""

    generations: int = 10
    """Number of generations (G)."""
    timesteps_per_generation: int = 2000
    """Number of (vectorized) environment steps of training per generation and learning team."""
    algorithm: str = "mappo"
    """Learning algorithm: ``"mappo"`` (centralized critic on the global state) or ``"ippo"``."""
    sampler: str = "uniform"
    """Opponent sampling strategy: ``"last"``, ``"best"``, ``"uniform"``, ``"hedge"``, ``"adaptive_pbt"``."""
    sampler_kwargs: dict = dataclasses.field(default_factory=dict)
    """Keyword arguments of the sampler's constructor."""
    training_scheme: str = "alternating"
    """``"alternating"`` (one team learns per generation) or ``"simultaneous"`` (both teams every generation)."""
    first_learner: str = "pursuers"
    """Team that learns first in the alternating scheme."""
    learning_teams: list[str] | None = None
    """Teams that learn (default: all). The other teams keep their initial (random) policy."""
    agent_cfg: dict = dataclasses.field(default_factory=dict)
    """Overrides of the MAPPO/IPPO configuration (see ``DEFAULT_AGENT_CFG``)."""
    hidden_sizes: tuple[int, ...] = (64, 64)
    """Hidden layer sizes of the MLP policy and value models."""
    eval_episodes: int = 32
    """Number of evaluation episodes against each evaluated opponent."""
    eval_max_opponents: int = 4
    """Maximum number of opponents evaluated per generation (the most recent ones and the best-rated one)."""
    eval_deterministic: bool = False
    """Whether the learner acts greedily during evaluation (stochastic by default: grid games can loop)."""
    opponent_deterministic: bool = False
    """Whether the frozen opponents act greedily."""
    elo_k: float = 32.0
    """Elo K-factor."""
    initial_rating: float = 1000.0
    """Elo rating of the generation-0 snapshots."""
    pool_max_size: int | None = None
    """Maximum snapshots per pool (unbounded if None)."""
    pbt: bool | None = None
    """Population-based exploit/explore of the learner (default: enabled only for the ``"adaptive_pbt"`` sampler)."""
    pbt_margin: float = 0.1
    """Exploit when the new snapshot's mean evaluation score is more than this below the score of the best-rated
    snapshot of its own pool, both evaluated against the same opponents."""
    pbt_perturb_factors: tuple[float, float] = (0.8, 1.25)
    """Multiplicative factors randomly applied to the learning rate and entropy coefficient on explore."""
    seed: int | None = None
    """Seed of the runner's random generators (opponent sampling and PBT perturbations)."""
    checkpoint_dir: str | None = None
    """If set, snapshots (``{team}_gen{g}.pt``) and ``history.json`` are written there."""
    verbose: bool = True
    """Log a summary of every generation."""


@dataclasses.dataclass
class _Assignment:
    """A learning team and the slice of sub-environments it trains on."""

    team: str
    start: int
    stop: int

    @property
    def slice(self) -> slice:
        return slice(self.start, self.stop)

    @property
    def size(self) -> int:
        return self.stop - self.start


class SelfPlayRunner:
    def __init__(self, env: Any, cfg: SelfPlayCfg | dict | None = None, teams: dict[str, list[str]] | None = None):
        """Self-play runner.

        :param env: Multi-agent environment (skrl ``MultiAgentEnvWrapper``-compatible; a raw
            :class:`~skrl.envs.pursuit_evasion.PursuitEvasionEnv` is wrapped automatically). It must reset its
            sub-environments automatically and expose ``num_envs > 1``.
        :param cfg: Runner configuration.
        :param teams: Mapping of exactly two team names to agent names. Read from ``env.teams`` if not given.
        """
        if cfg is None:
            cfg = SelfPlayCfg()
        elif isinstance(cfg, dict):
            cfg = SelfPlayCfg(**cfg)
        self.cfg = cfg

        if not hasattr(env, "state_spaces"):  # raw environment
            from skrl.envs.wrappers.torch import wrap_env

            env = wrap_env(env, wrapper="pursuit-evasion", verbose=False)
        self.env = env
        self.device = env.device
        self.num_envs = env.num_envs

        self.teams = {k: list(v) for k, v in (teams if teams is not None else env.teams).items()}
        if len(self.teams) != 2:
            raise ValueError(f"Exactly two teams are required, got {list(self.teams)}")
        names = list(self.teams)
        self.opponent_of = {names[0]: names[1], names[1]: names[0]}
        self.learning_teams = list(cfg.learning_teams) if cfg.learning_teams is not None else names
        for team in self.learning_teams + [cfg.first_learner]:
            if team not in self.teams:
                raise ValueError(f"Unknown team '{team}'. Teams: {names}")
        if cfg.training_scheme not in ("alternating", "simultaneous"):
            raise ValueError(f"Unknown training scheme: {cfg.training_scheme}")
        if cfg.algorithm not in ("mappo", "ippo"):
            raise ValueError(f"Unknown algorithm: {cfg.algorithm}")

        self._rng = random.Random(cfg.seed)
        self._generator = torch.Generator()
        if cfg.seed is not None:
            self._generator.manual_seed(cfg.seed)
            torch.manual_seed(cfg.seed)

        self.pbt = cfg.pbt if cfg.pbt is not None else cfg.sampler == "adaptive_pbt"
        self.agent_cfg = self._merge_agent_cfg(cfg.agent_cfg)

        # environment slices per learning team
        self.simultaneous = cfg.training_scheme == "simultaneous" and len(self.learning_teams) == 2
        if self.simultaneous:
            if self.num_envs < 2:
                raise ValueError("The simultaneous scheme requires at least 2 sub-environments")
            half = self.num_envs // 2
            spans = {names[0]: (0, half), names[1]: (half, self.num_envs)}
        else:
            spans = {team: (0, self.num_envs) for team in self.learning_teams}
        self._spans = spans

        # learners, pools and samplers (one sampler per pool: it chooses opponents from that pool)
        self.learners = {team: self._build_learner(team, *spans[team]) for team in self.learning_teams}
        self.pools = {team: OpponentPool(team, uids, max_size=cfg.pool_max_size) for team, uids in self.teams.items()}
        self.samplers: dict[str, OpponentSampler] = {
            team: make_sampler(cfg.sampler, **cfg.sampler_kwargs) for team in self.teams
        }
        self._timesteps = {team: 0 for team in self.learning_teams}
        self._total_timesteps = cfg.generations * cfg.timesteps_per_generation
        self.history: list[dict[str, Any]] = []
        self.generation = 0

        # generation 0: initial policies
        for team in self.teams:
            if team in self.learners:
                policies, values = self._learner_state(team)
            else:
                policies = {uid: clone_state_dict(self._make_policy(uid)) for uid in self.teams[team]}
                values = {}
            self.pools[team].add(
                PolicySnapshot(team=team, generation=0, policies=policies, values=values, rating=cfg.initial_rating)
            )

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_agent_cfg(overrides: dict) -> dict:
        merged = copy.deepcopy(DEFAULT_AGENT_CFG)
        for key, value in overrides.items():
            if key == "experiment" and isinstance(value, dict):
                merged["experiment"].update(value)
            else:
                merged[key] = value
        return merged

    def _make_policy(self, uid: str) -> MLPCategoricalPolicy:
        return MLPCategoricalPolicy(
            observation_space=self.env.observation_space(uid),
            state_space=self.env.state_space(uid),
            action_space=self.env.action_space(uid),
            device=self.device,
            hidden_sizes=self.cfg.hidden_sizes,
        )

    def _make_value(self, uid: str) -> MLPValue:
        return MLPValue(
            observation_space=self.env.observation_space(uid),
            state_space=self.env.state_space(uid),
            action_space=self.env.action_space(uid),
            device=self.device,
            hidden_sizes=self.cfg.hidden_sizes,
            use_states=self.cfg.algorithm == "mappo",
        )

    def _build_learner(self, team: str, start: int, stop: int):
        uids = self.teams[team]
        models = {uid: {"policy": self._make_policy(uid), "value": self._make_value(uid)} for uid in uids}
        memories = {
            uid: RandomMemory(memory_size=self.agent_cfg["rollouts"], num_envs=stop - start, device=self.device)
            for uid in uids
        }
        agent_cfg = copy.deepcopy(self.agent_cfg)
        if self.cfg.checkpoint_dir and not agent_cfg["experiment"].get("directory"):
            agent_cfg["experiment"]["directory"] = os.path.join(self.cfg.checkpoint_dir, "agents")
        agent_cfg["experiment"].setdefault("experiment_name", team)
        kwargs = dict(
            possible_agents=uids,
            models=models,
            memories=memories,
            observation_spaces={uid: self.env.observation_space(uid) for uid in uids},
            state_spaces={uid: self.env.state_space(uid) for uid in uids},
            action_spaces={uid: self.env.action_space(uid) for uid in uids},
            device=self.device,
            cfg=agent_cfg,
        )
        if self.cfg.algorithm == "mappo":
            from skrl.multi_agents.torch.mappo import MAPPO

            agent = MAPPO(**kwargs)
        else:
            from skrl.multi_agents.torch.ippo import IPPO

            agent = IPPO(**kwargs)
        agent.init(trainer_cfg=SequentialTrainerCfg(timesteps=self.cfg.generations * self.cfg.timesteps_per_generation))
        return agent

    def _learner_state(self, team: str) -> tuple[dict, dict]:
        learner = self.learners[team]
        policies = {uid: clone_state_dict(learner.policies[uid]) for uid in self.teams[team]}
        values = {uid: clone_state_dict(learner.values[uid]) for uid in self.teams[team]}
        return policies, values

    def _reset_learner_rollout(self, team: str) -> None:
        learner = self.learners[team]
        for memory in learner.memories.values():
            memory.reset()
        learner._rollout = 0

    # ------------------------------------------------------------------
    # acting
    # ------------------------------------------------------------------

    @staticmethod
    def _policy_actions(
        models: dict[str, torch.nn.Module],
        observations: dict[str, torch.Tensor],
        states: dict[str, torch.Tensor],
        mask: slice | torch.Tensor,
        deterministic: bool,
    ) -> dict[str, torch.Tensor]:
        actions = {}
        for uid, model in models.items():
            inputs = {"observations": observations[uid][mask], "states": states[uid][mask]}
            action, outputs = model.act(inputs, role="policy")
            if deterministic:
                action = outputs["net_output"].argmax(dim=-1, keepdim=True)
            actions[uid] = action
        return actions

    def _opponent_actions(
        self,
        team: str,
        indices: torch.Tensor,
        observations: dict[str, torch.Tensor],
        states: dict[str, torch.Tensor],
        env_slice: slice,
        actions: dict[str, torch.Tensor],
    ) -> None:
        """Fill ``actions[uid][env_slice]`` for the agents of ``team`` using the frozen snapshots ``indices``."""
        pool = self.pools[team]
        sub_observations = {uid: observations[uid][env_slice] for uid in self.teams[team]}
        sub_states = {uid: states[uid][env_slice] for uid in self.teams[team]}
        device_indices = indices.to(self.device)
        for index in indices.unique().tolist():
            mask = device_indices == index
            models = pool.policies(index, self._make_policy)
            for uid, action in self._policy_actions(
                models, sub_observations, sub_states, mask, self.cfg.opponent_deterministic
            ).items():
                actions[uid][env_slice][mask] = action.view(-1, 1).long()

    def _empty_actions(self) -> dict[str, torch.Tensor]:
        return {
            uid: torch.zeros((self.num_envs, 1), dtype=torch.long, device=self.device)
            for uid in self.env.possible_agents
        }

    def _learner_scores(self, team: str, terminated: torch.Tensor, done: torch.Tensor) -> torch.Tensor:
        """Learner score (1: win) of the finished games. The pursuing team (first team) wins on termination
        (capture); the other team wins on truncation (survival)."""
        first_team = next(iter(self.teams))
        captured = terminated[done]
        won = captured if team == first_team else ~captured
        return won.float().cpu()

    # ------------------------------------------------------------------
    # training / evaluation
    # ------------------------------------------------------------------

    def _assignments(self, generation: int) -> list[_Assignment]:
        if self.simultaneous:
            return [_Assignment(team, *self._spans[team]) for team in self.learning_teams]
        if len(self.learning_teams) == 1:
            team = self.learning_teams[0]
        else:
            order = [self.cfg.first_learner, self.opponent_of[self.cfg.first_learner]]
            team = order[(generation - 1) % 2]
        return [_Assignment(team, *self._spans[team])]

    def train_generation(self, assignments: list[_Assignment]) -> dict[str, dict[str, Any]]:
        """Train the learners of the given assignments for ``timesteps_per_generation`` steps."""
        cfg = self.cfg
        first_uid = self.env.possible_agents[0]
        stats = {a.team: {"games": 0, "wins": 0.0, "opponents": {}} for a in assignments}

        opponent_indices = {}
        for a in assignments:
            self._reset_learner_rollout(a.team)
            self.learners[a.team].enable_training_mode(True)
            opponent = self.opponent_of[a.team]
            opponent_indices[a.team] = self.samplers[opponent].sample(self.pools[opponent], a.size, self._generator)

        observations, infos = self.env.reset()
        states = self.env.state()
        for _ in range(cfg.timesteps_per_generation):
            actions = self._empty_actions()
            with torch.no_grad():
                for a in assignments:
                    learner, uids, sl = self.learners[a.team], self.teams[a.team], a.slice
                    learner.pre_interaction(timestep=self._timesteps[a.team], timesteps=self._total_timesteps)
                    learner_actions, _ = learner.act(
                        {uid: observations[uid][sl] for uid in uids},
                        {uid: states[uid][sl] for uid in uids},
                        timestep=self._timesteps[a.team],
                        timesteps=self._total_timesteps,
                    )
                    for uid in uids:
                        actions[uid][sl] = learner_actions[uid].view(-1, 1).long()
                    self._opponent_actions(
                        self.opponent_of[a.team], opponent_indices[a.team], observations, states, sl, actions
                    )

                next_observations, rewards, terminated, truncated, infos = self.env.step(actions)
                next_states = self.env.state()

                for a in assignments:
                    learner, uids, sl = self.learners[a.team], self.teams[a.team], a.slice
                    learner.record_transition(
                        observations={uid: observations[uid][sl] for uid in uids},
                        states={uid: states[uid][sl] for uid in uids},
                        actions={uid: actions[uid][sl] for uid in uids},
                        rewards={uid: rewards[uid][sl] for uid in uids},
                        next_observations={uid: next_observations[uid][sl] for uid in uids},
                        next_states={uid: next_states[uid][sl] for uid in uids},
                        terminated={uid: terminated[uid][sl] for uid in uids},
                        truncated={uid: truncated[uid][sl] for uid in uids},
                        infos=infos,
                        timestep=self._timesteps[a.team],
                        timesteps=self._total_timesteps,
                    )

            for a in assignments:
                self.learners[a.team].post_interaction(
                    timestep=self._timesteps[a.team], timesteps=self._total_timesteps
                )
                self._timesteps[a.team] += 1

                # bookkeeping of finished games and per-episode opponent resampling
                term = terminated[first_uid][a.slice].view(-1)
                done = (term | truncated[first_uid][a.slice].view(-1)).cpu()
                if done.any():
                    opponent = self.opponent_of[a.team]
                    pool, sampler = self.pools[opponent], self.samplers[opponent]
                    indices = opponent_indices[a.team][done]
                    scores = self._learner_scores(a.team, term.cpu(), done)
                    pool.record(indices, scores)
                    sampler.update(pool, indices, scores)
                    opponent_indices[a.team][done] = sampler.sample(pool, int(done.sum()), self._generator)
                    stats[a.team]["games"] += int(done.sum())
                    stats[a.team]["wins"] += float(scores.sum())
                    for index, score in zip(indices.tolist(), scores.tolist()):
                        generation = pool[index].generation
                        entry = stats[a.team]["opponents"].setdefault(generation, [0, 0.0])
                        entry[0] += 1
                        entry[1] += score

            observations, states = next_observations, next_states

        for a in assignments:
            self.learners[a.team].enable_training_mode(False)
            games = stats[a.team]["games"]
            stats[a.team]["win_rate"] = stats[a.team]["wins"] / games if games else None
            stats[a.team]["opponents"] = {
                g: round(w / n, 3) for g, (n, w) in sorted(stats[a.team]["opponents"].items())
            }
        return stats

    @torch.no_grad()
    def play(
        self,
        team: str,
        policies: dict[str, torch.nn.Module],
        opponent_index: int,
        episodes: int,
        deterministic: bool = False,
    ) -> float:
        """Play ``episodes`` episodes of ``team`` (given policies) against one opponent snapshot.

        :return: Mean score of ``team`` (1: win).
        """
        opponent = self.opponent_of[team]
        first_uid = self.env.possible_agents[0]
        indices = torch.full((self.num_envs,), opponent_index, dtype=torch.long)
        max_steps = getattr(getattr(self.env, "cfg", None), "max_steps", 1000)
        cap = (math.ceil(episodes / self.num_envs) + 1) * max_steps
        scores: list[float] = []

        observations, _ = self.env.reset()
        states = self.env.state()
        full = slice(0, self.num_envs)
        for _ in range(cap):
            actions = self._empty_actions()
            for uid, action in self._policy_actions(policies, observations, states, full, deterministic).items():
                actions[uid][:] = action.view(-1, 1).long()
            self._opponent_actions(opponent, indices, observations, states, full, actions)
            observations, _, terminated, truncated, _ = self.env.step(actions)
            states = self.env.state()
            term = terminated[first_uid].view(-1).cpu()
            done = term | truncated[first_uid].view(-1).cpu()
            if done.any():
                scores.extend(self._learner_scores(team, term, done).tolist())
                if len(scores) >= episodes:
                    break
        scores = scores[:episodes]
        return sum(scores) / len(scores) if scores else 0.5

    def _evaluation_opponents(self, pool: OpponentPool) -> list[int]:
        k = max(1, self.cfg.eval_max_opponents)
        indices = set(range(max(0, len(pool) - (k - 1)), len(pool))) if k > 1 else set()
        indices.add(pool.best_index())
        return sorted(indices)

    def evaluate_and_snapshot(self, team: str, generation: int) -> dict[str, Any]:
        """Evaluate the learner of ``team``, rate it, add its snapshot to the pool and apply PBT (if enabled)."""
        cfg = self.cfg
        learner = self.learners[team]
        own_pool, opponent_pool = self.pools[team], self.pools[self.opponent_of[team]]
        policies = {uid: learner.policies[uid] for uid in self.teams[team]}
        for model in policies.values():
            model.eval()

        rating = own_pool[-1].rating  # start from the previous snapshot's rating
        evaluation = {}
        for index in self._evaluation_opponents(opponent_pool):
            score = self.play(team, policies, index, cfg.eval_episodes, cfg.eval_deterministic)
            opponent_entry = opponent_pool[index]
            rating, opponent_entry.rating = elo_update(rating, opponent_entry.rating, score, cfg.elo_k)
            evaluation[opponent_entry.generation] = round(score, 3)

        policy_states, value_states = self._learner_state(team)
        snapshot = PolicySnapshot(
            team=team,
            generation=generation,
            policies=policy_states,
            values=value_states,
            rating=rating,
            metadata={"eval_win_rate": evaluation, "hyperparameters": self._hyperparameters(team)},
        )
        result: dict[str, Any] = {"eval_win_rate": evaluation, "rating": round(rating, 1)}

        if self.pbt and len(own_pool) > 1:
            # exploit/explore: compare the new snapshot head-to-head with the best-rated snapshot of its own pool on
            # the same opponents (Elo alone is not comparable across snapshots rated against different opponents)
            best = own_pool[own_pool.best_index()]
            if best.values:
                best_policies = own_pool.policies(own_pool.best_index(), self._make_policy)
                best_scores = [
                    self.play(team, best_policies, index, cfg.eval_episodes, cfg.eval_deterministic)
                    for index in self._evaluation_opponents(opponent_pool)
                ]
                new_score = sum(evaluation.values()) / len(evaluation)
                best_score = sum(best_scores) / len(best_scores)
                result["pbt"] = {"score": round(new_score, 3), "best_generation": best.generation}
                result["pbt"]["best_score"] = round(best_score, 3)
                if new_score < best_score - cfg.pbt_margin:
                    self._load_learner(team, best)
                    result["pbt"]["exploit_from_generation"] = best.generation
                    result["pbt"]["explore"] = self._perturb_hyperparameters(team)
                snapshot.metadata["pbt"] = result["pbt"]

        own_pool.add(snapshot)
        if cfg.checkpoint_dir:
            os.makedirs(cfg.checkpoint_dir, exist_ok=True)
            torch.save(dataclasses.asdict(snapshot), os.path.join(cfg.checkpoint_dir, f"{team}_gen{generation:03d}.pt"))
        return result

    def _hyperparameters(self, team: str) -> dict[str, float]:
        learner = self.learners[team]
        uid = self.teams[team][0]
        return {
            "learning_rate": learner.optimizers[uid].param_groups[0]["lr"],
            "entropy_loss_scale": learner.cfg.entropy_loss_scale[uid],
        }

    def _load_learner(self, team: str, snapshot: PolicySnapshot) -> None:
        learner = self.learners[team]
        for uid in self.teams[team]:
            learner.policies[uid].load_state_dict(snapshot.policies[uid])
            learner.values[uid].load_state_dict(snapshot.values[uid])

    def _perturb_hyperparameters(self, team: str) -> dict[str, float]:
        learner = self.learners[team]
        lr_factor = self._rng.choice(self.cfg.pbt_perturb_factors)
        entropy_factor = self._rng.choice(self.cfg.pbt_perturb_factors)
        for uid in self.teams[team]:
            for group in learner.optimizers[uid].param_groups:
                group["lr"] *= lr_factor
            learner.cfg.entropy_loss_scale[uid] *= entropy_factor
        return {"learning_rate_factor": lr_factor, "entropy_factor": entropy_factor, **self._hyperparameters(team)}

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------

    def run(self, generations: int | None = None, callback: Callable[[dict], None] | None = None) -> list[dict]:
        """Run self-play for ``generations`` generations (default: ``cfg.generations``).

        :param callback: Optional function called with each generation's record.

        :return: History (one record per generation).
        """
        generations = self.cfg.generations if generations is None else generations
        for _ in range(generations):
            self.generation += 1
            start = time.perf_counter()
            assignments = self._assignments(self.generation)
            train_stats = self.train_generation(assignments)
            record: dict[str, Any] = {"generation": self.generation, "teams": {}}
            for a in assignments:
                opponent = self.opponent_of[a.team]
                sampler = self.samplers[opponent]
                sampler.on_generation_end(self.pools[opponent], train_stats[a.team]["win_rate"])
                result = self.evaluate_and_snapshot(a.team, self.generation)
                record["teams"][a.team] = {
                    "train_games": train_stats[a.team]["games"],
                    "train_win_rate": train_stats[a.team]["win_rate"],
                    "train_win_rate_by_opponent": train_stats[a.team]["opponents"],
                    **result,
                    "opponent_distribution": [round(float(p), 3) for p in sampler.probabilities(self.pools[opponent])],
                    "sampler_state": sampler.state(),
                }
            record["ratings"] = {
                team: {entry.generation: round(entry.rating, 1) for entry in pool} for team, pool in self.pools.items()
            }
            record["elapsed_s"] = round(time.perf_counter() - start, 2)
            self.history.append(record)
            if self.cfg.verbose:
                self._log(record)
            if self.cfg.checkpoint_dir:
                os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)
                with open(os.path.join(self.cfg.checkpoint_dir, "history.json"), "w") as file:
                    json.dump(self.history, file, indent=2, default=str)
            if callback is not None:
                callback(record)
        return self.history

    def _log(self, record: dict) -> None:
        lines = [f"[self-play] generation {record['generation']} ({record['elapsed_s']} s)"]
        for team, info in record["teams"].items():
            train = info["train_win_rate"]
            lines.append(
                f"  {team}: train win rate {'n/a' if train is None else f'{train:.3f}'} "
                f"({info['train_games']} games) | eval {info['eval_win_rate']} | rating {info['rating']}"
                + (f" | pbt {info['pbt']}" if "pbt" in info else "")
            )
            lines.append(f"    opponent distribution: {info['opponent_distribution']} {info['sampler_state']}")
        logger.info("\n".join(lines))
