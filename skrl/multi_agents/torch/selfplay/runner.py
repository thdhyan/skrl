"""Generation-based self-play runner for two-team (zero-sum) multi-agent environments.

Each team (e.g. ``"pursuers"`` and ``"evader"``) has

* a **population of learners** (``population_size`` members, 1 by default and 4 by default for the
  ``"adaptive_pbt"`` sampler): independent skrl :class:`~skrl.multi_agents.torch.mappo.MAPPO` or
  :class:`~skrl.multi_agents.torch.ippo.IPPO` instances controlling the team's agents, each with its own weights,
  optimizer, hyper-parameters (learning rate and entropy coefficient) and opponent sampler. The members train
  concurrently: the team's sub-environments are split evenly among them, and
* an **opponent pool**: frozen snapshots of the team's policies, added every generation the team learns
  (generation 0 being the initial, untrained policies), each with an Elo rating.

Every generation:

1. **Train** every member of the learning team(s) for ``timesteps_per_generation`` environment steps on its own
   sub-environments. Each sub-environment plays against an opponent snapshot drawn from the other team's pool by the
   member's sampler; a new opponent is drawn for a sub-environment whenever its episode ends (per-episode opponent
   sampling). Game outcomes feed the pool statistics and the member's (adaptive) sampler.
2. **Evaluate** every member against the same (up to ``eval_max_opponents``) snapshots of the opposing pool -- the
   most recent ones plus the best-rated one -- and update the Elo ratings.
3. **Snapshot** the members into their team's pool (all of them, or only the best one, see
   ``population_snapshots``), in increasing order of evaluation score so the generation's strongest member is the
   pool's latest entry.
4. Optionally perform **population-based training** (``pbt``):

   * population (``population_size > 1``): truncation selection. Members are ranked by their mean evaluation score
     (comparable since all members play the same opponents). Each member of the bottom ``pbt_truncation`` fraction
     that scores more than ``pbt_margin`` below a randomly chosen member of the top fraction copies that member's
     weights, optimizer state, hyper-parameters, rating and sampler statistics (exploit), then randomly perturbs its
     learning rate and entropy coefficient by one of ``pbt_perturb_factors`` (explore). The members' initial
     hyper-parameters are spread log-uniformly over ``pbt_init_range`` around the configured values.
   * single learner: the best-rated snapshot of the learner's own pool is evaluated against the same opponents; if
     the new snapshot scores clearly worse, the learner restarts from it (exploit) and perturbs its hyper-parameters
     (explore).

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
    """Keyword arguments of the sampler's constructor (with ``sampler_schedule``: a mapping from sampler name to
    keyword arguments)."""
    sampler_schedule: list[str] | None = None
    """Opponent sampler per generation, cycled: generation ``g`` uses ``sampler_schedule[(g - 1) % len]`` (e.g.
    ``["last", "uniform", "hedge"]``). Overrides ``sampler`` for the opponent choice. The members get a fresh sampler
    whenever the scheduled name changes (adaptive statistics are not carried over)."""
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
    population_size: int | None = None
    """Concurrent learners per learning team (default: 4 for the ``"adaptive_pbt"`` sampler -- capped by the number
    of sub-environments the team trains on -- and 1 otherwise). The team's sub-environments are split among them."""
    population_snapshots: str = "all"
    """Members added to the team's pool every generation: ``"all"`` or ``"best"`` (highest evaluation score)."""
    pbt: bool | None = None
    """Population-based training (default: enabled only for the ``"adaptive_pbt"`` sampler)."""
    pbt_truncation: float = 0.25
    """Fraction of the population replaced (bottom) and copied from (top) by truncation selection (at least one
    member each when the population has at least 2 members)."""
    pbt_margin: float = 0.1
    """Exploit only when the member's mean evaluation score is more than this below the score of the member it
    would copy (single learner: below the best-rated snapshot of its own pool, evaluated on the same opponents)."""
    pbt_perturb_factors: tuple[float, float] = (0.8, 1.25)
    """Multiplicative factors randomly applied to the learning rate and entropy coefficient on explore."""
    pbt_init_range: tuple[float, float] = (0.5, 2.0)
    """Range of the log-uniform factors applied to the initial learning rate and entropy coefficient of the
    population members (except member 0, which keeps the configured values)."""
    seed: int | None = None
    """Seed of the runner's random generators (opponent sampling and PBT perturbations)."""
    checkpoint_dir: str | None = None
    """If set, snapshots (``{team}_gen{g}.pt``) and ``history.json`` are written there."""
    verbose: bool = True
    """Log a summary of every generation."""
    wandb: bool = False
    """Log every generation's metrics to Weights & Biases (one step per generation)."""
    wandb_kwargs: dict = dataclasses.field(default_factory=dict)
    """Keyword arguments of ``wandb.init`` (e.g. ``project``, ``group``, ``name``, ``mode``)."""
    video_dir: str | None = None
    """If set, a debug video (potential fields, trails, capture radius; see
    :class:`~skrl.envs.pursuit_evasion.render.RolloutRenderer`) of one episode of each learning team's best member
    against the latest opponent snapshot is written there every ``video_interval`` generations (and logged to W&B)."""
    video_interval: int = 1
    """Generations between debug videos."""
    video_max_steps: int | None = None
    """Maximum steps per debug video (default: the environment's ``max_steps``)."""


@dataclasses.dataclass
class _Member:
    """One learner of a team's population and the slice of sub-environments it trains on."""

    team: str
    index: int
    learner: Any
    sampler: OpponentSampler
    start: int
    stop: int
    rating: float = 1000.0
    timesteps: int = 0

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
        if cfg.population_snapshots not in ("all", "best"):
            raise ValueError(f"Unknown population_snapshots: {cfg.population_snapshots}")
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
        self.population_size = self._resolve_population_size(min(stop - start for start, stop in spans.values()))

        # populations of learners (each with its own sampler choosing opponents from the other team's pool) and pools
        self.members: dict[str, list[_Member]] = {}
        for team in self.learning_teams:
            start, stop = spans[team]
            bounds = [start + (stop - start) * i // self.population_size for i in range(self.population_size + 1)]
            self.members[team] = [
                _Member(
                    team=team,
                    index=i,
                    learner=self._build_learner(team, bounds[i], bounds[i + 1], i),
                    sampler=self._make_sampler(self._sampler_name(1)),
                    start=bounds[i],
                    stop=bounds[i + 1],
                    rating=cfg.initial_rating,
                )
                for i in range(self.population_size)
            ]
            if self.pbt and self.population_size > 1:
                low, high = (math.log(f) for f in cfg.pbt_init_range)
                for member in self.members[team][1:]:
                    self._scale_hyperparameters(
                        member, math.exp(self._rng.uniform(low, high)), math.exp(self._rng.uniform(low, high))
                    )
        self.best_member = {team: 0 for team in self.learning_teams}
        self.pools = {team: OpponentPool(team, uids, max_size=cfg.pool_max_size) for team, uids in self.teams.items()}
        self._total_timesteps = cfg.generations * cfg.timesteps_per_generation
        self.history: list[dict[str, Any]] = []
        self.generation = 0

        # generation 0: initial policies (of the first member: all initial policies are untrained)
        for team in self.teams:
            if team in self.members:
                policies, values = self._learner_state(self.members[team][0])
            else:
                policies = {uid: clone_state_dict(self._make_policy(uid)) for uid in self.teams[team]}
                values = {}
            self.pools[team].add(
                PolicySnapshot(team=team, generation=0, policies=policies, values=values, rating=cfg.initial_rating)
            )

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------

    def _sampler_name(self, generation: int) -> str:
        schedule = self.cfg.sampler_schedule
        return schedule[(generation - 1) % len(schedule)] if schedule else self.cfg.sampler

    def _make_sampler(self, name: str) -> OpponentSampler:
        kwargs = self.cfg.sampler_kwargs.get(name, {}) if self.cfg.sampler_schedule else self.cfg.sampler_kwargs
        return make_sampler(name, **kwargs)

    @property
    def learners(self) -> dict[str, Any]:
        """Learner of each learning team (the member with the best evaluation score in the last generation)."""
        return {team: members[self.best_member[team]].learner for team, members in self.members.items()}

    def _resolve_population_size(self, max_size: int) -> int:
        size = self.cfg.population_size
        if size is None:
            size = 4 if self.cfg.sampler == "adaptive_pbt" else 1
            if size > max_size:
                logger.warning(f"Population size reduced from {size} to {max_size} (one sub-environment per member)")
                size = max_size
        if size < 1:
            raise ValueError(f"The population size must be positive, got {size}")
        if size > max_size:
            raise ValueError(
                f"Population size {size} exceeds the {max_size} sub-environments available per learning team"
            )
        return size

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
        ).to(self.device)

    def _make_value(self, uid: str) -> MLPValue:
        return MLPValue(
            observation_space=self.env.observation_space(uid),
            state_space=self.env.state_space(uid),
            action_space=self.env.action_space(uid),
            device=self.device,
            hidden_sizes=self.cfg.hidden_sizes,
            use_states=self.cfg.algorithm == "mappo",
        )

    def _build_learner(self, team: str, start: int, stop: int, index: int = 0):
        uids = self.teams[team]
        models = {uid: {"policy": self._make_policy(uid), "value": self._make_value(uid)} for uid in uids}
        memories = {
            uid: RandomMemory(memory_size=self.agent_cfg["rollouts"], num_envs=stop - start, device=self.device)
            for uid in uids
        }
        agent_cfg = copy.deepcopy(self.agent_cfg)
        if self.cfg.checkpoint_dir and not agent_cfg["experiment"].get("directory"):
            agent_cfg["experiment"]["directory"] = os.path.join(self.cfg.checkpoint_dir, "agents")
        agent_cfg["experiment"].setdefault(
            "experiment_name", team if self.population_size == 1 else f"{team}_member{index}"
        )
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

    def _learner_state(self, member: _Member) -> tuple[dict, dict]:
        learner = member.learner
        policies = {uid: clone_state_dict(learner.policies[uid]) for uid in self.teams[member.team]}
        values = {uid: clone_state_dict(learner.values[uid]) for uid in self.teams[member.team]}
        return policies, values

    def _reset_learner_rollout(self, member: _Member) -> None:
        learner = member.learner
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

    def _learning_teams_of(self, generation: int) -> list[str]:
        if self.simultaneous:
            return list(self.learning_teams)
        if len(self.learning_teams) == 1:
            return [self.learning_teams[0]]
        order = [self.cfg.first_learner, self.opponent_of[self.cfg.first_learner]]
        return [order[(generation - 1) % 2]]

    def train_generation(self, teams: list[str]) -> dict[tuple[str, int], dict[str, Any]]:
        """Train every population member of the given teams for ``timesteps_per_generation`` steps.

        :return: Training statistics keyed by ``(team, member index)``.
        """
        cfg = self.cfg
        first_uid = self.env.possible_agents[0]
        assignments = [member for team in teams for member in self.members[team]]
        stats = {(a.team, a.index): {"games": 0, "wins": 0.0, "opponents": {}} for a in assignments}

        opponent_indices = {}
        for a in assignments:
            self._reset_learner_rollout(a)
            a.learner.enable_training_mode(True)
            opponent = self.opponent_of[a.team]
            opponent_indices[a.team, a.index] = a.sampler.sample(self.pools[opponent], a.size, self._generator)

        observations, infos = self.env.reset()
        states = self.env.state()
        for _ in range(cfg.timesteps_per_generation):
            actions = self._empty_actions()
            with torch.no_grad():
                for a in assignments:
                    learner, uids, sl = a.learner, self.teams[a.team], a.slice
                    learner.pre_interaction(timestep=a.timesteps, timesteps=self._total_timesteps)
                    learner_actions, _ = learner.act(
                        {uid: observations[uid][sl] for uid in uids},
                        {uid: states[uid][sl] for uid in uids},
                        timestep=a.timesteps,
                        timesteps=self._total_timesteps,
                    )
                    for uid in uids:
                        actions[uid][sl] = learner_actions[uid].view(-1, 1).long()
                    self._opponent_actions(
                        self.opponent_of[a.team], opponent_indices[a.team, a.index], observations, states, sl, actions
                    )

                next_observations, rewards, terminated, truncated, infos = self.env.step(actions)
                next_states = self.env.state()

                for a in assignments:
                    learner, uids, sl = a.learner, self.teams[a.team], a.slice
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
                        timestep=a.timesteps,
                        timesteps=self._total_timesteps,
                    )

            for a in assignments:
                a.learner.post_interaction(timestep=a.timesteps, timesteps=self._total_timesteps)
                a.timesteps += 1

                # bookkeeping of finished games and per-episode opponent resampling
                term = terminated[first_uid][a.slice].view(-1)
                done = (term | truncated[first_uid][a.slice].view(-1)).cpu()
                if done.any():
                    opponent = self.opponent_of[a.team]
                    pool, sampler, key = self.pools[opponent], a.sampler, (a.team, a.index)
                    indices = opponent_indices[key][done]
                    scores = self._learner_scores(a.team, term.cpu(), done)
                    pool.record(indices, scores)
                    sampler.update(pool, indices, scores)
                    opponent_indices[key][done] = sampler.sample(pool, int(done.sum()), self._generator)
                    stats[key]["games"] += int(done.sum())
                    stats[key]["wins"] += float(scores.sum())
                    for index, score in zip(indices.tolist(), scores.tolist()):
                        entry = stats[key]["opponents"].setdefault(self._entry_label(pool[index]), [0, 0.0])
                        entry[0] += 1
                        entry[1] += score

            observations, states = next_observations, next_states

        for a in assignments:
            a.learner.enable_training_mode(False)
            entry = stats[a.team, a.index]
            entry["win_rate"] = entry["wins"] / entry["games"] if entry["games"] else None
            entry["opponents"] = dict(sorted(entry["opponents"].items(), key=lambda item: str(item[0])))
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

    @torch.no_grad()
    def record_video(self, team: str, path: str, opponent_index: int | None = None, fps: int = 10) -> str:
        """Record one episode (sub-environment 0) of the team's best member against an opponent snapshot.

        :param opponent_index: Opponent pool index (default: the latest snapshot).

        :return: Path of the written video.
        """
        from skrl.envs.pursuit_evasion.render import RolloutRenderer, write_video

        opponent = self.opponent_of[team]
        pool = self.pools[opponent]
        index = len(pool) - 1 if opponent_index is None else opponent_index
        member = self.members[team][self.best_member[team]]
        policies = {uid: member.learner.policies[uid] for uid in self.teams[team]}
        for model in policies.values():
            model.eval()
        indices = torch.full((self.num_envs,), index, dtype=torch.long)
        first_uid = self.env.possible_agents[0]
        max_steps = self.cfg.video_max_steps or getattr(getattr(self.env, "cfg", None), "max_steps", 1000)
        label = f"gen {self.generation} | sampler {self._sampler_name(self.generation)} | {team} best member vs "
        label += f"{opponent} gen {self._entry_label(pool[index])}"

        renderer = RolloutRenderer(self.env)
        observations, _ = self.env.reset()
        states = self.env.state()
        frames = [renderer.frame(label)]
        full = slice(0, self.num_envs)
        for step in range(max_steps):
            actions = self._empty_actions()
            for uid, action in self._policy_actions(
                policies, observations, states, full, self.cfg.eval_deterministic
            ).items():
                actions[uid][:] = action.view(-1, 1).long()
            self._opponent_actions(opponent, indices, observations, states, full, actions)
            observations, _, terminated, truncated, infos = self.env.step(actions)
            states = self.env.state()
            if bool(terminated[first_uid][0] | truncated[first_uid][0]):
                outcome = "CAPTURED" if bool(terminated[first_uid][0]) else "EVADED"
                frame = renderer.frame(f"{label} | {outcome}", infos["final_positions"][0], step + 1)
                frames.extend([frame] * fps)  # hold the final frame for one second
                break
            frames.append(renderer.frame(label))
        renderer.close()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        return write_video(path, frames, fps=fps)

    def _evaluation_opponents(self, pool: OpponentPool) -> list[int]:
        k = max(1, self.cfg.eval_max_opponents)
        indices = set(range(max(0, len(pool) - (k - 1)), len(pool))) if k > 1 else set()
        indices.add(pool.best_index())
        return sorted(indices)

    def evaluate(self, member: _Member) -> dict[int, float]:
        """Evaluate a population member against the evaluation opponents and update the Elo ratings.

        :return: Mean score of the member per opponent (keyed by :meth:`_entry_label`).
        """
        cfg = self.cfg
        opponent_pool = self.pools[self.opponent_of[member.team]]
        policies = {uid: member.learner.policies[uid] for uid in self.teams[member.team]}
        for model in policies.values():
            model.eval()
        evaluation = {}
        for index in self._evaluation_opponents(opponent_pool):
            score = self.play(member.team, policies, index, cfg.eval_episodes, cfg.eval_deterministic)
            opponent_entry = opponent_pool[index]
            member.rating, opponent_entry.rating = elo_update(member.rating, opponent_entry.rating, score, cfg.elo_k)
            evaluation[self._entry_label(opponent_entry)] = round(score, 3)
        return evaluation

    def evaluate_and_snapshot(self, team: str, generation: int) -> dict[str, Any]:
        """Evaluate the population of ``team``, add its snapshot(s) to the pool and apply PBT (if enabled)."""
        cfg = self.cfg
        members = self.members[team]
        own_pool = self.pools[team]
        evaluations = {member.index: self.evaluate(member) for member in members}
        scores = {i: sum(e.values()) / len(e) for i, e in evaluations.items()}
        ranking = sorted(scores, key=lambda i: (scores[i], members[i].rating))  # ascending
        best = ranking[-1]
        self.best_member[team] = best

        results: dict[int, dict[str, Any]] = {}
        snapshots: dict[int, PolicySnapshot] = {}
        for member in members:
            policy_states, value_states = self._learner_state(member)
            snapshots[member.index] = PolicySnapshot(
                team=team,
                generation=generation,
                policies=policy_states,
                values=value_states,
                rating=member.rating,
                metadata={
                    "eval_win_rate": evaluations[member.index],
                    "hyperparameters": self._hyperparameters(member),
                },
            )
            if len(members) > 1:
                snapshots[member.index].metadata["member"] = member.index
            results[member.index] = {
                "eval_win_rate": evaluations[member.index],
                "eval_score": round(scores[member.index], 3),
                "rating": round(member.rating, 1),
                "hyperparameters": snapshots[member.index].metadata["hyperparameters"],
            }

        if self.pbt and len(members) > 1:
            for event in self._pbt_population(team, scores, ranking):
                results[event["member"]]["pbt"] = event
                snapshots[event["member"]].metadata["pbt"] = event
        elif self.pbt and len(own_pool) > 1:
            event = self._pbt_single(members[0], scores[0])
            if event is not None:
                results[0]["pbt"] = event
                snapshots[0].metadata["pbt"] = event

        # add to the pool in increasing order of score: the generation's strongest member is the latest entry
        for i in ranking if cfg.population_snapshots == "all" else [best]:
            own_pool.add(snapshots[i])
            if cfg.checkpoint_dir:
                os.makedirs(cfg.checkpoint_dir, exist_ok=True)
                suffix = "" if len(members) == 1 else f"_member{i}"
                path = os.path.join(cfg.checkpoint_dir, f"{team}_gen{generation:03d}{suffix}.pt")
                torch.save(dataclasses.asdict(snapshots[i]), path)
        return {"best_member": best, "members": results}

    def _pbt_population(self, team: str, scores: dict[int, float], ranking: list[int]) -> list[dict[str, Any]]:
        """Truncation selection: the bottom members exploit (copy) a top member, then explore (perturb)."""
        cfg = self.cfg
        members = self.members[team]
        count = min(max(1, int(len(members) * cfg.pbt_truncation)), len(members) // 2)
        top, bottom = ranking[-count:], ranking[:count]
        events = []
        for i in bottom:
            source = self._rng.choice(top)
            event: dict[str, Any] = {
                "member": i,
                "score": round(scores[i], 3),
                "source_member": source,
                "source_score": round(scores[source], 3),
            }
            if scores[i] < scores[source] - cfg.pbt_margin:
                self._copy_member(members[source], members[i])
                event["exploit_from_member"] = source
                event["explore"] = self._perturb_hyperparameters(members[i])
            events.append(event)
        return events

    def _pbt_single(self, member: _Member, score: float) -> dict[str, Any] | None:
        """Single learner: compare head-to-head with the best-rated snapshot of its own pool on the same opponents
        (Elo alone is not comparable across snapshots rated against different opponents)."""
        cfg = self.cfg
        own_pool, opponent_pool = self.pools[member.team], self.pools[self.opponent_of[member.team]]
        best = own_pool[own_pool.best_index()]
        if not best.values:
            return None
        best_policies = own_pool.policies(own_pool.best_index(), self._make_policy)
        best_scores = [
            self.play(member.team, best_policies, index, cfg.eval_episodes, cfg.eval_deterministic)
            for index in self._evaluation_opponents(opponent_pool)
        ]
        best_score = sum(best_scores) / len(best_scores)
        event = {"score": round(score, 3), "best_generation": best.generation, "best_score": round(best_score, 3)}
        if score < best_score - cfg.pbt_margin:
            self._load_learner(member, best)
            event["exploit_from_generation"] = best.generation
            event["explore"] = self._perturb_hyperparameters(member)
        return event

    def _hyperparameters(self, member: _Member) -> dict[str, float]:
        learner = member.learner
        uid = self.teams[member.team][0]
        return {
            "learning_rate": learner.optimizers[uid].param_groups[0]["lr"],
            "entropy_loss_scale": learner.cfg.entropy_loss_scale[uid],
        }

    def _load_learner(self, member: _Member, snapshot: PolicySnapshot) -> None:
        for uid in self.teams[member.team]:
            member.learner.policies[uid].load_state_dict(snapshot.policies[uid])
            member.learner.values[uid].load_state_dict(snapshot.values[uid])

    def _copy_member(self, source: _Member, target: _Member) -> None:
        """Exploit: copy weights, optimizer state (including the learning rate), preprocessors, entropy coefficient,
        rating and sampler statistics of ``source`` into ``target``."""
        src, dst = source.learner, target.learner
        for uid in self.teams[target.team]:
            dst.policies[uid].load_state_dict(src.policies[uid].state_dict())
            dst.values[uid].load_state_dict(src.values[uid].state_dict())
            dst.optimizers[uid].load_state_dict(copy.deepcopy(src.optimizers[uid].state_dict()))
            dst.cfg.entropy_loss_scale[uid] = src.cfg.entropy_loss_scale[uid]
            for name in ("_observation_preprocessor", "_state_preprocessor", "_value_preprocessor"):
                preprocessor = getattr(src, name, {}).get(uid)
                if isinstance(preprocessor, torch.nn.Module):
                    getattr(dst, name)[uid].load_state_dict(preprocessor.state_dict())
        target.rating = source.rating
        target.sampler = copy.deepcopy(source.sampler)

    def _scale_hyperparameters(self, member: _Member, lr_factor: float, entropy_factor: float) -> None:
        for uid in self.teams[member.team]:
            for group in member.learner.optimizers[uid].param_groups:
                group["lr"] *= lr_factor
            member.learner.cfg.entropy_loss_scale[uid] *= entropy_factor

    def _perturb_hyperparameters(self, member: _Member) -> dict[str, float]:
        lr_factor = self._rng.choice(self.cfg.pbt_perturb_factors)
        entropy_factor = self._rng.choice(self.cfg.pbt_perturb_factors)
        self._scale_hyperparameters(member, lr_factor, entropy_factor)
        return {"learning_rate_factor": lr_factor, "entropy_factor": entropy_factor, **self._hyperparameters(member)}

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
            teams = self._learning_teams_of(self.generation)
            sampler_name = self._sampler_name(self.generation)
            for team in teams:
                for member in self.members[team]:
                    if member.sampler.name != sampler_name:
                        member.sampler = self._make_sampler(sampler_name)
            train_stats = self.train_generation(teams)
            record: dict[str, Any] = {"generation": self.generation, "sampler": sampler_name, "teams": {}}
            for team in teams:
                opponent_pool = self.pools[self.opponent_of[team]]
                for member in self.members[team]:
                    member.sampler.on_generation_end(opponent_pool, train_stats[team, member.index]["win_rate"])
                # sampler distributions are logged before PBT (which may replace a member's sampler)
                distributions = {
                    member.index: (
                        [round(float(p), 3) for p in member.sampler.probabilities(opponent_pool)],
                        member.sampler.state(),
                    )
                    for member in self.members[team]
                }
                result = self.evaluate_and_snapshot(team, self.generation)
                best = result["best_member"]
                members = []
                for i, info in result["members"].items():
                    stats = train_stats[team, i]
                    members.append(
                        {
                            "member": i,
                            "train_games": stats["games"],
                            "train_win_rate": stats["win_rate"],
                            "opponent_distribution": distributions[i][0],
                            **info,
                        }
                    )
                # team summary: training statistics over the whole population, evaluation of its best member
                games = sum(train_stats[team, i]["games"] for i in result["members"])
                wins = sum(train_stats[team, i]["wins"] for i in result["members"])
                by_opponent: dict[int | str, list[float]] = {}
                for i in result["members"]:
                    for label, (n, w) in train_stats[team, i]["opponents"].items():
                        entry = by_opponent.setdefault(label, [0, 0.0])
                        entry[0] += n
                        entry[1] += w
                summary = {
                    "train_games": games,
                    "train_win_rate": wins / games if games else None,
                    "train_win_rate_by_opponent": {
                        label: round(w / n, 3) for label, (n, w) in sorted(by_opponent.items(), key=lambda x: str(x[0]))
                    },
                    "eval_win_rate": result["members"][best]["eval_win_rate"],
                    "rating": result["members"][best]["rating"],
                    "opponent_distribution": distributions[best][0],
                    "sampler_state": distributions[best][1],
                }
                if len(members) > 1:
                    summary["best_member"] = best
                    summary["members"] = members
                elif "pbt" in result["members"][0]:
                    summary["pbt"] = result["members"][0]["pbt"]
                record["teams"][team] = summary
            record["ratings"] = {
                team: {self._entry_label(entry): round(entry.rating, 1) for entry in pool}
                for team, pool in self.pools.items()
            }
            record["elapsed_s"] = round(time.perf_counter() - start, 2)
            if self.cfg.video_dir and self.generation % max(1, self.cfg.video_interval) == 0:
                record["videos"] = {
                    team: self.record_video(
                        team, os.path.join(self.cfg.video_dir, f"gen{self.generation:03d}_{team}.mp4")
                    )
                    for team in teams
                }
            self.history.append(record)
            if self.cfg.verbose:
                self._log(record)
            if self.cfg.checkpoint_dir:
                os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)
                with open(os.path.join(self.cfg.checkpoint_dir, "history.json"), "w") as file:
                    json.dump(self.history, file, indent=2, default=str)
            if self.cfg.wandb:
                self._wandb_log(record)
            if callback is not None:
                callback(record)
        return self.history

    def _wandb_log(self, record: dict) -> None:
        import wandb

        if wandb.run is None:
            kwargs = copy.deepcopy(self.cfg.wandb_kwargs)
            config = {**dataclasses.asdict(self.cfg), "population_size": self.population_size}
            config["env"] = {k: v for k, v in vars(getattr(self.env, "cfg", object())).items() if k != "obstacles"}
            config.update(kwargs.pop("config", {}))
            wandb.init(config=json.loads(json.dumps(config, default=str)), **kwargs)
        metrics: dict[str, float] = {"generation": record["generation"], "elapsed_s": record["elapsed_s"]}
        for team, info in record["teams"].items():
            if info["train_win_rate"] is not None:
                metrics[f"{team}/train_win_rate"] = info["train_win_rate"]
            metrics[f"{team}/train_games"] = info["train_games"]
            metrics[f"{team}/eval_score"] = sum(info["eval_win_rate"].values()) / len(info["eval_win_rate"])
            metrics[f"{team}/rating"] = info["rating"]
            probabilities = torch.tensor(info["opponent_distribution"], dtype=torch.float64).clamp(min=1e-12)
            metrics[f"{team}/opponent_entropy"] = float(-(probabilities * probabilities.log()).sum())
            metrics[f"{team}/pool_size"] = len(self.pools[team])
            for member in info.get("members", []):
                prefix = f"{team}/member{member['member']}"
                metrics[f"{prefix}/eval_score"] = member["eval_score"]
                metrics[f"{prefix}/rating"] = member["rating"]
                metrics[f"{prefix}/learning_rate"] = member["hyperparameters"]["learning_rate"]
                metrics[f"{prefix}/entropy_loss_scale"] = member["hyperparameters"]["entropy_loss_scale"]
                metrics[f"{prefix}/exploited"] = float("exploit_from_member" in member.get("pbt", {}))
        for team, path in record.get("videos", {}).items():
            metrics[f"{team}/video"] = wandb.Video(path, format="mp4")
        wandb.log(metrics, step=record["generation"])

    def close(self) -> None:
        """Finish the Weights & Biases run (if any)."""
        if self.cfg.wandb:
            import wandb

            if wandb.run is not None:
                wandb.finish()

    @staticmethod
    def _entry_label(entry: PolicySnapshot) -> str | int:
        """Generation of a snapshot (``"generation.member"`` for snapshots of a population member)."""
        member = entry.metadata.get("member")
        return entry.generation if member is None or entry.generation == 0 else f"{entry.generation}.{member}"

    def _log(self, record: dict) -> None:
        lines = [f"[self-play] generation {record['generation']} ({record['elapsed_s']} s, sampler {record['sampler']})"]
        for team, info in record["teams"].items():
            train = info["train_win_rate"]
            lines.append(
                f"  {team}: train win rate {'n/a' if train is None else f'{train:.3f}'} "
                f"({info['train_games']} games) | eval {info['eval_win_rate']} | rating {info['rating']}"
                + (f" | pbt {info['pbt']}" if "pbt" in info else "")
            )
            for member in info.get("members", []):
                hp = member["hyperparameters"]
                marker = "*" if member["member"] == info["best_member"] else " "
                lines.append(
                    f"   {marker}member {member['member']}: score {member['eval_score']:.3f} | rating "
                    f"{member['rating']} | lr {hp['learning_rate']:.2e} | entropy {hp['entropy_loss_scale']:.3g}"
                    + (
                        f" | exploit from {member['pbt']['exploit_from_member']}"
                        if "exploit_from_member" in member.get("pbt", {})
                        else ""
                    )
                )
            lines.append(f"    opponent distribution: {info['opponent_distribution']} {info['sampler_state']}")
        logger.info("\n".join(lines))
