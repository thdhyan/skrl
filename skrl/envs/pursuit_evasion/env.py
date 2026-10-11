"""Vectorized (batched) 2D grid pursuit-evasion environment: ``n`` pursuers vs. 1 evader.

The environment is implemented natively in torch and follows the Isaac Lab ``DirectMARLEnv``-like convention used by
skrl's multi-agent wrappers: ``reset()``/``step()`` return dictionaries (keyed by agent name) of tensors with a leading
``num_envs`` dimension, sub-environments are reset automatically when their episode ends, and a shared global state is
available through ``state()``/``state_space`` (for MAPPO's centralized critic).

Wrap it with :class:`skrl.envs.wrappers.torch.pursuit_evasion_envs.PursuitEvasionWrapper` (or
``wrap_env(env, wrapper="pursuit-evasion")``) to obtain a :class:`~skrl.envs.wrappers.torch.MultiAgentEnvWrapper`.
"""

from __future__ import annotations

from typing import Any, Sequence

import dataclasses
import gymnasium

import torch

from skrl.envs.pursuit_evasion.observations import ObservationBuilder, make_observation_builder
from skrl.envs.pursuit_evasion.potential_fields import (
    compute_agent_fields,
    grid_coordinates,
    grid_diagonal,
    obstacle_field,
)

# action id -> (d_row, d_col)
ACTIONS = ("stay", "up", "down", "left", "right")
MOVES = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))


@dataclasses.dataclass(kw_only=True)
class PursuitEvasionCfg:
    """Configuration of the pursuit-evasion grid environment."""

    # world
    num_envs: int = 16
    """Number of parallel (batched) sub-environments."""
    height: int = 10
    """Grid height (rows)."""
    width: int = 10
    """Grid width (columns)."""
    num_pursuers: int = 3
    """Number of pursuers."""
    obstacles: Sequence[tuple[int, int]] | None = None
    """Explicit list of static obstacle cells ``(row, col)``."""
    obstacle_density: float = 0.0
    """Fraction of cells randomly turned into static obstacles (sampled once with ``obstacle_seed``)."""
    obstacle_seed: int = 0
    """Seed used to sample random obstacles (the layout is shared by all sub-environments)."""
    max_steps: int = 100
    """Maximum episode length (truncation)."""
    capture_radius: int = 1
    """Capture when a pursuer is within this distance of the evader (0: same cell, 1: adjacent, ...)."""
    capture_metric: str = "manhattan"
    """Distance metric for the capture test: ``"manhattan"`` (4-adjacency) or ``"chebyshev"`` (8-adjacency)."""
    spawn_min_distance: int = 3
    """Minimum Manhattan distance between the evader and every pursuer at reset."""

    # rewards (zero-sum: evader reward = - pursuer team reward)
    capture_reward: float = 10.0
    """Pursuer team reward on capture."""
    time_penalty: float = 0.01
    """Per-step penalty for the pursuer team (i.e. survival bonus for the evader)."""
    shaping_scale: float = 1.0
    """Scale of the potential-based shaping term ``gamma * Phi(s') - Phi(s)``, ``Phi = -mean_i U_att(p_i)``."""
    shaping_gamma: float = 0.99
    """Discount used by the potential-based shaping term."""

    # potential fields (see skrl.envs.pursuit_evasion.potential_fields)
    k_att: float = 1.0
    """Gain of the pursuers' attractive (conic) potential towards the evader."""
    k_pp: float = 0.5
    """Gain of the inter-pursuer repulsive potential."""
    d0_pp: float = 2.5
    """Influence radius of the inter-pursuer repulsion (cells)."""
    k_ep: float = 1.0
    """Gain of the pursuer-to-evader repulsive potential (evader's field)."""
    d0_ep: float = 4.0
    """Influence radius of the pursuer-to-evader repulsion (cells)."""
    k_center: float = 0.2
    """Gain of the evader's weak attraction to the grid centre (anti corner-hiding)."""
    k_obs: float = 1.0
    """Gain of the obstacle repulsive potential."""
    d0_obs: float = 1.5
    """Influence radius of the obstacle repulsion (cells)."""
    d_min: float = 0.5
    """Distance clamp of the Khatib repulsive term (avoids the singularity on the source cell)."""
    potential_clip: float = 2.0
    """Potentials are divided by this value and clipped to ``[0, 1]`` in the observations."""

    # observations
    obs_mode: str = "full_state"
    """Observation group: ``"full_state"``, ``"global_potential"``, ``"local_potential"`` (or any registered)."""
    local_radius: int = 2
    """Radius of the egocentric window for ``"local_potential"`` (window size ``2 * r + 1``)."""
    full_state_obstacles: bool = True
    """Whether ``"full_state"`` (and the global state) include the flattened obstacle map (if there are obstacles)."""

    # misc
    device: str | torch.device | None = None
    """Device (default: ``"cuda:0"`` if available, otherwise ``"cpu"``)."""
    seed: int | None = None
    """Seed of the environment's random generator."""


class PursuitEvasionEnv:
    """Batched grid pursuit-evasion environment (see module docstring and :class:`PursuitEvasionCfg`)."""

    def __init__(self, cfg: PursuitEvasionCfg | dict | None = None, **kwargs: Any) -> None:
        if cfg is None:
            cfg = PursuitEvasionCfg(**kwargs)
        elif isinstance(cfg, dict):
            cfg = PursuitEvasionCfg(**{**cfg, **kwargs})
        elif kwargs:
            cfg = dataclasses.replace(cfg, **kwargs)
        self.cfg = cfg
        self._validate()

        if cfg.device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(cfg.device)

        self.num_envs = cfg.num_envs
        self.height, self.width = cfg.height, cfg.width
        self.num_pursuers = cfg.num_pursuers
        self.num_agents = self.max_num_agents = cfg.num_pursuers + 1

        self.pursuer_names = [f"pursuer_{i}" for i in range(cfg.num_pursuers)]
        self.evader_name = "evader"
        self.possible_agents = self.pursuer_names + [self.evader_name]
        self.agents = list(self.possible_agents)
        self.teams = {"pursuers": list(self.pursuer_names), "evader": [self.evader_name]}

        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(cfg.seed if cfg.seed is not None else torch.seed() % (2**63))

        # static map
        self.obstacles = self._build_obstacles()
        self._free_cells = (~self.obstacles).nonzero()  # (F, 2)
        self._coords = grid_coordinates(self.height, self.width, device=self.device)
        self._diagonal = grid_diagonal(self.height, self.width)
        self._obstacle_potential = obstacle_field(self.obstacles, cfg.k_obs, cfg.d0_obs, cfg.d_min)
        self._moves = torch.tensor(MOVES, device=self.device, dtype=torch.long)

        # state buffers
        self.positions = torch.zeros((self.num_envs, self.num_agents, 2), device=self.device, dtype=torch.long)
        self.timestep = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._fields_cache: dict[str, torch.Tensor] | None = None

        # episode statistics
        self._episode_return = torch.zeros(self.num_envs, device=self.device)
        self.extras: dict[str, Any] = {}

        # observations / spaces
        self.observation_builder: ObservationBuilder = make_observation_builder(cfg.obs_mode, self)
        obs_size = self.observation_builder.size()
        self.observation_spaces = {
            uid: gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(obs_size,)) for uid in self.possible_agents
        }
        self.action_spaces = {uid: gymnasium.spaces.Discrete(len(MOVES)) for uid in self.possible_agents}
        self._state_obstacles = bool(cfg.full_state_obstacles and self.obstacles.any())
        state_size = 2 * self.num_agents + 1 + (self.height * self.width if self._state_obstacles else 0)
        self.state_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(state_size,))

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        cfg = self.cfg
        if cfg.height < 2 or cfg.width < 2:
            raise ValueError("Grid must be at least 2x2")
        if cfg.num_pursuers < 1:
            raise ValueError("At least one pursuer is required")
        if cfg.capture_metric not in ("manhattan", "chebyshev"):
            raise ValueError(f"Unknown capture metric: {cfg.capture_metric}")
        if cfg.max_steps < 1:
            raise ValueError("max_steps must be positive")

    def _build_obstacles(self) -> torch.Tensor:
        cfg = self.cfg
        obstacles = torch.zeros((self.height, self.width), dtype=torch.bool, device=self.device)
        if cfg.obstacle_density > 0:
            generator = torch.Generator().manual_seed(cfg.obstacle_seed)
            obstacles |= (torch.rand((self.height, self.width), generator=generator) < cfg.obstacle_density).to(
                self.device
            )
        for r, c in cfg.obstacles or []:
            obstacles[r, c] = True
        if (~obstacles).sum() < self.num_agents + 1:
            raise ValueError("Not enough free cells for the agents")
        return obstacles

    @property
    def observation_image_shape(self) -> tuple[int, int, int] | None:
        """``(C, H, W)`` shape of the image part of the observation (potential modes only)."""
        fn = getattr(self.observation_builder, "image_shape", None)
        return fn() if fn else None

    # ------------------------------------------------------------------
    # potential fields
    # ------------------------------------------------------------------

    def potential_fields(self) -> dict[str, torch.Tensor]:
        """Potential fields of every agent for the current positions (cached until the next step/reset).

        :return: Dictionary of ``(num_envs, num_agents, H, W)`` tensors (see
            :func:`~skrl.envs.pursuit_evasion.potential_fields.compute_agent_fields`).
        """
        if self._fields_cache is None:
            cfg = self.cfg
            self._fields_cache = compute_agent_fields(
                pursuers=self.positions[:, : self.num_pursuers],
                evader=self.positions[:, -1],
                coords=self._coords,
                obstacle_potential=self._obstacle_potential,
                k_att=cfg.k_att,
                k_pp=cfg.k_pp,
                d0_pp=cfg.d0_pp,
                k_ep=cfg.k_ep,
                d0_ep=cfg.d0_ep,
                k_center=cfg.k_center,
                d_min=cfg.d_min,
            )
        return self._fields_cache

    def _shaping_potential(self, positions: torch.Tensor) -> torch.Tensor:
        """``Phi(s) = -mean_i U_att(p_i; evader)`` (evaluated only at the pursuers' cells), shape ``(E,)``."""
        diff = (positions[:, : self.num_pursuers] - positions[:, -1:]).float()
        return -self.cfg.k_att * torch.linalg.vector_norm(diff, dim=-1).mean(dim=-1) / self._diagonal

    # ------------------------------------------------------------------
    # dynamics
    # ------------------------------------------------------------------

    def _distance(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        diff = (a - b).abs()
        return diff.sum(-1) if self.cfg.capture_metric == "manhattan" else diff.max(-1).values

    def _spawn(self, env_ids: torch.Tensor) -> None:
        n = env_ids.numel()
        if not n:
            return
        num_free = self._free_cells.shape[0]
        index = torch.randint(num_free, (n, self.num_agents), device=self.device, generator=self._generator)
        positions = self._free_cells[index]  # (n, A, 2)
        min_distance = max(self.cfg.spawn_min_distance, self.cfg.capture_radius + 1)
        for _ in range(100):  # rejection sampling of the evader position
            too_close = ((positions[:, :-1] - positions[:, -1:]).abs().sum(-1) < min_distance).any(-1)
            if not too_close.any():
                break
            k = int(too_close.sum())
            resample = torch.randint(num_free, (k,), device=self.device, generator=self._generator)
            positions[too_close, -1] = self._free_cells[resample]
        self.positions[env_ids] = positions
        self.timestep[env_ids] = 0
        self._episode_return[env_ids] = 0

    def _observations(self) -> dict[str, torch.Tensor]:
        obs = self.observation_builder.build()
        return {uid: obs[:, i] for i, uid in enumerate(self.possible_agents)}

    def state(self) -> torch.Tensor:
        """Global state (shared by all agents): positions in ``[-1, 1]`` (2A), time fraction (1), [obstacles]."""
        scale = torch.tensor([max(self.height - 1, 1), max(self.width - 1, 1)], device=self.device)
        parts = [
            (self.positions.float() / scale * 2.0 - 1.0).view(self.num_envs, -1),
            (self.timestep.float() / self.cfg.max_steps).view(-1, 1),
        ]
        if self._state_obstacles:
            parts.append(self.obstacles.float().view(1, -1).expand(self.num_envs, -1))
        return torch.cat(parts, dim=-1)

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[dict[str, torch.Tensor], dict]:
        """Reset all sub-environments.

        :return: Observations (dict of ``(num_envs, obs_size)`` tensors) and info.
        """
        if seed is not None:
            self._generator.manual_seed(seed)
        self._spawn(torch.arange(self.num_envs, device=self.device))
        self._fields_cache = None
        self.extras = {}
        return self._observations(), self.extras

    def step(self, actions: dict[str, torch.Tensor]) -> tuple[
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, Any],
    ]:
        """Step all sub-environments (automatically resetting those whose episode ended).

        :param actions: Dict of discrete actions (``(num_envs,)`` or ``(num_envs, 1)``, any numeric dtype).

        :return: Observations, rewards ``(E, 1)``, terminated ``(E, 1)``, truncated ``(E, 1)`` and info. The
            observations of ended sub-environments are the first observations of the new episode. ``info`` contains
            ``"captured"`` and ``"done"`` (``(E,)`` bool), ``"episode_length"`` (``(E,)``, valid where done) and
            ``"episode"`` (scalar statistics of the episodes ended at this step, if any).
        """
        cfg = self.cfg
        action = torch.stack([actions[uid].view(-1) for uid in self.possible_agents], dim=-1)
        action = action.round().long().clamp(0, len(MOVES) - 1).to(self.device)  # (E, A)

        old = self.positions
        new = old + self._moves[action]
        inside = (new[..., 0] >= 0) & (new[..., 0] < self.height) & (new[..., 1] >= 0) & (new[..., 1] < self.width)
        rows, cols = new[..., 0].clamp(0, self.height - 1), new[..., 1].clamp(0, self.width - 1)
        valid = inside & ~self.obstacles[rows, cols]
        new = torch.where(valid.unsqueeze(-1), new, old)

        # capture: proximity after the move, or pursuer/evader swapping cells (passing through each other)
        pursuers, evader = new[:, :-1], new[:, -1:]
        close = self._distance(pursuers, evader) <= cfg.capture_radius
        swap = (pursuers == old[:, -1:]).all(-1) & (old[:, :-1] == evader).all(-1)
        captured = (close | swap).any(-1)  # (E,)

        self.timestep += 1
        truncated = (self.timestep >= cfg.max_steps) & ~captured

        # zero-sum rewards: pursuers' team reward and its negative for the evader
        phi_old = self._shaping_potential(old)
        phi_new = torch.where(captured, torch.zeros_like(phi_old), self._shaping_potential(new))
        shaping = cfg.shaping_scale * (cfg.shaping_gamma * phi_new - phi_old)
        team_reward = cfg.capture_reward * captured.float() - cfg.time_penalty + shaping
        self._episode_return += team_reward

        self.positions = new
        done = captured | truncated
        episode_length = self.timestep.clone()
        info: dict[str, Any] = {"captured": captured, "done": done, "episode_length": episode_length}
        if done.any():
            info["episode"] = {
                "capture_rate": captured[done].float().mean(),
                "episode_length": episode_length[done].float().mean(),
                "pursuers_return": self._episode_return[done].mean(),
            }
        # automatic reset
        self._spawn(done.nonzero().view(-1))
        self._fields_cache = None
        self.extras = info

        rewards = {uid: team_reward.view(-1, 1).clone() for uid in self.pursuer_names}
        rewards[self.evader_name] = -team_reward.view(-1, 1)
        terminated = {uid: captured.view(-1, 1).clone() for uid in self.possible_agents}
        truncated = {uid: truncated.view(-1, 1).clone() for uid in self.possible_agents}
        return self._observations(), rewards, terminated, truncated, info

    # ------------------------------------------------------------------
    # misc
    # ------------------------------------------------------------------

    def render(self, env_id: int = 0, mode: str = "ansi") -> str:
        """Text rendering of one sub-environment (``P`` pursuer, ``E`` evader, ``#`` obstacle)."""
        grid = [["#" if self.obstacles[r, c] else "." for c in range(self.width)] for r in range(self.height)]
        for i, (r, c) in enumerate(self.positions[env_id].tolist()):
            grid[r][c] = "E" if i == self.num_pursuers else str(i) if i < 10 else "P"
        text = "\n".join("".join(row) for row in grid)
        if mode == "human":
            print(text)
        return text

    def close(self) -> None:
        pass
