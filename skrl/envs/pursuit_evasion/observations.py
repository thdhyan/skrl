"""Pluggable observation groups for the pursuit-evasion grid environment.

Every observation builder produces a flat ``float32`` vector per agent; all agents share the same observation size so
the result is a single ``(num_envs, num_agents, size)`` tensor that the environment splits into the per-agent dict.

New groups are added by subclassing :class:`ObservationBuilder` and decorating the class with
:func:`register_observation`::

    @register_observation("my_group")
    class MyObservation(ObservationBuilder):
        def size(self) -> int: ...
        def build(self) -> torch.Tensor: ...  # (num_envs, num_agents, size)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from skrl.envs.pursuit_evasion.env import PursuitEvasionEnv


OBSERVATION_BUILDERS: dict[str, type["ObservationBuilder"]] = {}


def register_observation(name: str) -> Callable[[type], type]:
    """Class decorator that registers an observation builder under ``name``."""

    def decorator(cls: type) -> type:
        if name in OBSERVATION_BUILDERS:
            raise KeyError(f"Observation group '{name}' is already registered")
        OBSERVATION_BUILDERS[name] = cls
        cls.name = name
        return cls

    return decorator


def make_observation_builder(name: str, env: "PursuitEvasionEnv") -> "ObservationBuilder":
    """Instantiate the observation builder registered under ``name``."""
    if name not in OBSERVATION_BUILDERS:
        raise KeyError(f"Unknown observation group '{name}'. Available: {sorted(OBSERVATION_BUILDERS)}")
    return OBSERVATION_BUILDERS[name](env)


class ObservationBuilder:
    """Base class of the observation groups."""

    name: str = ""
    requires_fields: bool = False
    """Whether the builder needs the potential fields (the environment only computes them when required)."""

    def __init__(self, env: "PursuitEvasionEnv") -> None:
        self.env = env

    def size(self) -> int:
        """Flat observation size (same for every agent)."""
        raise NotImplementedError

    def build(self) -> torch.Tensor:
        """Build the observations, shape ``(num_envs, num_agents, size)``."""
        raise NotImplementedError

    # helpers

    def normalized_positions(self) -> torch.Tensor:
        """Agent positions mapped to ``[-1, 1]``, shape ``(E, A, 2)``."""
        env = self.env
        scale = torch.tensor([max(env.height - 1, 1), max(env.width - 1, 1)], device=env.device)
        return env.positions.float() / scale * 2.0 - 1.0

    def role_one_hot(self) -> torch.Tensor:
        """Role one-hot ``[is_pursuer, is_evader]``, shape ``(A, 2)``."""
        env = self.env
        role = torch.zeros((env.num_agents, 2), device=env.device)
        role[: env.num_pursuers, 0] = 1.0
        role[env.num_pursuers :, 1] = 1.0
        return role

    def normalized_potential(self, field: torch.Tensor) -> torch.Tensor:
        """Map a potential to ``[0, 1]`` by clipping at ``cfg.potential_clip``."""
        return (field / self.env.cfg.potential_clip).clamp(0.0, 1.0)


@register_observation("full_state")
class FullStateObservation(ObservationBuilder):
    """Full (privileged) state from the agent's point of view.

    Layout (``A = n_pursuers + 1``)::

        [ absolute positions of all agents (2A, in [-1, 1], pursuers first, evader last),
          positions of all agents relative to self (2A, divided by max(H, W) - 1),
          role one-hot (2), agent id one-hot (A), elapsed time fraction (1),
          obstacle map flattened (H*W, only if the map has obstacles and cfg.full_state_obstacles) ]
    """

    def __init__(self, env: "PursuitEvasionEnv") -> None:
        super().__init__(env)
        self._include_obstacles = bool(env.cfg.full_state_obstacles and env.obstacles.any())

    def size(self) -> int:
        a = self.env.num_agents
        return 2 * a + 2 * a + 2 + a + 1 + (self.env.height * self.env.width if self._include_obstacles else 0)

    def build(self) -> torch.Tensor:
        env = self.env
        num_envs, a = env.num_envs, env.num_agents
        absolute = self.normalized_positions()  # (E, A, 2)
        pos = env.positions.float()
        relative = (pos[:, None, :, :] - pos[:, :, None, :]) / max(env.height - 1, env.width - 1, 1)  # (E, A, A, 2)
        parts = [
            absolute.reshape(num_envs, 1, 2 * a).expand(num_envs, a, 2 * a),
            relative.reshape(num_envs, a, 2 * a),
            self.role_one_hot().expand(num_envs, a, 2),
            torch.eye(a, device=env.device).expand(num_envs, a, a),
            (env.timestep.float() / env.cfg.max_steps).view(num_envs, 1, 1).expand(num_envs, a, 1),
        ]
        if self._include_obstacles:
            obstacles = env.obstacles.float().view(1, 1, -1)
            parts.append(obstacles.expand(num_envs, a, obstacles.shape[-1]))
        return torch.cat(parts, dim=-1)


@register_observation("global_potential")
class GlobalPotentialObservation(ObservationBuilder):
    """Entire ``H x W`` potential field map(s) of the agent, flattened channel-first (``C x H x W``).

    Channels (``C = 5``): ``[total potential, attractive component, repulsive component, obstacle mask,
    self position one-hot]``. Potentials are normalized to ``[0, 1]`` (see ``cfg.potential_clip``).
    Use ``env.observation_image_shape`` to reshape it into an image for a CNN.
    """

    requires_fields = True
    num_channels = 5

    def size(self) -> int:
        return self.num_channels * self.env.height * self.env.width

    def image_shape(self) -> tuple[int, int, int]:
        return (self.num_channels, self.env.height, self.env.width)

    def build(self) -> torch.Tensor:
        env = self.env
        fields = env.potential_fields()
        num_envs, a, h, w = fields["total"].shape
        self_position = torch.zeros((num_envs, a, h * w), device=env.device)
        flat_index = env.positions[..., 0] * w + env.positions[..., 1]
        self_position.scatter_(-1, flat_index.unsqueeze(-1), 1.0)
        channels = torch.stack(
            [
                self.normalized_potential(fields["total"]),
                self.normalized_potential(fields["attractive"]),
                self.normalized_potential(fields["repulsive"]),
                env.obstacles.float().expand(num_envs, a, h, w),
                self_position.view(num_envs, a, h, w),
            ],
            dim=2,
        )  # (E, A, C, H, W)
        return channels.reshape(num_envs, a, -1)


@register_observation("local_potential")
class LocalPotentialObservation(ObservationBuilder):
    """Egocentric ``k x k`` window (``k = 2 * cfg.local_radius + 1``) of the potential field around the agent.

    Layout::

        [ window channels flattened channel-first (4 * k * k): total potential, attractive component,
          repulsive component, obstacle/out-of-bounds mask,
          own normalized position (2), role one-hot (2) ]

    Cells outside the grid are padded with the maximum normalized potential (1) and marked as obstacles.
    """

    requires_fields = True
    num_channels = 4

    def __init__(self, env: "PursuitEvasionEnv") -> None:
        super().__init__(env)
        self.radius = int(env.cfg.local_radius)
        self.window = 2 * self.radius + 1
        offsets = torch.arange(-self.radius, self.radius + 1, device=env.device)
        self._offsets = offsets

    def size(self) -> int:
        return self.num_channels * self.window * self.window + 2 + 2

    def image_shape(self) -> tuple[int, int, int]:
        return (self.num_channels, self.window, self.window)

    def build(self) -> torch.Tensor:
        env = self.env
        fields = env.potential_fields()
        num_envs, a, h, w = fields["total"].shape
        r = self.radius
        maps = torch.stack(
            [
                self.normalized_potential(fields["total"]),
                self.normalized_potential(fields["attractive"]),
                self.normalized_potential(fields["repulsive"]),
                env.obstacles.float().expand(num_envs, a, h, w),
            ],
            dim=2,
        )  # (E, A, C, H, W)
        padded = F.pad(maps, (r, r, r, r), value=1.0)  # (E, A, C, H + 2r, W + 2r)
        # window indices in padded coordinates: position + offset + r
        rows = env.positions[..., 0, None] + self._offsets + r  # (E, A, k)
        cols = env.positions[..., 1, None] + self._offsets + r  # (E, A, k)
        e_idx = torch.arange(num_envs, device=env.device).view(num_envs, 1, 1, 1, 1)
        a_idx = torch.arange(a, device=env.device).view(1, a, 1, 1, 1)
        c_idx = torch.arange(self.num_channels, device=env.device).view(1, 1, -1, 1, 1)
        window = padded[e_idx, a_idx, c_idx, rows[:, :, None, :, None], cols[:, :, None, None, :]]  # (E, A, C, k, k)
        return torch.cat(
            [
                window.reshape(num_envs, a, -1),
                self.normalized_positions(),
                self.role_one_hot().expand(num_envs, a, 2),
            ],
            dim=-1,
        )
