"""Artificial potential fields (APF) on a 2D grid, vectorized over environments with torch.

All fields are evaluated at the *cell centres* of an ``H x W`` grid. Distances are Euclidean in cell units.
Lower potential is better: an agent following the field moves to the neighbouring cell with the lowest value
(gradient descent on the potential surface).

Formulas
--------

Let ``x`` be a cell, ``d(x, y)`` the Euclidean distance between cells and ``D = sqrt((H-1)^2 + (W-1)^2)`` the grid
diagonal (largest possible distance).

* **Attractive (conic) potential** towards a goal ``g`` (used by pursuers with ``g`` = evader position)::

      U_att(x; g) = k_att * d(x, g) / D                      in [0, k_att], minimum (0) exactly at g

  The conic (linear in distance) form is used instead of the quadratic ``0.5 * k * d^2`` because it keeps a constant
  gradient magnitude far from the goal and stays bounded.

* **Repulsive (Khatib) potential** from a point source ``s`` with influence radius ``d0``, normalized to ``[0, 1]``::

      g(d) = ((1 / max(d, d_min) - 1 / d0) / (1 / d_min - 1 / d0))^2    if d < d0
           = 0                                                       otherwise

      U_rep(x; s) = k_rep * g(d(x, s))                       in [0, k_rep], maximum at the source

  ``d_min`` (default 0.5 cells) clamps the singularity of the classic ``0.5 * k * (1/d - 1/d0)^2`` term so the
  field is finite on the source cell.

* **Obstacle potential**: the same Khatib term evaluated on the distance to the nearest obstacle cell
  (pre-computed once since obstacles are static); obstacle cells themselves get the maximum ``k_obs``.

Agent fields
------------

* Pursuer ``i``::

      U_i(x) = U_att(x; evader) + sum_{j != i} k_pp * g_pp(d(x, p_j)) + U_obs(x)

  (attraction to the evader + inter-pursuer repulsion that spreads the team out + obstacle repulsion)

* Evader::

      U_e(x) = sum_j k_ep * g_ep(d(x, p_j)) + k_center * d(x, c) / D + U_obs(x)

  (repulsion from every pursuer + a weak attraction to the grid centre ``c`` that discourages corner-hiding,
  which is a losing strategy on a bounded grid + obstacle repulsion)
"""

from __future__ import annotations

import math

import torch


def grid_coordinates(height: int, width: int, device: str | torch.device | None = None) -> torch.Tensor:
    """Cell coordinates of the grid.

    :return: Float tensor of shape ``(H, W, 2)`` with ``(row, col)`` coordinates.
    """
    rows = torch.arange(height, device=device, dtype=torch.float32)
    cols = torch.arange(width, device=device, dtype=torch.float32)
    rr, cc = torch.meshgrid(rows, cols, indexing="ij")
    return torch.stack([rr, cc], dim=-1)


def grid_diagonal(height: int, width: int) -> float:
    """Largest Euclidean distance between two cells of the grid (at least 1)."""
    return max(1.0, math.hypot(height - 1, width - 1))


def distance_maps(points: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """Euclidean distance from every grid cell to each point.

    :param points: Tensor of shape ``(..., 2)`` with (row, col) positions (any dtype).
    :param coords: Grid coordinates of shape ``(H, W, 2)`` (see :func:`grid_coordinates`).

    :return: Tensor of shape ``(..., H, W)``.
    """
    diff = coords - points.to(coords.dtype)[..., None, None, :]
    return torch.linalg.vector_norm(diff, dim=-1)


def khatib_profile(distance: torch.Tensor, d0: float, d_min: float = 0.5) -> torch.Tensor:
    """Normalized Khatib repulsive profile ``g(d)`` in ``[0, 1]`` (see module docstring)."""
    if d0 <= d_min:
        raise ValueError(f"Repulsive influence radius d0 ({d0}) must be greater than d_min ({d_min})")
    d = distance.clamp(min=d_min)
    value = ((1.0 / d - 1.0 / d0) / (1.0 / d_min - 1.0 / d0)) ** 2
    return torch.where(distance < d0, value, torch.zeros_like(value))


def attractive_field(goals: torch.Tensor, coords: torch.Tensor, k_att: float, diagonal: float) -> torch.Tensor:
    """Conic attractive potential ``k_att * d(x, goal) / D``.

    :param goals: Tensor of shape ``(..., 2)``.

    :return: Tensor of shape ``(..., H, W)``.
    """
    return k_att * distance_maps(goals, coords) / diagonal


def repulsive_fields(
    sources: torch.Tensor, coords: torch.Tensor, k_rep: float, d0: float, d_min: float = 0.5
) -> torch.Tensor:
    """Per-source repulsive potentials ``k_rep * g(d(x, s))``.

    :param sources: Tensor of shape ``(..., S, 2)``.

    :return: Tensor of shape ``(..., S, H, W)`` (sum over ``S`` for the combined field).
    """
    return k_rep * khatib_profile(distance_maps(sources, coords), d0=d0, d_min=d_min)


def obstacle_field(obstacles: torch.Tensor, k_obs: float, d0: float, d_min: float = 0.5) -> torch.Tensor:
    """Static obstacle potential.

    :param obstacles: Boolean tensor of shape ``(H, W)``.

    :return: Float tensor of shape ``(H, W)``. All zeros if there are no obstacles.
    """
    height, width = obstacles.shape
    coords = grid_coordinates(height, width, device=obstacles.device)
    field = torch.zeros((height, width), device=obstacles.device, dtype=torch.float32)
    if not obstacles.any() or k_obs == 0:
        return field
    obstacle_cells = coords[obstacles]  # (O, 2)
    # distance from each cell to the nearest obstacle cell
    nearest = torch.cdist(coords.view(-1, 2), obstacle_cells).min(dim=-1).values.view(height, width)
    field = k_obs * khatib_profile(nearest, d0=d0, d_min=d_min)
    field[obstacles] = k_obs
    return field


def compute_agent_fields(
    *,
    pursuers: torch.Tensor,
    evader: torch.Tensor,
    coords: torch.Tensor,
    obstacle_potential: torch.Tensor,
    k_att: float,
    k_pp: float,
    d0_pp: float,
    k_ep: float,
    d0_ep: float,
    k_center: float,
    d_min: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Compute every agent's potential field and its components.

    :param pursuers: Pursuer positions, shape ``(E, N, 2)``.
    :param evader: Evader position, shape ``(E, 2)``.
    :param coords: Grid coordinates, shape ``(H, W, 2)``.
    :param obstacle_potential: Static obstacle field, shape ``(H, W)``.

    :return: Dictionary of tensors of shape ``(E, N + 1, H, W)`` (pursuers first, evader last):

        - ``"total"``: full potential of each agent.
        - ``"attractive"``: attractive component (evader: attraction to the grid centre).
        - ``"repulsive"``: agent-induced repulsive component (inter-pursuer / pursuer-to-evader).
        - ``"obstacle"``: obstacle component (broadcast of ``obstacle_potential``).
    """
    height, width = coords.shape[:2]
    diagonal = grid_diagonal(height, width)
    num_envs, num_pursuers = pursuers.shape[:2]

    # pursuers: attraction to the evader (same for all pursuers) + repulsion from the other pursuers
    att_p = attractive_field(evader, coords, k_att, diagonal)  # (E, H, W)
    att_p = att_p[:, None].expand(num_envs, num_pursuers, height, width)
    if k_pp and num_pursuers > 1:
        rep_each = repulsive_fields(pursuers, coords, k_pp, d0_pp, d_min)  # (E, N, H, W)
        rep_p = rep_each.sum(dim=1, keepdim=True) - rep_each  # exclude own contribution
    else:
        rep_p = torch.zeros_like(att_p)

    # evader: repulsion from all pursuers + weak attraction to the grid centre
    rep_e = repulsive_fields(pursuers, coords, k_ep, d0_ep, d_min).sum(dim=1)  # (E, H, W)
    centre = torch.tensor([(height - 1) / 2.0, (width - 1) / 2.0], device=coords.device)
    att_e = (k_center * distance_maps(centre, coords) / diagonal).expand(num_envs, height, width)

    attractive = torch.cat([att_p, att_e[:, None]], dim=1)
    repulsive = torch.cat([rep_p, rep_e[:, None]], dim=1)
    obstacle = obstacle_potential.expand_as(attractive)
    return {
        "total": attractive + repulsive + obstacle,
        "attractive": attractive,
        "repulsive": repulsive,
        "obstacle": obstacle,
    }
