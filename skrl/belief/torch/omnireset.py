from __future__ import annotations

from dataclasses import dataclass

import torch


#: Mode ids returned by :func:`sample_omnireset_starts`.
REACHING_MODE = 0
PATHWAY_MODE = 1
NEAR_GOAL_MODE = 2


@dataclass
class OmniResetCfg:
    """Configuration for OmniReset-style diverse episode starts.

    Ported from ``Belief-based-pushing`` (Franka cube push, RSS-20-style
    belief training): at each reset the manipulated object's start position is
    drawn from a three-component mixture around a fixed goal so that training
    covers reaching, transport-corridor and near-successful states without
    exposing ground-truth state to the actor.

    Ratios must satisfy ``reaching_ratio + pathway_ratio <= 1``. The remaining
    mass is assigned to the near-goal mode.
    """

    goal_xy: tuple[float, float] = (0.62, 0.0)
    """Task-space position of the (fixed) goal."""

    start_x_range: tuple[float, float] = (0.34, 0.47)
    """Reaching-mode X range (env-local frame)."""

    start_y_range: tuple[float, float] = (-0.14, 0.14)
    """Reaching-mode Y range (env-local frame)."""

    path_alpha_range: tuple[float, float] = (0.15, 0.65)
    """Corridor-mode interpolation interval between a random start and the goal."""

    path_lateral_range: tuple[float, float] = (-0.075, 0.075)
    """Corridor-mode lateral offset range along the start→goal tangent."""

    near_goal_radius_range: tuple[float, float] = (0.012, 0.090)
    """Near-goal-mode ring radii around the goal (inner bound ≈ success threshold)."""

    reaching_ratio: float = 0.35
    """Fraction of samples in the reaching mode."""

    pathway_ratio: float = 0.30
    """Fraction of samples in the transport-corridor mode."""

    workspace_bounds_xy: tuple[tuple[float, float], tuple[float, float]] = ((0.25, 0.75), (-0.30, 0.30))
    """Task-space XY bounds all samples are clamped to."""

    workspace_margin: float = 0.010
    """Safety margin applied inside the workspace bounds."""


def sample_omnireset_starts(
    cfg: OmniResetCfg,
    num_samples: int,
    device: str | torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample OmniReset-style start positions in the env-local XY plane.

    The mixture matches ``reset_cube_and_goal_omnireset`` from the source
    repository:

    - **reaching** (``u < reaching_ratio``): uniform over
      ``[start_x_range] x [start_y_range]``;
    - **pathway** (next ``pathway_ratio``): point on the segment between a
      random start sample and the goal, parameterized by
      ``alpha ~ U(path_alpha_range)``, displaced laterally along the segment
      tangent by ``U(path_lateral_range)``;
    - **near-goal** (remainder): ring sample around the goal with radius
      ``U(near_goal_radius_range)`` and uniform angle.

    All samples are clamped to the workspace bounds (with margin).

    :param cfg: Mixture configuration.
    :param num_samples: Number of start positions to draw (e.g. the number of
        environments being reset).
    :param device: Torch device for the returned tensors.

    :return: Start positions ``[num_samples, 2]`` (env-local XY) and the
        corresponding mode ids ``[num_samples]``
        (:data:`REACHING_MODE` / :data:`PATHWAY_MODE` / :data:`NEAR_GOAL_MODE`).
    """
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    total_ratio = cfg.reaching_ratio + cfg.pathway_ratio
    if total_ratio > 1.0 + 1.0e-6:
        raise ValueError("reaching_ratio + pathway_ratio must not exceed 1")

    device = torch.device(device) if device is not None else torch.device("cpu")
    n = int(num_samples)

    mode_sample = torch.rand(n, device=device)
    reaching_mask = mode_sample < cfg.reaching_ratio
    pathway_mask = (mode_sample >= cfg.reaching_ratio) & (mode_sample < total_ratio)
    near_goal_mask = ~(reaching_mask | pathway_mask)

    start_x = torch.empty(n, device=device).uniform_(*cfg.start_x_range)
    start_y = torch.empty(n, device=device).uniform_(*cfg.start_y_range)
    xy = torch.stack((start_x, start_y), dim=-1)

    goal_xy = torch.tensor(cfg.goal_xy, device=device, dtype=xy.dtype)

    if pathway_mask.any():
        count = int(pathway_mask.sum().item())
        alpha = torch.empty(count, device=device).uniform_(*cfg.path_alpha_range).unsqueeze(-1)
        start_xy = xy[pathway_mask]
        direction = goal_xy.unsqueeze(0) - start_xy
        direction = direction / torch.linalg.norm(direction, dim=-1, keepdim=True).clamp_min(1.0e-6)
        tangent = torch.stack((-direction[:, 1], direction[:, 0]), dim=-1)
        lateral = torch.empty(count, 1, device=device).uniform_(*cfg.path_lateral_range)
        xy[pathway_mask] = (1.0 - alpha) * start_xy + alpha * goal_xy.unsqueeze(0) + lateral * tangent

    if near_goal_mask.any():
        count = int(near_goal_mask.sum().item())
        radius = torch.empty(count, device=device).uniform_(*cfg.near_goal_radius_range)
        angle = torch.empty(count, device=device).uniform_(-3.14159, 3.14159)
        offset = torch.stack((radius * torch.cos(angle), radius * torch.sin(angle)), dim=-1)
        xy[near_goal_mask] = goal_xy.unsqueeze(0) + offset

    (x_min, x_max), (y_min, y_max) = cfg.workspace_bounds_xy
    margin = cfg.workspace_margin
    xy[..., 0].clamp_(x_min + margin, x_max - margin)
    xy[..., 1].clamp_(y_min + margin, y_max - margin)

    mode_ids = (
        REACHING_MODE * reaching_mask.long()
        + PATHWAY_MODE * pathway_mask.long()
        + NEAR_GOAL_MODE * near_goal_mask.long()
    )
    return xy, mode_ids
