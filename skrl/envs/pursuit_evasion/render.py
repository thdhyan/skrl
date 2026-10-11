"""Debug rendering of pursuit-evasion rollouts: potential fields, agent trails and capture radius, as video frames.

Each frame has three labelled panels for one sub-environment:

* the total potential field of pursuer 0 (attraction to the evader + repulsion from the other pursuers),
* the total potential field of the evader (repulsion from the pursuers + weak attraction to the centre),
* an overview of the arena with the trails of every agent.

All panels show the arena boundary, obstacles, the agents and the capture region around the evader. A header line
gives the step, the closest pursuer-evader distance and any caller-supplied text (generation, opponent, ...).

Example::

    renderer = RolloutRenderer(env)              # raw PursuitEvasionEnv or its skrl wrapper
    frames = [renderer.frame("gen 3")]
    ... env.step(actions); frames.append(renderer.frame("gen 3"))
    write_video("rollout.mp4", frames)
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

PURSUER_COLOR = "#1f77b4"
EVADER_COLOR = "#d62728"


class RolloutRenderer:
    def __init__(self, env: Any, env_id: int = 0, trail_length: int = 60, dpi: int = 80) -> None:
        """Renderer of one sub-environment of a :class:`~skrl.envs.pursuit_evasion.PursuitEvasionEnv`.

        :param env: Raw environment or a skrl wrapper around it.
        :param env_id: Sub-environment to draw.
        :param trail_length: Number of past positions drawn per agent.
        :param dpi: Figure resolution.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        self.env = getattr(env, "_unwrapped", env)
        self.env_id = env_id
        self.trail_length = trail_length
        self.trails: list[list[tuple[int, int]]] = [[] for _ in range(self.env.num_agents)]
        self._last_timestep = -1
        self._fig, self._axes = plt.subplots(1, 3, figsize=(15, 5.6), dpi=dpi)
        self._fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.88, wspace=0.03)

    def reset(self) -> None:
        """Forget the trails (call when a new episode starts)."""
        self.trails = [[] for _ in range(self.env.num_agents)]

    def _draw_common(self, ax: Any, positions: np.ndarray) -> None:
        env, cfg = self.env, self.env.cfg
        h, w = env.height, env.width
        ax.add_patch(_rectangle(-0.5, -0.5, w, h, edgecolor="black", linewidth=2.0, fill=False))  # arena boundary
        obstacles = env.obstacles.cpu().numpy()
        if obstacles.any():
            rows, cols = np.nonzero(obstacles)
            ax.scatter(cols, rows, marker="s", s=max(4, 3000 // max(h, w)), c="black")
        # capture region around the evader (manhattan: diamond, chebyshev: square), transparent
        er, ec = positions[-1]
        r = cfg.capture_radius + 0.5
        if cfg.capture_metric == "manhattan":
            corners = [(ec, er - r), (ec + r, er), (ec, er + r), (ec - r, er)]
        else:
            corners = [(ec - r, er - r), (ec + r, er - r), (ec + r, er + r), (ec - r, er + r)]
        ax.add_patch(_polygon(corners, facecolor=EVADER_COLOR, alpha=0.25, edgecolor=EVADER_COLOR))
        size = max(80, 6000 // max(h, w))
        for i, (row, col) in enumerate(positions[:-1]):
            ax.scatter([col], [row], s=size, c=PURSUER_COLOR, edgecolors="white", zorder=5)
            ax.annotate(f"P{i}", (col, row), color="white", fontsize=7, ha="center", va="center", zorder=6)
        ax.scatter([ec], [er], s=size, c=EVADER_COLOR, marker="*", edgecolors="white", zorder=5)
        ax.set_xlim(-0.5, w - 0.5)
        ax.set_ylim(h - 0.5, -0.5)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    def frame(self, header: str = "", positions: Any = None, timestep: int | None = None) -> np.ndarray:
        """Draw the current state of the sub-environment.

        :param header: Extra text appended to the status line.
        :param positions: ``(A, 2)`` positions to draw instead of the current ones (e.g. ``info["final_positions"]``
            of an episode that was just reset automatically).
        :param timestep: Step shown in the status line (default: the sub-environment's current step).

        :return: RGB image ``(H, W, 3)`` (``uint8``).
        """
        env = self.env
        if positions is None:
            positions = env.positions[self.env_id]
        if timestep is None:
            timestep = int(env.timestep[self.env_id])
        if timestep <= self._last_timestep:  # the sub-environment was reset
            self.reset()
        self._last_timestep = timestep
        positions_t = torch.as_tensor(positions, device=env.device).view(1, -1, 2)
        positions = positions_t[0].cpu().numpy()
        for trail, position in zip(self.trails, positions):
            trail.append((int(position[0]), int(position[1])))
            del trail[: -self.trail_length]

        fields = env.potential_fields_at(positions_t)["total"][0].cpu().numpy()  # (A, H, W)
        clip = env.cfg.potential_clip
        titles = ["pursuer 0: total potential", "evader: total potential", "overview + trails"]
        for ax, title in zip(self._axes, titles):
            ax.clear()
            ax.set_title(title, fontsize=10)
        for ax, field in ((self._axes[0], fields[0]), (self._axes[1], fields[-1])):
            # per-frame colour range: far from the agents the fields are flat compared to potential_clip
            field = np.clip(field, 0, clip)
            ax.imshow(field, cmap="viridis", vmin=field.min(), vmax=field.max(), interpolation="nearest")
            self._draw_common(ax, positions)
        overview = self._axes[2]
        overview.imshow(np.full((env.height, env.width), 0.95), cmap="gray", vmin=0, vmax=1)
        for i, trail in enumerate(self.trails):
            rows, cols = zip(*trail)
            color = EVADER_COLOR if i == env.num_pursuers else PURSUER_COLOR
            overview.plot(cols, rows, color=color, alpha=0.7, linewidth=2.0)
        self._draw_common(overview, positions)

        distance = np.abs(positions[:-1] - positions[-1]).sum(-1).min()
        status = (
            f"step {timestep}/{env.cfg.max_steps} | arena {env.height}x{env.width} | closest pursuer "
            f"(manhattan) {distance} | capture radius {env.cfg.capture_radius} | obs {env.cfg.obs_mode}"
        )
        self._fig.suptitle(status + (f" | {header}" if header else ""), fontsize=10)
        self._fig.canvas.draw()
        image = np.asarray(self._fig.canvas.buffer_rgba())[..., :3]
        return image.copy()

    def close(self) -> None:
        import matplotlib.pyplot as plt

        plt.close(self._fig)


def _rectangle(x: float, y: float, w: float, h: float, **kwargs: Any) -> Any:
    from matplotlib.patches import Rectangle

    return Rectangle((x, y), w, h, **kwargs)


def _polygon(corners: Sequence[tuple[float, float]], **kwargs: Any) -> Any:
    from matplotlib.patches import Polygon

    return Polygon(corners, closed=True, **kwargs)


def write_video(path: str, frames: Sequence[np.ndarray], fps: int = 10) -> str:
    """Write frames to a video (``.mp4`` via imageio-ffmpeg, ``.gif`` otherwise)."""
    import imageio

    imageio.mimsave(path, list(frames), fps=fps)
    return path
