"""Debug video of the cube push belief: tiled MuJoCo views with the belief drawn into the scene, and belief plots.

Panels: overview camera, top-down camera, follow camera on the pusher, belief plots (physics and yaw marginals with
the true values, entropy and P(goal) over time). In the 3D views (transparent markers): the most likely belief cells
as flat orange tiles (opacity ∝ mass), the MAP state w* as a blue box, the goal pose as a green box with its
position tolerance disc, the table limit the cube must stay inside in red, and the current WRT's pusher path in yellow.
"""
from __future__ import annotations

import math
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PANEL_H, PANEL_W, HEADER_H = 360, 480, 64


def _quat_mat(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([c, -s, 0, s, c, 0, 0, 0, 1], dtype=np.float64)


def _add(scene, gtype, size, pos, mat, rgba) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom], gtype, np.asarray(size, np.float64),
                        np.asarray(pos, np.float64), np.asarray(mat, np.float64), np.asarray(rgba, np.float32))
    scene.ngeom += 1


def _line(scene, a, b, width, rgba) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                        np.asarray(rgba, np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, np.asarray(a, np.float64), np.asarray(b, np.float64))
    scene.ngeom += 1


class BeliefVideo:
    def __init__(self, env, planner, status: Callable[[str], str], path: str, every: int = 2, fps: int = 25) -> None:
        """:param status: Header text (up to 3 lines) for the current execution phase.
        :param path: Output mp4, written while recording."""
        import imageio.v2 as imageio

        self.env, self.planner, self.status, self.every, self.fps = env, planner, status, every, fps
        self.writer = imageio.get_writer(path, fps=fps, quality=7, macro_block_size=16)
        self.n_frames = 0
        self.renderer = mujoco.Renderer(env.model, PANEL_H, PANEL_W)
        self.renderer.scene.maxgeom = 10000 if self.renderer.scene.maxgeom < 10000 else self.renderer.scene.maxgeom
        self.follow = mujoco.MjvCamera()
        self.follow.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.follow.distance, self.follow.elevation, self.follow.azimuth = 0.6, -50.0, 135.0
        self.ticks = 0
        self.wrt = None
        self.phase = "start"
        self.history: list[tuple[int, float, float]] = []  # (wrt index, entropy, P_goal)
        self._cells = None
        self._plot = None
        self.font = ImageFont.load_default(size=15)
        self.refresh()

    # --- belief snapshot ------------------------------------------------------------------------------------------

    def refresh(self, n_cells: int = 120) -> None:
        """Re-read the belief (after an update): the most likely (x, y) cells, each drawn at its most likely yaw, and
        the plots panel."""
        b = self.planner.belief
        pose = b.marginal((0, 1, 2))
        xy = pose.sum(2)
        flat = np.argsort(-xy.ravel())[:n_cells]
        flat = flat[xy.ravel()[flat] > 1e-4]
        ix, iy = np.unravel_index(flat, xy.shape)
        yaw = b.axes[2][pose[ix, iy].argmax(-1)]
        mass = xy[ix, iy]
        self._cells = (np.stack([b.axes[0][ix], b.axes[1][iy], yaw], -1), mass / max(mass.max(), 1e-12))
        self._plot = self._plots()

    def _plots(self) -> np.ndarray:
        b, env = self.planner.belief, self.env
        fig, axes = plt.subplots(2, 2, figsize=(PANEL_W / 80, PANEL_H / 80), dpi=80)
        names = [("μ", 3, 0), ("m [kg]", 4, 1), ("k_d [N s/m]", 5, 2)]
        width = 0.25
        ax = axes[0, 0]
        for k, (name, d, t) in enumerate(names):
            p = b.marginal((d,))
            ax.bar(np.arange(3) + (k - 1) * width, p, width, label=name, color=f"C{k}")
        ax.set_xticks(range(3), ["low", "mid", "high"])
        ax.set_ylim(0, 1)
        ax.set_title("physics marginals", fontsize=9)
        ax.legend(fontsize=7, loc="upper left")
        ax = axes[0, 1]
        ax.axis("off")
        lines = []
        for name, d, t in names:
            p = b.marginal((d,))
            est = float(p @ b.axes[d])
            lines.append(f"{name:12s} true {env.physics[t]:.2f}  E_b {est:.2f}")
        ax.text(0.0, 0.95, "\n".join(lines), fontsize=8, va="top", family="monospace")
        ax.text(0.0, 0.35, "levels  μ " + "/".join(f"{v:.2f}" for v in b.axes[3]) + "\n        m "
                + "/".join(f"{v:.2f}" for v in b.axes[4]) + "\n      k_d " + "/".join(f"{v:.1f}" for v in b.axes[5]),
                fontsize=7, va="top", family="monospace", color="0.35")
        ax = axes[1, 0]
        yaw = np.degrees(b.axes[2])
        ax.bar(yaw, b.marginal((2,)), width=8, color="C1")
        true_yaw = math.degrees(env.cube_pose()[2])
        ax.axvline(true_yaw, color="k", lw=1.5, label="true")
        ax.axvline(math.degrees(self.planner.goal[2]), color="g", lw=1.5, ls="--", label="goal")
        ax.set_xlim(-45, 45)
        ax.set_ylim(0, 1)
        ax.set_title("yaw marginal (mod 90°)", fontsize=9)
        ax.legend(fontsize=7)
        ax = axes[1, 1]
        if self.history:
            h = np.array(self.history)
            ax.plot(h[:, 0], h[:, 1], "C0.-", label="H(b) [nats]")
            ax2 = ax.twinx()
            ax2.plot(h[:, 0], h[:, 2], "C2.-", label="P(goal)")
            ax2.set_ylim(0, 1)
            ax2.tick_params(labelsize=7)
            ax.legend(fontsize=7, loc="upper right")
            ax2.legend(fontsize=7, loc="center right")
        ax.set_title("entropy / P(goal) per WRT", fontsize=9)
        for a in axes.ravel():
            a.tick_params(labelsize=7)
        fig.tight_layout(pad=0.4)
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        return np.asarray(Image.fromarray(img).resize((PANEL_W, PANEL_H)))

    # --- 3D markers -----------------------------------------------------------------------------------------------

    def _markers(self, scene) -> None:
        e = self.env.cfg
        h = e.cube_half
        cells, alpha = self._cells
        for (x, y, yaw), a in zip(cells, alpha):
            _add(scene, mujoco.mjtGeom.mjGEOM_BOX, [0.5 * h, 0.5 * h, 0.0015], [x, y, 0.002], _quat_mat(yaw),
                 [0.85, 0.35, 0.05, 0.05 + 0.6 * a])
        w = self.planner.map_state()
        _add(scene, mujoco.mjtGeom.mjGEOM_BOX, [h * 1.02] * 3, [w[0], w[1], h], _quat_mat(w[2]), [0.1, 0.3, 1.0, 0.3])
        g = self.planner.goal
        _add(scene, mujoco.mjtGeom.mjGEOM_BOX, [h * 1.02] * 3, [g[0], g[1], h], _quat_mat(g[2]), [0.1, 0.9, 0.2, 0.3])
        _add(scene, mujoco.mjtGeom.mjGEOM_CYLINDER, [e.goal_tolerance[0], 0.001, 0], [g[0], g[1], 0.001],
             np.eye(3).ravel(), [0.1, 0.9, 0.2, 0.35])
        tx, ty = np.asarray(e.table_half) - h
        corners = [(-tx, -ty), (tx, -ty), (tx, ty), (-tx, ty)]
        for a, b in zip(corners, corners[1:] + corners[:1]):
            _line(scene, [*a, 0.003], [*b, 0.003], 0.003, [1.0, 0.1, 0.1, 0.5])
        if self.wrt is not None:
            pts = np.concatenate([self.wrt.start[None], self.wrt.targets])[::3]
            for p in pts:
                _add(scene, mujoco.mjtGeom.mjGEOM_SPHERE, [0.004, 0, 0], p, np.eye(3).ravel(), [1.0, 0.9, 0.1, 0.5])

    def _view(self, camera) -> np.ndarray:
        self.renderer.update_scene(self.env.data, camera=camera)
        self._markers(self.renderer.scene)
        return self.renderer.render()

    # --- frames ---------------------------------------------------------------------------------------------------

    def _label(self, img: np.ndarray, text: str) -> np.ndarray:
        im = Image.fromarray(img)
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, 7 * len(text) + 10, 18], fill=(0, 0, 0))
        d.text((5, 1), text, fill=(255, 255, 255), font=self.font)
        return np.asarray(im)

    def frame(self) -> np.ndarray:
        self.follow.lookat[:] = self.env.data.qpos[:3] * [1, 1, 0]
        overview = self._label(self._view("overview"), "overview")
        top = self._label(self._view("top"), "top-down")
        follow = self._label(self._view(self.follow), "follow: pusher")
        plots = self._plot
        grid = np.concatenate([np.concatenate([overview, top], 1), np.concatenate([follow, plots], 1)], 0)
        header = Image.new("RGB", (grid.shape[1], HEADER_H), (20, 20, 28))
        d = ImageDraw.Draw(header)
        for k, line in enumerate(self.status(self.phase).split("\n")[:3]):
            d.text((8, 4 + 19 * k), line, fill=(235, 235, 235), font=self.font)
        return np.concatenate([np.asarray(header), grid], 0)

    def _write(self, frame: np.ndarray, repeat: int = 1) -> None:
        for _ in range(repeat):
            self.writer.append_data(frame)
        self.n_frames += repeat

    def on_step(self, phase: str) -> None:
        self.ticks += 1
        self.phase = phase
        if self.ticks % self.every == 0:
            self._write(self.frame())

    def hold(self, seconds: float = 1.0) -> np.ndarray:
        f = self.frame()
        self._write(f, int(seconds * self.fps))
        return f

    def close(self) -> None:
        self.writer.close()
        self.renderer.close()
