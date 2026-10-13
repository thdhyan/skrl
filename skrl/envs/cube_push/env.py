"""Partially observable cube push in MuJoCo.

The agent commands a pusher tip (position targets in x, y, z) and is told only the wanted final cube pose.
Hidden: the cube's initial pose (x, y, yaw) and its friction μ, mass m and damping k_d. It observes the pusher's
encoders and contact forces (noisy), so it has to find the cube by touch and infer its physics from the forces.

With the default gantry the tip rides on x, y, z slides; with ``CubePushCfg(pusher="arm")`` the same tip is
carried by a 3-joint arm (shoulder/elbow yaw + vertical slide) under analytic IK — the hierarchical split: the
high level only ever sees end-effector targets, joint control stays in the low level.

Observation ``[tip xyz (3), tip force xyz (3), goal (x, y, cos 4ψ, sin 4ψ)]``: the goal yaw ψ appears as
``4ψ`` because the cube looks the same every 90°. Action: the tip position target, clipped to ``max_speed``.
``info`` carries the true cube pose and physics for evaluation and debug rendering only.
"""
from __future__ import annotations

import math
from typing import Any

import gymnasium
import mujoco
import numpy as np

from skrl.envs.cube_push.sim import (
    CubePushCfg,
    CubePushModel,
    MESH_ARMS,
    N_JOINTS,
    N_TIP,
    cube_pose,
    expand_controls,
    fk,
    ik,
    mjcf,
    sense,
    substep_controls,
    wrap_yaw,
)


def pose_error(pose: np.ndarray, goal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Position and yaw (modulo 90°) errors of ``[..., 3]`` poses to the goal."""
    pose = np.asarray(pose)
    return np.linalg.norm(pose[..., :2] - goal[:2], axis=-1), np.abs(wrap_yaw(pose[..., 2] - goal[2]))


class CubePushEnv(gymnasium.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, cfg: CubePushCfg | None = None, render_mode: str | None = None) -> None:
        self.cfg = cfg or CubePushCfg()
        self.sim = CubePushModel(self.cfg)
        self.model = mujoco.MjModel.from_xml_string(mjcf(self.cfg))
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self._renderer = None
        high = np.full(10, np.inf, dtype=np.float32)
        self.observation_space = gymnasium.spaces.Box(-high, high, dtype=np.float32)
        tx, ty = self.cfg.table_half
        self.action_space = gymnasium.spaces.Box(
            np.array([-tx, -ty, self.cfg.z_low], np.float32), np.array([tx, ty, 0.2], np.float32), dtype=np.float32
        )
        self.goal = np.zeros(3)
        self.physics = np.zeros(3)
        self.target = np.zeros(3)
        self.joint_target = np.zeros(3)  # arm mode: previous joint-space target (EE targets map through arm_ik)

    # --- gymnasium API --------------------------------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        cfg, rng = self.cfg, self.np_random
        options = options or {}
        pose = options.get("cube_pose")
        if pose is None:
            pose = [rng.uniform(*cfg.cube_x), rng.uniform(*cfg.cube_y), wrap_yaw(rng.uniform(-math.pi, math.pi))]
        physics = options.get("physics")
        if physics is None:
            physics = [rng.uniform(*cfg.mu), rng.uniform(*cfg.mass), rng.uniform(*cfg.damping)]
        goal = options.get("goal")
        if goal is None:
            goal = [rng.uniform(*cfg.goal_x), rng.uniform(*cfg.goal_y), wrap_yaw(rng.uniform(-math.pi, math.pi))]
        self.physics, self.goal = np.asarray(physics, float), np.asarray(goal, float)
        self.sim.set_physics(self.model, *self.physics)
        self.target = np.array([*cfg.pusher_start, cfg.z_high])
        self.sim.set_state(self.data, self.target, pose)
        if cfg.pusher in MESH_ARMS:
            # Numeric IK will be solved in step; init joint_target from current qpos
            n = N_JOINTS[cfg.pusher]
            self.joint_target = self.data.qpos[:n].copy()
        else:
            self.joint_target = np.asarray(ik(cfg, self.target), dtype=np.float64).ravel()
        mujoco.mj_forward(self.model, self.data)
        self.steps = 0
        return self._observation(), self._info()

    def step(self, action):
        cfg = self.cfg
        n = N_JOINTS[cfg.pusher]
        limit = cfg.max_speed * cfg.control_dt
        target = np.clip(np.asarray(action, float), self.action_space.low, self.action_space.high)
        target = self.target + np.clip(target - self.target, -limit, limit)
        if cfg.pusher == "gantry":
            for ctrl in substep_controls(self.target, target[None], cfg.substeps):
                self.data.ctrl[:n] = ctrl
                mujoco.mj_step(self.model, self.data)
        elif cfg.pusher in MESH_ARMS:
            # Numeric IK via expand_controls with model/data
            controls = expand_controls(cfg, self.target, target[None], cfg.substeps, self.model, self.data)
            for ctrl in controls:
                self.data.ctrl[:n] = ctrl
                mujoco.mj_step(self.model, self.data)
            self.joint_target = self.data.qpos[:n].copy()
        else:
            joint = np.asarray(ik(cfg, target), dtype=np.float64).ravel()
            for ctrl in substep_controls(self.joint_target, joint[None], cfg.substeps):
                self.data.ctrl[:n] = ctrl
                mujoco.mj_step(self.model, self.data)
            self.joint_target = joint
        self.target = target
        self.steps += 1
        pose = self.cube_pose()
        dist, dyaw = pose_error(pose, self.goal)
        success = bool(dist < cfg.goal_tolerance[0] and dyaw < cfg.goal_tolerance[1])
        off_table = bool(np.any(np.abs(pose[:2]) > np.asarray(cfg.table_half) - cfg.cube_half))
        reward = -float(dist) - 0.05 * float(dyaw) - (10.0 if off_table else 0.0)
        info = self._info()
        info.update(success=success, off_table=off_table)
        return self._observation(), reward, off_table, self.steps >= cfg.max_steps, info

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, 480, 640)
        self._renderer.update_scene(self.data, camera="overview")
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # --- helpers --------------------------------------------------------------------------------------------------

    def cube_pose(self) -> np.ndarray:
        return cube_pose(self.data.sensordata, self.cfg)

    def _observation(self) -> np.ndarray:
        rng, cfg = self.np_random, self.cfg
        tip, force = sense(cfg, self.data.sensordata)
        tip = tip + rng.normal(0.0, cfg.position_noise, 3)
        force = force + rng.normal(0.0, cfg.force_noise, 3)
        g = self.goal
        return np.concatenate([tip, force, [g[0], g[1], math.cos(4 * g[2]), math.sin(4 * g[2])]]).astype(np.float32)

    def _info(self) -> dict[str, Any]:
        return {"cube_pose": self.cube_pose(), "physics": self.physics.copy(), "goal": self.goal.copy()}

