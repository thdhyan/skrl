"""MuJoCo model of the planar cube push: a position-controlled cylindrical pusher (x, y, z slides) and a free cube on
a table, with the cube's friction μ, mass m and damping k_d settable per model.

The same control expansion (:func:`substep_controls`) drives the environment and the batched hypothesis rollouts
(:func:`rollout`), so a hypothesis equal to the true state predicts the true sensor trace up to sensor noise.
"""
from __future__ import annotations

import dataclasses
import math

import mujoco
import numpy as np
from mujoco import rollout as mj_rollout


@dataclasses.dataclass
class CubePushCfg:
    """Scene, control and randomization settings of the cube push."""

    timestep: float = 0.002
    substeps: int = 10  # physics steps per control step (control period 0.02 s)
    cube_half: float = 0.03
    pusher_radius: float = 0.012
    pusher_half_height: float = 0.03
    pusher_kp: float = 3000.0
    pusher_kv: float = 60.0
    z_low: float = 0.033  # pusher centre when lowered (bottom 3 mm above the table)
    z_high: float = 0.12  # above the cube for transit
    max_speed: float = 0.5  # m/s per axis, clips the commanded target change
    table_half: tuple[float, float] = (0.45, 0.35)
    pusher_start: tuple[float, float] = (-0.2, 0.0)
    # hidden state: initial cube pose and physics, sampled uniformly
    cube_x: tuple[float, float] = (0.0, 0.12)
    cube_y: tuple[float, float] = (-0.1, 0.1)
    mu: tuple[float, float] = (0.3, 0.8)
    mass: tuple[float, float] = (0.2, 0.6)
    damping: tuple[float, float] = (0.0, 3.0)  # N s/m on the cube's linear dofs (k_d h² on the angular ones)
    # command: the wanted final cube pose
    goal_x: tuple[float, float] = (0.18, 0.3)
    goal_y: tuple[float, float] = (-0.12, 0.12)
    goal_tolerance: tuple[float, float] = (0.02, math.radians(10.0))  # position (m), yaw (rad, modulo 90°)
    # sensors
    position_noise: float = 0.0005  # pusher encoder noise (m)
    force_noise: float = 0.05  # pusher actuator force noise (N)
    max_steps: int = 4000  # control steps per episode (80 s)

    @property
    def control_dt(self) -> float:
        return self.timestep * self.substeps


def mjcf(cfg: CubePushCfg) -> str:
    h, r, hh = cfg.cube_half, cfg.pusher_radius, cfg.pusher_half_height
    tx, ty = cfg.table_half
    ctrl = f'kp="{cfg.pusher_kp}" kv="{cfg.pusher_kv}"'
    return f"""
<mujoco model="cube_push">
  <option timestep="{cfg.timestep}" integrator="implicitfast"/>
  <visual>
    <global offwidth="1280" offheight="960"/><quality shadowsize="2048"/>
    <headlight ambient=".25 .25 .25" diffuse=".3 .3 .3" specular="0 0 0"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".72 .7 .66" rgb2=".64 .62 .58" width="300" height="300"/>
    <material name="table" texture="grid" texrepeat="9 7"/>
  </asset>
  <worldbody>
    <light pos="0 -0.4 1.5" dir="0 0.25 -1" diffuse=".45 .45 .45"/>
    <geom name="table" type="plane" size="{tx} {ty} 0.01" material="table" friction="0.01 0.005 0.0001"/>
    <camera name="overview" pos="-0.55 -0.75 0.65" xyaxes="0.81 -0.59 0 0.33 0.46 0.82"/>
    <camera name="top" pos="0.1 0 1.05" xyaxes="1 0 0 0 1 0"/>
    <body name="pusher" gravcomp="1">
      <joint name="px" type="slide" axis="1 0 0"/>
      <joint name="py" type="slide" axis="0 1 0"/>
      <joint name="pz" type="slide" axis="0 0 1"/>
      <geom name="pusher" type="cylinder" size="{r} {hh}" mass="0.1" priority="1" friction="0.3 0.005 0.0001"
            rgba=".25 .25 .3 1"/>
    </body>
    <body name="cube" pos="0.06 0 {h}">
      <freejoint name="cube"/>
      <geom name="cube" type="box" size="{h} {h} {h}" mass="0.4" friction="0.5 0.005 0.0001" rgba=".9 .45 .1 1"/>
    </body>
  </worldbody>
  <actuator>
    <position name="ax" joint="px" {ctrl} ctrllimited="false"/>
    <position name="ay" joint="py" {ctrl} ctrllimited="false"/>
    <position name="az" joint="pz" {ctrl} ctrllimited="false"/>
  </actuator>
  <sensor>
    <actuatorfrc actuator="ax"/><actuatorfrc actuator="ay"/><actuatorfrc actuator="az"/>
    <jointpos joint="px"/><jointpos joint="py"/><jointpos joint="pz"/>
    <framepos objtype="body" objname="cube"/>
    <framequat objtype="body" objname="cube"/>
  </sensor>
</mujoco>
"""


# sensordata layout
FORCE, PUSHER, CUBE_POS, CUBE_QUAT = slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 13)


def wrap_yaw(yaw):
    """Cube yaw modulo its 90° symmetry, in [-45°, 45°)."""
    return np.mod(np.asarray(yaw) + math.pi / 4, math.pi / 2) - math.pi / 4


def quat_yaw(quat: np.ndarray) -> np.ndarray:
    """Yaw of ``[..., (w, x, y, z)]`` quaternions, wrapped to the cube symmetry."""
    w, x, y, z = (quat[..., i] for i in range(4))
    return wrap_yaw(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def cube_pose(sensordata: np.ndarray) -> np.ndarray:
    """``[..., (x, y, yaw)]`` of the cube from sensordata."""
    return np.concatenate([sensordata[..., CUBE_POS][..., :2], quat_yaw(sensordata[..., CUBE_QUAT])[..., None]], -1)


class CubePushModel:
    """The compiled model plus helpers to set the hidden physics and build rollout states."""

    def __init__(self, cfg: CubePushCfg | None = None) -> None:
        self.cfg = cfg or CubePushCfg()
        self.model = mujoco.MjModel.from_xml_string(mjcf(self.cfg))
        self.cube_body = self.model.body("cube").id
        self.cube_geom = self.model.geom("cube").id
        self.cube_dof = self.model.jnt_dofadr[self.model.joint("cube").id]
        self.cube_qpos = self.model.jnt_qposadr[self.model.joint("cube").id]
        self._inertia = self.model.body_inertia[self.cube_body].copy() / self.model.body_mass[self.cube_body]
        self._spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
        self._variants: dict[tuple, mujoco.MjModel] = {}

    def set_physics(self, model: mujoco.MjModel, mu: float, mass: float, damping: float) -> None:
        model.geom_friction[self.cube_geom, 0] = mu
        model.body_mass[self.cube_body] = mass
        model.body_inertia[self.cube_body] = self._inertia * mass
        d = self.cube_dof
        model.dof_damping[d : d + 3] = damping
        model.dof_damping[d + 3 : d + 6] = damping * self.cfg.cube_half**2

    def variant(self, mu: float, mass: float, damping: float) -> mujoco.MjModel:
        """A copy of the model with the given cube physics (cached per value)."""
        key = (round(mu, 6), round(mass, 6), round(damping, 6))
        if key not in self._variants:
            model = mujoco.MjModel.from_xml_string(mjcf(self.cfg))
            self.set_physics(model, *key)
            self._variants[key] = model
        return self._variants[key]

    def set_state(self, data: mujoco.MjData, pusher_xyz, cube_xyyaw) -> None:
        """Pusher at ``pusher_xyz`` (control too), cube resting at ``(x, y, yaw)``, everything at rest."""
        mujoco.mj_resetData(self.model, data)
        data.qpos[:3] = pusher_xyz
        data.ctrl[:3] = pusher_xyz
        x, y, yaw = cube_xyyaw
        a = self.cube_qpos
        data.qpos[a : a + 7] = [x, y, self.cfg.cube_half, math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]

    def initial_states(self, pusher_xyz, poses: np.ndarray) -> np.ndarray:
        """Rollout initial states ``[K, nstate]`` for cube poses ``[K, 3]`` and a common pusher position."""
        data = mujoco.MjData(self.model)
        states = np.empty((len(poses), mujoco.mj_stateSize(self.model, self._spec)))
        for i, pose in enumerate(poses):
            self.set_state(data, pusher_xyz, pose)
            mujoco.mj_forward(self.model, data)
            mujoco.mj_getState(self.model, data, states[i], self._spec)
        return states


def substep_controls(previous: np.ndarray, targets: np.ndarray, substeps: int) -> np.ndarray:
    """Per-physics-step pusher controls ``[len(targets) * substeps, 3]``: linear interpolation from the previous
    target to each control-step target."""
    previous = np.asarray(previous, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64).reshape(-1, 3)
    starts = np.concatenate([previous[None], targets[:-1]])
    alpha = (np.arange(1, substeps + 1) / substeps)[None, :, None]
    return (starts[:, None] + alpha * (targets - starts)[:, None]).reshape(-1, 3)


_DATAS: list[mujoco.MjData] = []


def rollout(models: list[mujoco.MjModel], states: np.ndarray, controls: np.ndarray) -> np.ndarray:
    """Batched open-loop rollouts (one model per initial state, multithreaded) of shared controls ``[nstep, nu]`` or
    per-state controls ``[K, nstep, nu]``; returns sensordata at the end of every physics step, ``[K, nstep, nsens]``."""
    import os

    nthread = int(os.environ.get("CUBE_PUSH_THREADS", min(os.cpu_count() or 1, 16)))
    while len(_DATAS) < nthread:
        _DATAS.append(mujoco.MjData(models[0]))
    controls = controls if controls.ndim == 3 else controls[None]
    _, sensordata = mj_rollout.rollout(models, _DATAS[:nthread], states, controls, persistent_pool=True)
    return sensordata
