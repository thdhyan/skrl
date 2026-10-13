"""MuJoCo model of the planar cube push: a position-controlled cylindrical pusher (x, y, z slides) and a free cube on
a table, with the cube's friction μ, mass m and damping k_d settable per model.

The same control expansion (:func:`substep_controls`) drives the environment and the batched hypothesis rollouts
(:func:`rollout`), so a hypothesis equal to the true state predicts the true sensor trace up to sensor noise.

``pusher="arm"`` replaces the gantry slides with a 3-joint arm (shoulder yaw, elbow yaw, vertical rod slide) carrying
the same cylindrical tip. The task-level interface is unchanged: actions are end-effector (tip) position targets and
:func:`sense` reports the tip position (forward kinematics of the joint encoders) and the tip contact force (site
sensor). :func:`arm_ik` is a pure analytic function of the target, so the environment and every hypothesis expand an
end-effector trajectory to identical joint controls.
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
    # low level: "gantry" (x, y, z slides), "arm" (SCARA: shoulder/elbow yaw + vertical rod slide),
    # "polar" (cylindrical: turret yaw + radial slide + vertical rod slide) or "artic" (articulated: base yaw +
    # shoulder/elbow/wrist pitch, rod kept vertical) — all carry the same tip under analytic IK
    pusher: str = "gantry"
    arm_base: tuple[float, float] = (-0.40, 0.0)
    arm_l1: float = 0.5
    arm_l2: float = 0.5
    arm_shoulder_z: float = 0.25
    arm_branch: float = 1.0  # elbow sign of the analytic IK (fixed, so IK is a pure function of the target)
    arm_kp: tuple[float, float, float] = (150.0, 150.0, 3000.0)
    arm_kv: tuple[float, float, float] = (6.0, 6.0, 60.0)
    arm_min_reach: float = 0.08
    arm_max_reach: float = 0.97
    polar_base: tuple[float, float] = (-0.40, 0.0)
    polar_turret_z: float = 0.25
    polar_kp: tuple[float, float, float] = (150.0, 3000.0, 3000.0)
    polar_kv: tuple[float, float, float] = (6.0, 60.0, 60.0)
    polar_min_reach: float = 0.02
    polar_max_reach: float = 0.97
    artic_base: tuple[float, float] = (-0.40, 0.0)
    artic_shoulder_z: float = 0.10
    artic_l1: float = 0.5
    artic_l2: float = 0.5
    artic_rod: float = 0.25
    artic_branch: float = 1.0  # elbow-up sign of the planar IK (elbow stays clear of the table)
    artic_kp: tuple[float, float, float, float] = (150.0, 1200.0, 1200.0, 400.0)
    artic_kv: tuple[float, float, float, float] = (6.0, 18.0, 18.0, 10.0)
    artic_min_reach: float = 0.08
    artic_max_reach: float = 0.97
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
    if cfg.pusher == "arm":
        return _arm_mjcf(cfg)
    if cfg.pusher == "polar":
        return _polar_mjcf(cfg)
    if cfg.pusher == "artic":
        return _artic_mjcf(cfg)
    if cfg.pusher != "gantry":
        raise ValueError(f"unknown pusher {cfg.pusher!r} (gantry, arm, polar, artic)")
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


# sensordata layout (both models order sensors as force, joints, cube pos, cube quat, so the cube slices match)
FORCE, PUSHER, CUBE_POS, CUBE_QUAT = slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 13)


def _arm_mjcf(cfg: CubePushCfg) -> str:
    """Same table, cube, cameras and tip as the gantry, but the tip rides on a yaw/shoulder/elbow arm.

    Joints ``j0`` (shoulder yaw about z), ``j1`` (elbow yaw about z) place the wrist in the plane; ``j2`` slides
    the rod vertically, so ``j2`` *is* the tip height and the rod stays vertical (SCARA-like). The tip site carries
    a force sensor: unlike the gantry's actuator-force readings, this measures contact only (arm gravity and
    inertia never pass through the tip site).
    """
    h, r = cfg.cube_half, cfg.pusher_radius
    tx, ty = cfg.table_half
    bx, by = cfg.arm_base
    l1, l2, sz = cfg.arm_l1, cfg.arm_l2, cfg.arm_shoulder_z
    kp0, kp1, kp2 = cfg.arm_kp
    kv0, kv1, kv2 = cfg.arm_kv
    return f"""
<mujoco model="cube_push_arm">
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
    <body name="arm_base" pos="{bx} {by} 0">
      <geom name="column" type="cylinder" size="0.035 0.125" pos="0 0 0.125" contype="0" conaffinity="0"
            rgba=".3 .3 .35 1"/>
      <body name="link1" pos="0 0 {sz}">
        <joint name="j0" type="hinge" axis="0 0 1" damping="0.2"/>
        <geom name="link1" type="box" size="{l1 / 2} 0.035 0.025" pos="{l1 / 2} 0 0" mass="0.4" contype="0"
              conaffinity="0" rgba=".2 .4 .7 1"/>
        <body name="link2" pos="{l1} 0 0">
          <joint name="j1" type="hinge" axis="0 0 1" range="-169 169" damping="0.2"/>
          <!-- MuJoCo reads hinge ranges in degrees: ±169° keeps the analytic IK branch clear of the fold -->
          <geom name="link2" type="box" size="{l2 / 2} 0.03 0.02" pos="{l2 / 2} 0 0" mass="0.3" contype="0"
                conaffinity="0" rgba=".2 .5 .8 1"/>
          <body name="rod" pos="{l2} 0 0">
            <joint name="j2" type="slide" axis="0 0 1" range="0.0 0.18"/>
            <geom name="rod" type="cylinder" size="{r} 0.125" pos="0 0 -0.125" mass="0.1" priority="1"
                  friction="0.3 0.005 0.0001" rgba=".25 .25 .3 1"/>
            <site name="tip" pos="0 0 -0.25"/>
          </body>
        </body>
      </body>
    </body>
    <body name="cube" pos="0.06 0 {h}">
      <freejoint name="cube"/>
      <geom name="cube" type="box" size="{h} {h} {h}" mass="0.4" friction="0.5 0.005 0.0001" rgba=".9 .45 .1 1"/>
    </body>
  </worldbody>
  <actuator>
    <position name="a0" joint="j0" kp="{kp0}" kv="{kv0}" ctrllimited="false"/>
    <position name="a1" joint="j1" kp="{kp1}" kv="{kv1}" ctrllimited="false"/>
    <position name="a2" joint="j2" kp="{kp2}" kv="{kv2}" ctrllimited="false"/>
  </actuator>
  <sensor>
    <force name="tipf" site="tip"/>
    <jointpos joint="j0"/><jointpos joint="j1"/><jointpos joint="j2"/>
    <framepos objtype="body" objname="cube"/>
    <framequat objtype="body" objname="cube"/>
  </sensor>
</mujoco>
"""


def _polar_mjcf(cfg: CubePushCfg) -> str:
    """Cylindrical (RPP) arm: turret yaw about z, radial slide, vertical rod slide. Same tip, site and EE
    convention as the SCARA arm; yaw is unbounded so every planar target is reachable with no branch choice."""
    h, r = cfg.cube_half, cfg.pusher_radius
    tx, ty = cfg.table_half
    bx, by = cfg.polar_base
    tz = cfg.polar_turret_z
    kp0, kp1, kp2 = cfg.polar_kp
    kv0, kv1, kv2 = cfg.polar_kv
    return f"""
<mujoco model="cube_push_polar">
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
    <body name="arm_base" pos="{bx} {by} 0">
      <geom name="column" type="cylinder" size="0.035 0.125" pos="0 0 0.125" contype="0" conaffinity="0"
            rgba=".3 .3 .35 1"/>
      <body name="turret" pos="0 0 {tz}">
        <joint name="j0" type="hinge" axis="0 0 1" damping="0.2"/>
        <geom name="guide" type="box" size="0.5 0.04 0.03" pos="0.5 0 0" mass="0.3" contype="0" conaffinity="0"
              rgba=".2 .7 .4 1"/>
        <body name="carriage" pos="0 0 0">
          <joint name="j1" type="slide" axis="1 0 0" range="0.02 1.0"/>
          <geom name="carriage" type="box" size="0.06 0.05 0.05" mass="0.2" contype="0" conaffinity="0"
                rgba=".2 .8 .5 1"/>
          <body name="rod" pos="0 0 0">
            <joint name="j2" type="slide" axis="0 0 1" range="0.0 0.18"/>
            <geom name="rod" type="cylinder" size="{r} 0.125" pos="0 0 -0.125" mass="0.1" priority="1"
                  friction="0.3 0.005 0.0001" rgba=".25 .25 .3 1"/>
            <site name="tip" pos="0 0 -0.25"/>
          </body>
        </body>
      </body>
    </body>
    <body name="cube" pos="0.06 0 {h}">
      <freejoint name="cube"/>
      <geom name="cube" type="box" size="{h} {h} {h}" mass="0.4" friction="0.5 0.005 0.0001" rgba=".9 .45 .1 1"/>
    </body>
  </worldbody>
  <actuator>
    <position name="a0" joint="j0" kp="{kp0}" kv="{kv0}" ctrllimited="false"/>
    <position name="a1" joint="j1" kp="{kp1}" kv="{kv1}" ctrllimited="false"/>
    <position name="a2" joint="j2" kp="{kp2}" kv="{kv2}" ctrllimited="false"/>
  </actuator>
  <sensor>
    <force name="tipf" site="tip"/>
    <jointpos joint="j0"/><jointpos joint="j1"/><jointpos joint="j2"/>
    <framepos objtype="body" objname="cube"/>
    <framequat objtype="body" objname="cube"/>
  </sensor>
</mujoco>
"""


def _artic_mjcf(cfg: CubePushCfg) -> str:
    """Articulated arm: base yaw plus shoulder/elbow/wrist pitch in the yawed vertical plane. The wrist pitch is
    solved to keep the rod vertical, so the tip behaves like the other pushers (same EE convention, same contact
    geometry); links ride high on the elbow-up branch."""
    h, r = cfg.cube_half, cfg.pusher_radius
    tx, ty = cfg.table_half
    bx, by = cfg.artic_base
    sz, l1, l2 = cfg.artic_shoulder_z, cfg.artic_l1, cfg.artic_l2
    kp0, kp1, kp2, kp3 = cfg.artic_kp
    kv0, kv1, kv2, kv3 = cfg.artic_kv
    return f"""
<mujoco model="cube_push_artic">
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
    <body name="arm_base" pos="{bx} {by} 0">
      <geom name="column" type="cylinder" size="0.04 0.05" pos="0 0 0.05" contype="0" conaffinity="0"
            rgba=".3 .3 .35 1"/>
      <body name="turret" pos="0 0 {sz}">
        <joint name="j0" type="hinge" axis="0 0 1" damping="0.2"/>
        <geom name="hub" type="cylinder" size="0.05 0.04" pos="0 0 0" mass="0.2" contype="0" conaffinity="0"
              rgba=".7 .3 .2 1"/>
        <body name="upper" pos="0 0 0">
          <joint name="j1" type="hinge" axis="0 1 0" damping="0.5"/>
          <geom name="link1" type="box" size="{l1 / 2} 0.035 0.03" pos="{l1 / 2} 0 0" mass="0.5" contype="0"
                conaffinity="0" rgba=".7 .3 .2 1"/>
          <body name="fore" pos="{l1} 0 0">
            <joint name="j2" type="hinge" axis="0 1 0" damping="0.5"/>
            <geom name="link2" type="box" size="{l2 / 2} 0.03 0.025" pos="{l2 / 2} 0 0" mass="0.4" contype="0"
                  conaffinity="0" rgba=".8 .4 .25 1"/>
            <body name="wrist" pos="{l2} 0 0">
              <joint name="j3" type="hinge" axis="0 1 0" damping="0.2"/>
              <geom name="rod" type="cylinder" size="{r} 0.125" pos="0 0 -0.125" mass="0.1" priority="1"
                    friction="0.3 0.005 0.0001" rgba=".25 .25 .3 1"/>
              <site name="tip" pos="0 0 -0.25"/>
            </body>
          </body>
        </body>
      </body>
    </body>
    <body name="cube" pos="0.06 0 {h}">
      <freejoint name="cube"/>
      <geom name="cube" type="box" size="{h} {h} {h}" mass="0.4" friction="0.5 0.005 0.0001" rgba=".9 .45 .1 1"/>
    </body>
  </worldbody>
  <actuator>
    <position name="a0" joint="j0" kp="{kp0}" kv="{kv0}" ctrllimited="false"/>
    <position name="a1" joint="j1" kp="{kp1}" kv="{kv1}" ctrllimited="false"/>
    <position name="a2" joint="j2" kp="{kp2}" kv="{kv2}" ctrllimited="false"/>
    <position name="a3" joint="j3" kp="{kp3}" kv="{kv3}" ctrllimited="false"/>
  </actuator>
  <sensor>
    <force name="tipf" site="tip"/>
    <jointpos joint="j0"/><jointpos joint="j1"/><jointpos joint="j2"/><jointpos joint="j3"/>
    <framepos objtype="body" objname="cube"/>
    <framequat objtype="body" objname="cube"/>
  </sensor>
</mujoco>
"""


def arm_fk(cfg: CubePushCfg, q: np.ndarray) -> np.ndarray:
    """End-effector position ``[..., 3]`` from arm joints ``[..., (j0, j1, j2)]``.

    The effector point sits 3 cm above the physical rod tip (the rod's bottom face), so it coincides with the
    gantry pusher's centre convention: at ``z_low`` the rod bottom rests 3 mm above the table in both models.
    """
    q = np.asarray(q, dtype=np.float64)
    a0, a1 = q[..., 0], q[..., 0] + q[..., 1]
    x = cfg.arm_base[0] + cfg.arm_l1 * np.cos(a0) + cfg.arm_l2 * np.cos(a1)
    y = cfg.arm_base[1] + cfg.arm_l1 * np.sin(a0) + cfg.arm_l2 * np.sin(a1)
    return np.stack([x, y, q[..., 2] + 0.03], -1)


def arm_ik(cfg: CubePushCfg, ee: np.ndarray) -> np.ndarray:
    """Arm joints for the end-effector target ``[..., 3]``: analytic 2-link IK with the fixed branch
    ``cfg.arm_branch``.

    A pure function of the target (no warm start), so the environment and every hypothesis expand the same
    end-effector trajectory to identical joint controls. Unreachable targets are clamped to the annulus.
    """
    ee = np.asarray(ee, dtype=np.float64)
    dx, dy = ee[..., 0] - cfg.arm_base[0], ee[..., 1] - cfg.arm_base[1]
    r = np.hypot(dx, dy)
    rc = np.clip(r, cfg.arm_min_reach, cfg.arm_max_reach)
    scale = np.where(r > 1e-12, rc / np.maximum(r, 1e-12), 0.0)
    dx, dy = dx * scale, dy * scale
    d = np.clip((rc * rc - cfg.arm_l1**2 - cfg.arm_l2**2) / (2 * cfg.arm_l1 * cfg.arm_l2), -1.0, 1.0)
    j1 = np.arctan2(cfg.arm_branch * np.sqrt(np.maximum(0.0, 1.0 - d * d)), d)
    j0 = np.arctan2(dy, dx) - np.arctan2(cfg.arm_l2 * np.sin(j1), cfg.arm_l1 + cfg.arm_l2 * np.cos(j1))
    j2 = np.clip(ee[..., 2] - 0.03, 0.0, 0.18)
    return np.stack([j0, j1, j2], -1)


def polar_fk(cfg: CubePushCfg, q: np.ndarray) -> np.ndarray:
    """End-effector from cylindrical joints ``[..., (yaw, radius, height)]`` (same EE convention as the SCARA)."""
    q = np.asarray(q, dtype=np.float64)
    x = cfg.polar_base[0] + q[..., 1] * np.cos(q[..., 0])
    y = cfg.polar_base[1] + q[..., 1] * np.sin(q[..., 0])
    return np.stack([x, y, q[..., 2] + EE_OFF], -1)


def polar_ik(cfg: CubePushCfg, ee: np.ndarray) -> np.ndarray:
    """Cylindrical joints for the target: yaw aims the turret, slides set radius and height. Pure in the target."""
    ee = np.asarray(ee, dtype=np.float64)
    dx, dy = ee[..., 0] - cfg.polar_base[0], ee[..., 1] - cfg.polar_base[1]
    r = np.clip(np.hypot(dx, dy), cfg.polar_min_reach, cfg.polar_max_reach)
    return np.stack([np.arctan2(dy, dx), r, np.clip(ee[..., 2] - EE_OFF, 0.0, 0.18)], -1)


def artic_fk(cfg: CubePushCfg, q: np.ndarray) -> np.ndarray:
    """End-effector from ``[..., (yaw, shoulder, elbow, wrist)]``. Pitch is measured from the horizontal, positive
    down; the wrist is solved (see :func:`artic_ik`) so the rod hangs vertical and the EE sits 3 cm above its tip."""
    q = np.asarray(q, dtype=np.float64)
    a1, a2 = q[..., 1], q[..., 1] + q[..., 2]
    rr = cfg.artic_l1 * np.cos(a1) + cfg.artic_l2 * np.cos(a2)
    wz = cfg.artic_shoulder_z - cfg.artic_l1 * np.sin(a1) - cfg.artic_l2 * np.sin(a2)
    x = cfg.artic_base[0] + rr * np.cos(q[..., 0])
    y = cfg.artic_base[1] + rr * np.sin(q[..., 0])
    return np.stack([x, y, wz - cfg.artic_rod + EE_OFF], -1)


def artic_ik(cfg: CubePushCfg, ee: np.ndarray) -> np.ndarray:
    """Yaw aims the vertical plane; planar 2R IK places the wrist above the target (elbow-up branch
    ``cfg.artic_branch``); wrist keeps the rod vertical. Pure in the target."""
    ee = np.asarray(ee, dtype=np.float64)
    dx, dy = ee[..., 0] - cfg.artic_base[0], ee[..., 1] - cfg.artic_base[1]
    r = np.hypot(dx, dy)
    wz = ee[..., 2] - EE_OFF + cfg.artic_rod
    dr = r
    dz = cfg.artic_shoulder_z - wz  # planar target below the shoulder, positive down
    dist = np.clip(np.hypot(dr, dz), cfg.artic_min_reach, cfg.artic_max_reach)
    scale = dist / np.maximum(np.hypot(dr, dz), 1e-12)
    dr, dz = dr * scale, dz * scale
    d = np.clip((dist * dist - cfg.artic_l1**2 - cfg.artic_l2**2) / (2 * cfg.artic_l1 * cfg.artic_l2), -1.0, 1.0)
    j2 = np.arctan2(cfg.artic_branch * np.sqrt(np.maximum(0.0, 1.0 - d * d)), d)
    j1 = np.arctan2(dz, dr) - np.arctan2(cfg.artic_l2 * np.sin(j2), cfg.artic_l1 + cfg.artic_l2 * np.cos(j2))
    j3 = -j1 - j2  # the rod extends along the wrist body's -z: identity wrist keeps it vertical
    return np.stack([np.arctan2(dy, dx), j1, j2, j3], -1)


#: end-effector point above the physical rod tip (m); matches the gantry pusher-centre convention.
EE_OFF = 0.03

#: arm joints per low level (gantry slides count as 3).
N_JOINTS = {"gantry": 3, "arm": 3, "polar": 3, "artic": 4}

#: tip gravity bias (N) seen by the site force sensor, per low level (gantry reads actuator forces: no bias).
TIP_BIAS = {"gantry": 0.0, "arm": 0.981, "polar": 0.981, "artic": 0.981}


def sense(cfg: CubePushCfg, sensordata: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """End-effector position and contact force ``([..., 3], [..., 3])`` from sensordata.

    Gantry: the slides *are* the tip position and the slide actuator forces see contact. Arms: tip from forward
    kinematics of the joint encoders, force from the tip site sensor (contact only, gravity bias subtracted).
    """
    sd = np.asarray(sensordata, dtype=np.float64)
    if cfg.pusher == "gantry":
        return sd[..., PUSHER].copy(), sd[..., FORCE].copy()
    n = N_JOINTS[cfg.pusher]
    force = sd[..., 0:3].copy()
    force[..., 2] -= TIP_BIAS[cfg.pusher]
    return fk(cfg, sd[..., 3 : 3 + n]), force


def fk(cfg: CubePushCfg, q: np.ndarray) -> np.ndarray:
    """End-effector position ``[..., 3]`` from the low-level joints."""
    if cfg.pusher == "artic":
        return artic_fk(cfg, q)
    if cfg.pusher == "polar":
        return polar_fk(cfg, q)
    if cfg.pusher == "arm":
        return arm_fk(cfg, q)
    return np.asarray(q, dtype=np.float64)[..., :3].copy()


def ik(cfg: CubePushCfg, ee: np.ndarray) -> np.ndarray:
    """Low-level joints for the end-effector target. Pure in the target for every embodiment."""
    if cfg.pusher == "artic":
        return artic_ik(cfg, ee)
    if cfg.pusher == "polar":
        return polar_ik(cfg, ee)
    if cfg.pusher == "arm":
        return arm_ik(cfg, ee)
    return np.asarray(ee, dtype=np.float64)[..., :3].copy()


def cube_offsets(cfg: CubePushCfg) -> tuple[slice, slice]:
    """Sensordata slices of the cube position/quaternion (arms carry extra joint sensors before them)."""
    n = 3 if cfg.pusher == "gantry" else N_JOINTS[cfg.pusher]
    return slice(3 + n, 6 + n), slice(6 + n, 10 + n)


def wrap_yaw(yaw):
    """Cube yaw modulo its 90° symmetry, in [-45°, 45°)."""
    return np.mod(np.asarray(yaw) + math.pi / 4, math.pi / 2) - math.pi / 4


def quat_yaw(quat: np.ndarray) -> np.ndarray:
    """Yaw of ``[..., (w, x, y, z)]`` quaternions, wrapped to the cube symmetry."""
    w, x, y, z = (quat[..., i] for i in range(4))
    return wrap_yaw(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def cube_pose(sensordata: np.ndarray, cfg: CubePushCfg | None = None) -> np.ndarray:
    """``[..., (x, y, yaw)]`` of the cube from sensordata (cube slices shift with the low level's joint count)."""
    pos, quat = cube_offsets(cfg or CubePushCfg())
    return np.concatenate([sensordata[..., pos][..., :2], quat_yaw(sensordata[..., quat])[..., None]], -1)


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
        """Tip at ``pusher_xyz`` (control too), cube resting at ``(x, y, yaw)``, everything at rest."""
        mujoco.mj_resetData(self.model, data)
        n = N_JOINTS[self.cfg.pusher]
        q = np.asarray(ik(self.cfg, np.asarray(pusher_xyz, float)), dtype=np.float64).ravel()[:n]
        data.qpos[:n] = q
        data.ctrl[:n] = q
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
    """Per-physics-step controls ``[len(targets) * substeps, n]``: linear interpolation from the previous target
    to each control-step target (n = 3 end-effector axes, or n joints for the arm)."""
    previous = np.asarray(previous, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64).reshape(-1, previous.shape[-1])
    starts = np.concatenate([previous[None], targets[:-1]])
    alpha = (np.arange(1, substeps + 1) / substeps)[None, :, None]
    return (starts[:, None] + alpha * (targets - starts)[:, None]).reshape(-1, targets.shape[-1])


def expand_controls(cfg: CubePushCfg, start_ee: np.ndarray, ee_targets: np.ndarray, substeps: int) -> np.ndarray:
    """Per-physics-step actuator controls for end-effector targets ``[T, 3]`` from the tip start ``[3]``.

    Gantry: interpolate in task space (actuators are slides). Arms: analytic IK per target (a pure function of the
    end-effector waypoints), then interpolate in joint space. Both are deterministic in the waypoints, so the
    environment and every hypothesis execute identical actuator trajectories.
    """
    if cfg.pusher == "gantry":
        return substep_controls(start_ee, ee_targets, substeps)
    qs = [np.asarray(ik(cfg, start_ee), dtype=np.float64).ravel()]
    qs += [np.asarray(ik(cfg, t), dtype=np.float64).ravel() for t in np.reshape(ee_targets, (-1, 3))]
    qs = np.stack(qs)
    return substep_controls(qs[0], qs[1:], substeps)


_DATAS: dict[tuple[int, int, int, int], list[mujoco.MjData]] = {}


def rollout(models: list[mujoco.MjModel], states: np.ndarray, controls: np.ndarray) -> np.ndarray:
    """Batched open-loop rollouts (one model per initial state, multithreaded) of shared controls ``[nstep, nu]`` or
    per-state controls ``[K, nstep, nu]``; returns sensordata at the end of every physics step, ``[K, nstep, nsens]``."""
    import os

    nthread = int(os.environ.get("CUBE_PUSH_THREADS", min(os.cpu_count() or 1, 16)))
    key = (models[0].nq, models[0].nv, models[0].nu, models[0].nsensor)
    pool = _DATAS.setdefault(key, [])
    while len(pool) < nthread:
        pool.append(mujoco.MjData(models[0]))
    controls = controls if controls.ndim == 3 else controls[None]
    _, sensordata = mj_rollout.rollout(models, pool[:nthread], states, controls, persistent_pool=True)
    return sensordata
