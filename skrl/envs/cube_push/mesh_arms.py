"""Real-robot low levels (Franka Panda, UR5e) from MuJoCo Menagerie assets.

Each arm is transplanted into our table scene with a cylindrical pusher tool bolted to the flange and a wrist
camera on the tool. Control is the same hierarchical interface as the analytic arms: the high level commands
end-effector targets, :class:`NumericIK` (damped least squares on tip position + tool-down orientation, fixed
seed and iteration count, hence a pure function of the target) maps them to joints, and joint position servos
track them. Hypotheses expand the same nominal joint trajectory, so env/hypothesis consistency is preserved.

Assets are fetched (not vendored)::

    bash skrl/envs/cube_push/robots/fetch_menagerie.sh
"""

from __future__ import annotations

import copy
import os
import xml.etree.ElementTree as ET

import numpy as np

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "menagerie")

ARMS = {
    "franka": {"dir": "franka_emika_panda", "file": "panda_nohand.xml", "root": "link0", "flange": "attachment",
               "mount": (-0.42, 0.0), "kp": 2000.0, "kv": 80.0,
               "seed": [0.0, -0.4, 0.0, -2.2, 0.0, 1.6, 0.8]},
    "ur5e": {"dir": "universal_robots_ur5e", "file": "ur5e.xml", "root": "base", "flange": "wrist_3_link",
             "mount": (-0.42, 0.0, 0.35), "kp": 1500.0, "kv": 60.0,
             "seed": [0.0, -1.4, 1.6, -1.2, -1.57, 0.0],
             "ik_null": 0.0, "tool_rod_len": 0.15},
}

TOOL_ROD_R = 0.012
TOOL_TIP = -0.31  # tip site offset below the flange frame (rod spans -0.06..-0.31)
TOOL_CAM = "0.045 0 -0.03"


def _parse(name: str) -> ET.ElementTree:
    spec = ARMS[name]
    path = os.path.join(ASSETS, spec["dir"], spec["file"])
    if not os.path.exists(path):
        raise FileNotFoundError(f"menagerie assets missing: run robots/fetch_menagerie.sh ({path})")
    return ET.parse(path)


def transplant(name: str) -> tuple[str, str, ET.Element, list[str]]:
    """Returns ``(default_xml, asset_xml, arm_body_element, joint_names)`` for the arm rooted at its base."""
    spec = ARMS[name]
    root = _parse(name).getroot()
    default = root.find("default")
    asset = root.find("asset")
    default_xml = ET.tostring(default, encoding="unicode") if default is not None else ""
    asset = root.find("asset")
    # strip the wrapping <asset> so the caller can splice inner elements into its own <asset> block
    asset_xml = "".join(ET.tostring(c, encoding="unicode") for c in asset) if asset is not None else ""
    wb = root.find("worldbody")
    arm = next(b for b in wb.findall("body") if b.get("name") == spec["root"])
    arm = copy.deepcopy(arm)
    joint_names = [j.get("name") for j in arm.iter("joint")]
    # Store joint names in spec for later use
    spec["joint_names"] = joint_names
    return default_xml, asset_xml, arm, joint_names


def tool_element(rod_len: float = 0.31) -> ET.Element:
    """Pusher tool + tip site + wrist camera, mounted at the flange frame (identity)."""
    half = rod_len / 2
    tip_off = -rod_len
    tool = ET.Element("body", {"name": "tool", "pos": "0 0 0"})
    ET.SubElement(tool, "geom", {"name": "tool_rod", "type": "cylinder",
                                 "size": f"{TOOL_ROD_R} {half}", "pos": f"0 0 {-half}", "mass": "0.1",
                                 "priority": "1", "friction": "0.3 0.005 0.0001", "rgba": ".25 .25 .3 1"})
    ET.SubElement(tool, "site", {"name": "tip", "pos": f"0 0 {tip_off}"})
    ET.SubElement(tool, "camera", {"name": "wrist", "pos": TOOL_CAM, "fovy": "70"})
    return tool


def meshdir(name: str) -> str:
    return os.path.join(ASSETS, ARMS[name]["dir"], "assets")


class NumericIK:
    """Deterministic DLS IK on tip position + tool-down orientation (fixed seed/iters → pure in the target)."""

    def __init__(self, seed, iters: int = 120, damp: float = 0.2, step: float = 0.5, ori_w: float = 0.3,
                 null: float = 0.02) -> None:
        self.seed = np.asarray(seed, float)
        self.iters, self.damp, self.step, self.ori_w, self.null = iters, damp, step, ori_w, null

    def solve(self, model, data, site_id: int, dofs: np.ndarray, q_min: np.ndarray, q_max: np.ndarray,
              target_tip: np.ndarray, target_zcol: np.ndarray = np.array([0.0, 0.0, 1.0])) -> np.ndarray:
        import mujoco

        mujoco.mj_forward(model, data)
        q = self.seed.copy()
        n = len(dofs)
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        for _ in range(self.iters):
            data.qpos[:n] = q
            mujoco.mj_forward(model, data)
            p = data.site_xpos[site_id]
            zcol = data.site_xmat[site_id].reshape(3, 3)[:, 2]
            e = np.concatenate([target_tip - p, self.ori_w * (target_zcol - zcol)])
            mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
            skew = np.array([[0.0, -zcol[2], zcol[1]], [zcol[2], 0.0, -zcol[0]], [-zcol[1], zcol[0], 0.0]])
            j_ori = -skew @ jacr[:, dofs]
            J = np.concatenate([jacp[:, dofs], self.ori_w * j_ori], 0)
            dq = J.T @ np.linalg.solve(J @ J.T + self.damp**2 * np.eye(6), e)
            q = q + self.step * dq - self.null * (q - self.seed)
            q = np.clip(q, q_min, q_max)
        return q
