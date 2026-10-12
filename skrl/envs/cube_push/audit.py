"""Audit Before You Commit (Abouagour & Min, arXiv 2609.30608) on the partially observable cube push.

Same grid belief, WRT library and execution as :mod:`skrl.envs.cube_push.wafr08`; what changes is how the belief is
used:

* probes follow a fixed schedule instead of an information-gain argmax: guarded touches cycling the faces of w*
  (then the offsets) until the retained set is tight, then ``physics_probes`` force probes, within a budget;
* the retained set C_{k,ν} is the shortest top-weight prefix of cells holding mass ν;
* a commit is the goal push minimizing the worst-case goal cost over C (scenario minimax), with its predicted miss
  rate R̂ = fraction of C whose outcome leaves the tolerance around the nominal outcome;
* every commit is audited with the true state (evaluation only): coverage (true state in C), outcome coverage (true
  outcome inside the box hull of the outcomes predicted over C) and optimism e = R − R̂.

The paper commits once; one push cannot carry the cube to the goal here, so the probe-then-commit cycle repeats until
the goal holds in belief.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

from skrl.envs.cube_push.sim import CubePushCfg
from skrl.envs.cube_push.wafr08 import WAFR08Cfg, WAFR08CubePlanner, WRT


@dataclasses.dataclass
class AuditCfg(WAFR08Cfg):
    nu: float = 0.8  # retained-set mass
    max_retained: int = 400
    tight_position: float = 0.02  # m: C is tight when its weighted RMS distance to w* is below this
    tight_yaw_deg: float = 15.0
    probe_budget: int = 5  # touches per probe phase
    physics_probes: int = 1  # force probes per probe phase, once the pose is tight
    commit_candidates: int = 12


def _yaw_diff(a, b):
    return np.mod(a - b + math.pi / 4, math.pi / 2) - math.pi / 4


class AuditCubePlanner(WAFR08CubePlanner):
    def __init__(self, env_cfg: CubePushCfg | None = None, cfg: AuditCfg | None = None) -> None:
        super().__init__(env_cfg, cfg or AuditCfg())
        self.audits: list[dict] = []
        self._schedule = 0
        self._phase_probes = 0
        self._phase_physics = 0
        self._pending: dict | None = None
        self.true_state = None  # set by the runner each step for the audit (evaluation only)

    def reset(self, goal: np.ndarray) -> None:
        super().reset(goal)
        self.audits, self._schedule, self._phase_probes, self._phase_physics, self._pending = [], 0, 0, 0, None

    def retained_set(self) -> tuple[np.ndarray, np.ndarray]:
        c = self.cfg
        flat, w = self.belief.support(c.nu, c.max_retained)
        return flat, w / w.sum()

    def _tight(self, states: np.ndarray, weights: np.ndarray, w: np.ndarray) -> bool:
        c = self.cfg
        dist2 = np.sum((states[:, :2] - w[:2]) ** 2, axis=1)
        dyaw2 = _yaw_diff(states[:, 2], w[2]) ** 2
        return bool(weights @ dist2 <= c.tight_position**2 and weights @ dyaw2 <= math.radians(c.tight_yaw_deg) ** 2)

    def decide(self) -> tuple[WRT | None, dict]:
        c = self.cfg
        w = self.map_state()
        flat, weights = self.retained_set()
        states = self.belief.values(flat)
        report = {"map": w, "entropy": self.belief.entropy(), "p_goal": self.goal_probability(), "retained": len(flat)}
        if report["p_goal"] >= 1.0 - c.delta:
            report["reason"] = "goal reached in belief"
            return None, report
        tight = self._tight(states, weights, w)
        # probe phase: fixed schedule, not an information-gain argmax
        if not tight and self._phase_probes < c.probe_budget:
            face, offset = self._schedule % 4, c.touch_offsets[(1, 0, 2)[(self._schedule // 4) % 3]]
            self._schedule += 1
            self._phase_probes += 1
            report["reason"] = f"probe {self._phase_probes}/{c.probe_budget}: |C| = {len(flat)} not tight"
            return self.touch(w, face, offset), report
        if tight and self._phase_physics < c.physics_probes:
            goal_dir = self.goal[:2] - w[:2]
            faces = [f for f in range(4) if np.dot(self._frame(w, f, 0.0, 0.0)[1], goal_dir) >= 0]
            self._phase_physics += 1
            report["reason"] = f"physics probe {self._phase_physics}/{c.physics_probes}: C tight in pose"
            return self.probe(w, faces[0]), report
        # commit: scenario minimax over the retained set
        cands = self.goal_candidates(w)[: c.commit_candidates]
        if not cands:
            self._phase_probes = 0
            report["reason"] = "no goal push from w*: probe again"
            return self.touch(w, self._schedule % 4, 0.0), report
        tol_p, tol_y = self.env_cfg.goal_tolerance
        best = None
        for wrt, nominal, cost in cands:
            pred = self.predict(wrt, states)
            outcome_cost = self._goal_cost(pred.next_pose)
            outcome_cost = np.where(pred.blocked | pred.fail, np.inf, outcome_cost)
            worst = float(outcome_cost.max())
            scale = 1.0 if cost <= 1.0 else c.waypoint_scale
            miss = (np.linalg.norm(pred.next_pose[:, :2] - nominal[:2], axis=1) > scale * tol_p) | (
                np.abs(_yaw_diff(pred.next_pose[:, 2], nominal[2])) > scale * tol_y) | pred.blocked | pred.fail
            if best is None or worst < best[1]:
                best = (wrt, worst, float(miss.mean()), nominal, pred.next_pose, scale)
        wrt, worst, r_hat, nominal, outcomes, scale = best
        self._pending = {"wrt": wrt.name, "worst_cost": worst, "r_hat": r_hat, "nominal": nominal, "scale": scale,
                         "lo": outcomes.min(0), "hi": outcomes.max(0), "retained": len(flat),
                         "true_in_C": self._true_in(flat)}
        self._phase_probes, self._phase_physics = 0, 0
        cov = self._pending["true_in_C"]
        report["reason"] = (f"commit: minimax cost {worst:.1f} over |C| = {len(flat)}, R^ {r_hat:.2f}"
                            + ("" if cov is None else f", truth in C: {cov}"))
        return wrt, report

    def _true_in(self, flat: np.ndarray) -> bool | None:
        if self.true_state is None:
            return None
        return bool(np.isin(self.belief.nearest(self.true_state[None])[0], flat))

    def audit_outcome(self, true_pose: np.ndarray) -> dict | None:
        """Close the pending commit's audit with the realized cube pose (evaluation only)."""
        if self._pending is None:
            return None
        a, self._pending = self._pending, None
        tol_p, tol_y = self.env_cfg.goal_tolerance
        s = a.pop("scale")
        nominal, lo, hi = a.pop("nominal"), a.pop("lo"), a.pop("hi")
        missed = bool(np.linalg.norm(true_pose[:2] - nominal[:2]) > s * tol_p
                      or abs(_yaw_diff(true_pose[2], nominal[2])) > s * tol_y)
        a.update(missed=missed, outcome_in_hull=bool(np.all(true_pose[:2] >= lo[:2] - 1e-3)
                                                     and np.all(true_pose[:2] <= hi[:2] + 1e-3)))
        self.audits.append(a)
        return a

    def audit_summary(self) -> dict:
        if not self.audits:
            return {}
        r = np.mean([a["missed"] for a in self.audits])
        r_hat = np.mean([a["r_hat"] for a in self.audits])
        known = [a["true_in_C"] for a in self.audits if a["true_in_C"] is not None]
        return {"commits": len(self.audits), "coverage": float(np.mean(known)) if known else None,
                "outcome_coverage": float(np.mean([a["outcome_in_hull"] for a in self.audits])),
                "miss_rate": float(r), "predicted_miss_rate": float(r_hat), "optimism": float(r - r_hat)}

    def extra_markers(self) -> list:
        """Retained set C as purple dots (pose cells)."""
        flat, _ = self.retained_set()
        states = self.belief.values(flat)
        poses = np.unique(np.round(states[:, :2], 4), axis=0)[:150]
        return [("sphere", [0.004, 0, 0], [x, y, 0.004], 0.0, [0.6, 0.2, 0.9, 0.6]) for x, y in poses]
