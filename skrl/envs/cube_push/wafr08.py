"""WAFR08 belief-based execution (Hsiao, Lozano-Pérez, Kaelbling 2008) for the partially observable cube push.

Belief: a grid over the cube pose (x, y, yaw modulo 90°) and its physics (μ, m, k_d), :class:`GridBelief`.

World-relative trajectories (WRTs) τ(e) are pusher motions defined in the frame of an estimated cube state e. The
planner always instantiates them at the most likely state w* = argmax b:

* ``touch``: lower the pusher next to a face of w*, move towards it and stop at contact (guarded). Observation:
  contact flag and travel to contact. Mostly informs the pose; treated as independent of the physics.
* ``probe``: push a face of w* slowly (``v1``) then fast (``v2``). Observation: contact, travel to contact, mean
  force at ``v1`` and at ``v2`` (≈ μmg + k_d v) and the force peak when accelerating (≈ m a): informs μ, m, k_d.
* ``goal``: a push of a face of w* (offset, speed, length) chosen so the simulated outcome from w* is closest to
  the goal pose. Its outcome depends on the physics too (the cube coasts after fast pushes).

Every WRT is simulated in MuJoCo from each hypothesis of the belief support (batched ``mujoco.rollout``), which
gives the expected observation Ω_τ(w, e) and the next state of every hypothesis; WAFR08 tabulates Ω offline, here
it is simulated on demand. Decision rule (WAFR08 §4):

1. stop if P_b(cube within goal tolerance) ≥ 1 − δ;
2. execute the goal WRT if b is in its belief pre-image: P_b(w ∈ G_δ(w*)) ≥ 1 − δ, where G_δ(w*) holds the states
   whose outcome under τ_goal(w*) is within tolerance of the outcome predicted for w*;
3. otherwise execute the information-gathering WRT maximizing IG(τ, b) − c · FP(τ, b), with FP the probability
   that τ fails (the pusher lands on the cube, or the cube leaves the table).
"""
from __future__ import annotations

import dataclasses
import math
from typing import Callable

import numpy as np

from skrl.belief.grid_belief import GridBelief, entropy, expected_information_gain
from skrl.envs.cube_push.sim import CubePushCfg, CubePushModel, cube_pose, expand_controls, rollout, sense

FACES = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])  # outward face normals in the cube frame


@dataclasses.dataclass
class WAFR08Cfg:
    grid_step: float = 0.015
    yaw_step_deg: float = 10.0
    x_range: tuple[float, float] = (-0.105, 0.42)
    y_range: tuple[float, float] = (-0.255, 0.255)
    mu_levels: tuple[float, ...] = (0.383, 0.55, 0.717)
    mass_levels: tuple[float, ...] = (0.267, 0.4, 0.533)
    damping_levels: tuple[float, ...] = (0.5, 1.5, 2.5)
    ig_support: int = 400  # hypotheses used to score a WRT
    update_support: int = 6000  # hypotheses simulated for the belief update
    support_mass: float = 0.999
    clearance: float = 0.04  # pusher start distance from the face of w*
    touch_depth: float = 0.04  # touch travel beyond the face of w*
    touch_speed: float = 0.1
    touch_offsets: tuple[float, ...] = (-0.5, 0.0, 0.5)  # along the face, in cube half sizes
    probe_speeds: tuple[float, float] = (0.05, 0.2)
    probe_lengths: tuple[float, float] = (0.015, 0.015)  # pushed distance at v1 (after the face), then at v2
    goal_offsets: tuple[float, ...] = (-0.6, 0.0, 0.6)
    goal_speeds: tuple[float, ...] = (0.05, 0.25)
    goal_lengths: tuple[float, ...] = (0.01, 0.02, 0.04, 0.07, 0.1)
    contact_threshold: float = 0.4  # N along the push direction
    blocked_height: float = 0.01  # pusher stuck this high above z_low after lowering: it landed on the cube
    contact_error: float = 0.05  # P(observed contact flag differs from the predicted one)
    sigma_travel: float = 0.008  # m; grid resolution dominates
    sigma_force: tuple[float, float] = (0.15, 0.15)  # σ = a + b |F|
    fp_cost: float = 2.0  # nats per unit failure probability
    delta: float = 0.2
    waypoint_scale: float = 2.0  # G_δ tolerance multiplier for pushes whose predicted outcome is not yet the goal
    probe_entropy: float = 1.5  # nats; probes (which push the cube) only once the pose belief is this concentrated
    min_gain: float = 0.15  # nats; below this no sensing WRT is worth it and the best goal push is executed
    max_wrts: int = 40


@dataclasses.dataclass
class WRT:
    """A world-relative trajectory instantiated at an estimated state: pusher targets from ``start`` (lowered)."""

    kind: str  # touch | probe | goal
    name: str
    start: np.ndarray  # [3] lowered start position
    direction: np.ndarray  # [2] push direction
    targets: np.ndarray  # [T, 3] control-step targets after the start
    guarded: bool
    segments: dict  # name -> slice of control steps (feature windows)


@dataclasses.dataclass
class Prediction:
    """Simulated outcome of a WRT for each hypothesis."""

    obs: np.ndarray  # [K, n] observation features (column 0: 0 no contact, 1 contact, 2 blocked)
    next_pose: np.ndarray  # [K, 3]
    fail: np.ndarray  # [K] cube off the table
    blocked: np.ndarray  # [K] pusher lands on the cube


class WAFR08CubePlanner:
    def __init__(self, env_cfg: CubePushCfg | None = None, cfg: WAFR08Cfg | None = None) -> None:
        self.env_cfg = env_cfg or CubePushCfg()
        self.cfg = cfg or WAFR08Cfg()
        self.sim = CubePushModel(self.env_cfg)
        c = self.cfg
        yaw = np.deg2rad(np.arange(-45.0 + c.yaw_step_deg / 2, 45.0, c.yaw_step_deg))
        axes = [
            np.arange(c.x_range[0], c.x_range[1] + 1e-9, c.grid_step),
            np.arange(c.y_range[0], c.y_range[1] + 1e-9, c.grid_step),
            yaw,
            np.array(c.mu_levels),
            np.array(c.mass_levels),
            np.array(c.damping_levels),
        ]
        self.belief = GridBelief(axes, moving=(0, 1, 2), periods={2: math.pi / 2})
        self.goal = np.zeros(3)
        self.history: list[dict] = []

    # --- belief ---------------------------------------------------------------------------------------------------

    def reset(self, goal: np.ndarray) -> None:
        """Uniform prior over the cube's initial region (any yaw) and the physics levels."""
        e, b = self.env_cfg, self.belief
        x, y = b.axes[0], b.axes[1]
        half = self.cfg.grid_step / 2
        mx = (x >= e.cube_x[0] - half) & (x <= e.cube_x[1] + half)
        my = (y >= e.cube_y[0] - half) & (y <= e.cube_y[1] + half)
        b.set(np.ones(b.shape) * mx[:, None, None, None, None, None] * my[None, :, None, None, None, None])
        self.goal = np.asarray(goal, dtype=np.float64)
        self.history = []

    def map_state(self) -> np.ndarray:
        return self.belief.values(np.array([self.belief.map_cell()]))[0]

    def goal_probability(self) -> float:
        """P_b(cube within the goal tolerance)."""
        b = self.belief
        x, y, yaw = np.meshgrid(*b.axes[:3], indexing="ij")
        dist = np.hypot(x - self.goal[0], y - self.goal[1])
        dyaw = np.abs(np.mod(yaw - self.goal[2] + math.pi / 4, math.pi / 2) - math.pi / 4)
        tol_p, tol_y = self.env_cfg.goal_tolerance
        inside = (dist <= tol_p) & (dyaw <= tol_y + math.radians(self.cfg.yaw_step_deg) / 2)
        return float(b.marginal((0, 1, 2))[inside].sum())

    # --- world-relative trajectories ------------------------------------------------------------------------------

    def _frame(self, e: np.ndarray, face: int, offset: float, distance: float) -> tuple[np.ndarray, np.ndarray]:
        """Start point at ``distance`` from face ``face`` of the cube at e (``offset`` along the face) and the push
        direction (into the face)."""
        c, s = math.cos(e[2]), math.sin(e[2])
        rot = np.array([[c, -s], [s, c]])
        n = rot @ FACES[face]
        t = rot @ np.array([-FACES[face][1], FACES[face][0]])
        h, r = self.env_cfg.cube_half, self.env_cfg.pusher_radius
        start = e[:2] + n * (h + r + distance) + t * offset * h
        return start, -n

    def _profile(self, start2: np.ndarray, d: np.ndarray, segments: list[tuple[float, float]], tail: bool):
        """Targets along d for ``(speed, distance)`` segments (3-step ramp at the start), then, with ``tail``, a hold,
        a 1 cm retreat, a lift and a settling wait."""
        e, dt = self.env_cfg, self.env_cfg.control_dt
        pos, out, windows, k = 0.0, [], [], 0
        for i, (v, length) in enumerate(segments):
            n = max(1, int(round(length / (v * dt))))
            first = len(out)
            for j in range(n):
                ramp = min(1.0, (j + 1) / 3) if i == 0 else 1.0
                pos += v * dt * ramp
                out.append(pos)
            windows.append(slice(first, len(out)))
        xy = start2[None] + np.array(out)[:, None] * d[None]
        targets = np.concatenate([xy, np.full((len(xy), 1), e.z_low)], axis=1)
        if tail:
            last = targets[-1]
            hold = np.repeat(last[None], 5, axis=0)
            back = last + np.r_[-0.01 * d, 0.0]
            retreat = last + np.linspace(0, 1, 6)[1:, None] * (back - last)
            lift = back + np.linspace(0, 1, 11)[1:, None] * np.array([0.0, 0.0, e.z_high - e.z_low])
            settle = np.repeat(lift[-1][None], 20, axis=0)
            targets = np.concatenate([targets, hold, retreat, lift, settle])
        return targets, windows

    def touch(self, e: np.ndarray, face: int, offset: float) -> WRT:
        c = self.cfg
        start, d = self._frame(e, face, offset, c.clearance)
        targets, (w,) = self._profile(start, d, [(c.touch_speed, c.clearance + c.touch_depth)], tail=False)
        return WRT("touch", f"touch face {face} offset {offset:+.1f}", np.r_[start, self.env_cfg.z_low], d,
                   targets, True, {"push": w})

    def probe(self, e: np.ndarray, face: int) -> WRT:
        c = self.cfg
        start, d = self._frame(e, face, 0.0, c.clearance)
        (v1, v2), (l1, l2) = c.probe_speeds, c.probe_lengths
        targets, (w1, w2) = self._profile(start, d, [(v1, c.clearance + l1), (v2, l2)], tail=True)
        return WRT("probe", f"probe face {face}", np.r_[start, self.env_cfg.z_low], d, targets, False,
                   {"push": slice(w1.start, w2.stop), "v1": w1, "v2": w2})

    def goal_push(self, e: np.ndarray, face: int, offset: float, speed: float, length: float) -> WRT:
        c = self.cfg
        start, d = self._frame(e, face, offset, c.clearance * 0.5)
        targets, (w,) = self._profile(start, d, [(speed, c.clearance * 0.5 + length)], tail=True)
        return WRT("goal", f"goal push face {face} off {offset:+.1f} v {speed:.2f} L {100 * length:.0f}cm",
                   np.r_[start, self.env_cfg.z_low], d, targets, False, {"push": w, "v1": w})

    # --- observations ---------------------------------------------------------------------------------------------

    def features(self, wrt: WRT, force: np.ndarray, pusher: np.ndarray) -> np.ndarray:
        """Observation features from per-control-step force and pusher traces of the WRT (``[T, 3]`` each)."""
        c = self.cfg
        f = force[:, :2] @ wrt.direction
        push = wrt.segments["push"]
        idx = np.arange(push.start, min(push.stop, len(f)))
        hit = idx[(idx >= push.start + 3) & (f[idx] > c.contact_threshold)]
        contact = len(hit) > 0
        k = hit[0] if contact else min(push.stop, len(f)) - 1
        travel = float((pusher[k, :2] - wrt.start[:2]) @ wrt.direction)
        if wrt.kind == "touch":
            return np.array([float(contact), travel])

        def mean_after(w: slice, start: int) -> float:
            sel = np.arange(max(w.start, start), w.stop)
            sel = sel[sel >= w.start + (w.stop - w.start) // 2] if len(sel) > 4 else sel
            return float(f[sel].mean()) if contact and len(sel) else 0.0

        f1 = mean_after(wrt.segments["v1"], k + 3)
        if wrt.kind == "goal":
            return np.array([float(contact), travel, f1])
        w2 = wrt.segments["v2"]
        f2 = mean_after(w2, k + 3)
        peak = float(f[w2.start : w2.start + 4].max()) if contact else 0.0
        return np.array([float(contact), travel, f1, f2, peak])

    def log_likelihood(self, observed: np.ndarray, predicted: np.ndarray) -> np.ndarray:
        """``log P(observed | w)`` for predicted features ``[K, n]`` (``observed`` ``[n]`` or ``[M, n]`` → ``[K, M]``)."""
        c = self.cfg
        obs = np.atleast_2d(observed)[None]  # [1, M, n]
        pred = predicted[:, None]  # [K, 1, n]
        same = obs[..., 0] == pred[..., 0]
        ll = np.where(same, math.log(1.0 - c.contact_error), math.log(c.contact_error))
        both = same & (pred[..., 0] == 1.0)
        z = (obs[..., 1] - pred[..., 1]) / c.sigma_travel
        cont = -0.5 * z**2
        if pred.shape[-1] > 2:
            sigma = c.sigma_force[0] + c.sigma_force[1] * np.abs(pred[..., 2:])
            cont = cont - (0.5 * ((obs[..., 2:] - pred[..., 2:]) / sigma) ** 2 + np.log(sigma)).sum(-1)
        ll = ll + np.where(both, cont, 0.0)
        return ll if np.ndim(observed) > 1 else ll[:, 0]

    # --- simulation of hypotheses ---------------------------------------------------------------------------------

    def predict(self, wrt: WRT, states: np.ndarray, physics_free: bool | None = None) -> Prediction:
        """Simulate the WRT from each hypothesis ``[K, 6]`` (pose + physics). Touches use the middle physics levels
        for every hypothesis (``physics_free``): a guarded touch barely depends on them."""
        e = self.env_cfg
        physics_free = wrt.kind == "touch" if physics_free is None else physics_free
        k = len(states)
        rel = states[:, :2] - wrt.start[None, :2]
        c, s = np.cos(states[:, 2]), np.sin(states[:, 2])
        u, v = np.abs(c * rel[:, 0] + s * rel[:, 1]), np.abs(-s * rel[:, 0] + c * rel[:, 1])
        outside = np.hypot(np.maximum(u - e.cube_half, 0), np.maximum(v - e.cube_half, 0))
        blocked = outside < e.pusher_radius + 0.003
        n_obs = {"touch": 2, "probe": 5, "goal": 3}[wrt.kind]
        obs = np.zeros((k, n_obs))
        obs[blocked, 0] = 2.0
        next_pose = states[:, :3].copy()
        free = np.flatnonzero(~blocked)
        if len(free):
            if physics_free:
                mid = [self.belief.axes[d][len(self.belief.axes[d]) // 2] for d in (3, 4, 5)]
                models = [self.sim.variant(*mid)] * len(free)
            else:
                models = [self.sim.variant(*p) for p in states[free, 3:]]
            init = self.sim.initial_states(wrt.start, states[free, :3])
            sd = rollout(models, init, expand_controls(e, wrt.start, wrt.targets, e.substeps))
            steps = sd[:, e.substeps - 1 :: e.substeps]  # end of every control step
            for j, i in enumerate(free):
                tip, force = sense(e, steps[j])
                obs[i] = self.features(wrt, force, tip)
                if wrt.guarded and obs[i, 0] == 1.0:
                    hit = self._contact_step(wrt, force)
                    next_pose[i] = cube_pose(steps[j, hit], e)
                else:
                    next_pose[i] = cube_pose(steps[j, -1], e)
        fail = np.any(np.abs(next_pose[:, :2]) > np.asarray(e.table_half) - e.cube_half, axis=1)
        return Prediction(obs, next_pose, fail, blocked)

    def _contact_step(self, wrt: WRT, force: np.ndarray) -> int:
        f = force[:, :2] @ wrt.direction
        push = wrt.segments["push"]
        idx = np.arange(push.start + 3, min(push.stop, len(f)))
        hit = idx[f[idx] > self.cfg.contact_threshold]
        return int(hit[0]) if len(hit) else len(f) - 1

    # --- WAFR08 decision rule -------------------------------------------------------------------------------------

    def _goal_cost(self, pose: np.ndarray) -> np.ndarray:
        tol_p, tol_y = self.env_cfg.goal_tolerance
        dist = np.linalg.norm(pose[..., :2] - self.goal[:2], axis=-1)
        dyaw = np.abs(np.mod(pose[..., 2] - self.goal[2] + math.pi / 4, math.pi / 2) - math.pi / 4)
        return dist / tol_p + dyaw / tol_y

    def goal_candidates(self, w: np.ndarray) -> list[tuple[WRT, np.ndarray, float]]:
        """Goal pushes instantiated at w, sorted by the goal cost of their outcome simulated from w."""
        c = self.cfg
        wrts = [self.goal_push(w, f, o, v, length) for f in range(4) for o in c.goal_offsets
                for v in c.goal_speeds for length in c.goal_lengths]
        preds = self._predict_many(wrts, w)
        ranked = []
        for wrt, pred in zip(wrts, preds):
            if pred.blocked[0] or pred.fail[0]:
                continue
            ranked.append((wrt, pred.next_pose[0], float(self._goal_cost(pred.next_pose[0]))))
        return sorted(ranked, key=lambda x: x[2])

    def _predict_many(self, wrts: list[WRT], w: np.ndarray) -> list[Prediction]:
        """Rollouts of many WRTs from the single state w (padded to a common length, holding the last target)."""
        e = self.env_cfg
        length = max(len(x.targets) for x in wrts)
        controls = np.stack([
            expand_controls(e, x.start, np.concatenate([x.targets, np.repeat(x.targets[-1:], length - len(x.targets), 0)]),
                            e.substeps)
            for x in wrts
        ])
        model = self.sim.variant(*w[3:])
        init = np.concatenate([self.sim.initial_states(x.start, w[None, :3]) for x in wrts])
        sd = rollout([model] * len(wrts), init, controls)
        out = []
        for x, s in zip(wrts, sd):
            steps = s[e.substeps - 1 :: e.substeps][: len(x.targets)]
            pose = cube_pose(steps[-1], e)
            fail = bool(np.any(np.abs(pose[:2]) > np.asarray(e.table_half) - e.cube_half))
            tip, force = sense(e, steps)
            obs = self.features(x, force, tip)
            out.append(Prediction(obs[None], pose[None], np.array([fail]), np.array([False])))
        return out

    def decide(self) -> tuple[WRT | None, dict]:
        """WAFR08 rule. Returns ``(None, report)`` when the goal is reached in belief."""
        c, b = self.cfg, self.belief
        w = self.map_state()
        report = {"map": w, "entropy": b.entropy(), "p_goal": self.goal_probability()}
        if report["p_goal"] >= 1.0 - c.delta:
            report["reason"] = "goal reached in belief"
            return None, report
        flat, weights = b.support(c.support_mass, c.ig_support)
        weights = weights / weights.sum()
        states = b.values(flat)
        cost_now = float(self._goal_cost(w[:3]))
        # 1. goal-seeking: the best pushes from w*, executed if b is in their belief pre-image
        report["p_preimage"] = 0.0
        candidates = [x for x in self.goal_candidates(w) if x[2] < cost_now - 0.1]
        tol_p, tol_y = self.env_cfg.goal_tolerance
        for wrt, outcome, cost in candidates[:3]:
            pred = self.predict(wrt, states)
            dist = np.linalg.norm(pred.next_pose[:, :2] - outcome[:2], axis=1)
            dyaw = np.abs(np.mod(pred.next_pose[:, 2] - outcome[2] + math.pi / 4, math.pi / 2) - math.pi / 4)
            scale = 1.0 if self._goal_cost(outcome) <= 1.0 else c.waypoint_scale  # final push vs waypoint push
            in_g = (dist <= scale * tol_p) & (dyaw <= scale * tol_y) & ~pred.blocked & ~pred.fail
            p_pre = float(weights[in_g].sum())
            report["p_preimage"] = max(report["p_preimage"], p_pre)
            if p_pre >= 1.0 - c.delta:
                report.update(reason=f"b in BPre (P = {p_pre:.2f}), predicted cost {cost:.2f}", score=p_pre)
                return wrt, report
        # 2. information gathering: argmax IG - c FP over touches and probes at w*
        goal_dir = self.goal[:2] - w[:2]
        options = [self.touch(w, f, o) for f in range(4) for o in c.touch_offsets]
        # probes push the cube: only once the pose is localized (their force windows then see the contact), and only
        # towards the goal
        report["pose_entropy"] = float(entropy(b.marginal((0, 1, 2))))
        if report["pose_entropy"] < c.probe_entropy:
            options += [self.probe(w, f) for f in range(4) if np.dot(self._frame(w, f, 0.0, 0.0)[1], goal_dir) >= 0]
        best, best_score, scores = None, -np.inf, {}
        for wrt in options:
            pred = self.predict(wrt, states)
            ll = self.log_likelihood(pred.obs, pred.obs)
            next_values = states.copy()
            next_values[:, :3] = pred.next_pose
            ig = expected_information_gain(weights, ll, b.nearest(next_values))
            fp = float(weights[pred.blocked | pred.fail].sum())
            score = ig - c.fp_cost * fp
            scores[wrt.name] = (ig, fp)
            if score > best_score:
                best, best_score = wrt, score
        report["scores"] = scores
        if best_score < c.min_gain and candidates:  # nothing left to learn: push anyway (outside WAFR08's rule)
            report.update(reason="no informative WRT, best goal push", score=best_score)
            return candidates[0][0], report
        ig, fp = scores[best.name]
        report.update(reason=f"info gathering: IG {ig:.2f} nats, FP {fp:.2f}", score=best_score)
        return best, report

    def update(self, wrt: WRT, observed: np.ndarray) -> float:
        """Bayes update of the belief with the observed features of an executed WRT."""
        c, b = self.cfg, self.belief
        flat, _ = b.support(c.support_mass, c.update_support if wrt.kind != "touch" else None)
        states = b.values(flat)
        if wrt.kind == "touch":
            # physics-free: simulate each distinct pose once
            poses, inverse = np.unique(flat // np.prod(b.shape[3:]), return_inverse=True)
            pose_states = states[np.unique(inverse, return_index=True)[1]]
            pred = self.predict(wrt, pose_states)
            pred = Prediction(pred.obs[inverse], pred.next_pose[inverse], pred.fail[inverse], pred.blocked[inverse])
        else:
            pred = self.predict(wrt, states)
        ll = self.log_likelihood(observed, pred.obs)
        next_values = states.copy()
        next_values[:, :3] = pred.next_pose
        evidence = b.update(flat, ll, next_values, keep=~pred.fail)
        if wrt.kind != "touch":
            b.blur()
        return evidence

    # --- execution in the environment -----------------------------------------------------------------------------

    def execute(self, env, wrt: WRT, on_step: Callable[[str], None] | None = None) -> np.ndarray:
        """Run the WRT on the environment (transit above the cube, guarded lowering, the trajectory, guarded stop
        for touches) and return the observed features."""
        e = env.cfg
        on_step = on_step or (lambda phase: None)

        def move(to, phase, speed=e.max_speed * 0.9):
            limit = speed * e.control_dt
            obs = None
            while np.max(np.abs(np.asarray(to) - env.target)) > 1e-9:
                target = env.target + np.clip(np.asarray(to) - env.target, -limit, limit)
                obs, *_ = env.step(target)
                on_step(phase)
            return obs

        here = env.target
        if here[2] < e.z_high - 1e-6:
            move([here[0], here[1], e.z_high], "lift")
        move([wrt.start[0], wrt.start[1], e.z_high], "transit")
        move(wrt.start, "lower", speed=0.2)
        for _ in range(3):
            obs, *_ = env.step(wrt.start)
            on_step("settle")
        if obs[2] > e.z_low + self.cfg.blocked_height:  # the pusher rests on the cube
            move([*env.target[:2], e.z_high], "lift")
            return np.r_[2.0, np.zeros({"touch": 1, "probe": 4, "goal": 2}[wrt.kind])]
        forces, pushers = [], []
        for k, target in enumerate(wrt.targets):
            obs, *_ = env.step(target)
            on_step(wrt.kind)
            forces.append(obs[3:6])
            pushers.append(obs[0:3])
            if wrt.guarded and k >= wrt.segments["push"].start + 3 and float(obs[3:5] @ wrt.direction) > \
                    self.cfg.contact_threshold:
                break
        features = self.features(wrt, np.array(forces), np.array(pushers))
        if wrt.guarded:
            back = np.r_[env.target[:2] - 0.01 * wrt.direction, e.z_low]
            move(back, "retreat")
            move([*back[:2], e.z_high], "lift")
        return features
