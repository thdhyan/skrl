"""Quasi-static belief dynamics for robust pushing (Jankowski et al., IJRR 2025) on the partially observable cube push.

The belief is moved through the pushes without any observation: nominal particles are simulated (slow pushes, so
quasi-static), and the variance of the particle set after each push measures how much the push contracts (funnels)
the belief. A plan of K pushes is optimized with CMA-ES (BS-VP-STO-style; the decision variables are per-push
approach angle, lateral offset and length instead of via-points) for

    c = Π_k exp(−(1 − γ_k) / (K − 1)) + λ_c Σ_k max(γ_k − 1, 0) + ‖E[p_K] − p_des‖² / tol_p² + (E[ψ_K] − ψ_des)² / tol_ψ²,
    γ_k = (V̂_{k+1} + Ê_k[η] V_w) / (V̂_k + V_w),

with V̂ the particle variance of the cube position (yaw scaled by the half size), η = 1 for particles in contact
and V_w the contact noise. The paper multiplies the contraction term by 10³ once some γ_k > 1; here a long push of
a cube with unknown friction and mass always spreads the belief a little, so that switch made CMA-ES prefer pushes
that never touch the cube, and the expansion is penalized linearly instead (λ_c = ``expansion_weight``). The plan runs open loop (the belief only follows T). The method assumes a moderately
known start; with this env's 12 × 20 cm, any-yaw prior it first localizes with guarded touches chosen by WAFR08's
IG − c·FP, and after the plan it touches again and replans if the goal is not reached in belief.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

from skrl.belief.grid_belief import entropy, expected_information_gain
from skrl.envs.cube_push.sim import CubePushCfg, cube_pose, rollout, substep_controls
from skrl.envs.cube_push.wafr08 import WAFR08Cfg, WAFR08CubePlanner, WRT


@dataclasses.dataclass
class QuasiStaticCfg(WAFR08Cfg):
    localize_entropy: float = 3.0  # nats of the pose marginal below which the plan is made
    pushes: int = 3  # K
    particles: int = 32
    popsize: int = 16
    iterations: int = 15
    push_speed: float = 0.05  # slow: quasi-static
    max_length: float = 0.12  # per push, beyond the face
    min_length: float = 0.02
    contact_noise: float = 0.003  # m: √V_w
    max_touches: int = 8  # per localization phase
    expansion_weight: float = 10.0  # λ_c per unit of γ_k above 1


def _wrap(yaw):
    return np.mod(yaw + math.pi / 4, math.pi / 2) - math.pi / 4


class QuasiStaticCubePlanner(WAFR08CubePlanner):
    def __init__(self, env_cfg: CubePushCfg | None = None, cfg: QuasiStaticCfg | None = None, seed: int = 0) -> None:
        super().__init__(env_cfg, cfg or QuasiStaticCfg())
        self.rng = np.random.default_rng(seed)
        self.plan: list[WRT] = []
        self.plan_means: list[np.ndarray] = []
        self.plan_info: dict = {}
        self._touches = 0
        self._open_loop = False

    def reset(self, goal: np.ndarray) -> None:
        super().reset(goal)
        self.plan, self.plan_means, self.plan_info, self._touches, self._open_loop = [], [], {}, 0, False

    # --- pushes defined around a cube position ------------------------------------------------------------------

    def push(self, center: np.ndarray, angle: float, offset: float, length: float) -> WRT:
        """Slow push towards ``center`` from approach ``angle`` (world), ``offset`` along the side (cube half sizes),
        ``length`` beyond the face; starts clear of the cube at any yaw."""
        e, c = self.env_cfg, self.cfg
        h, r = e.cube_half, e.pusher_radius
        n = np.array([math.cos(angle), math.sin(angle)])
        t = np.array([-n[1], n[0]])
        reach = h * math.sqrt(2) + r + 0.01
        start = center[:2] + n * reach + t * offset * h
        targets, (w,) = self._profile(start, -n, [(c.push_speed, reach - r - h + length)], tail=True)
        return WRT("goal", f"qs push {math.degrees(angle):4.0f}° off {offset:+.1f} L {100 * length:.0f}cm",
                   np.r_[start, e.z_low], -n, targets, False, {"push": w, "v1": w})

    @staticmethod
    def _decode(x: np.ndarray, k: int) -> tuple[float, float, float]:
        a, o, length = x[3 * k : 3 * k + 3]
        return math.pi * a, float(np.clip(o, -1.0, 1.0)), float(np.clip(length, 0.0, 1.0))

    def _variance(self, poses: np.ndarray) -> float:
        h = self.env_cfg.cube_half
        mean_yaw = math.atan2(np.sin(4 * poses[:, 2]).mean(), np.cos(4 * poses[:, 2]).mean()) / 4
        return float(poses[:, 0].var() + poses[:, 1].var() + (h * _wrap(poses[:, 2] - mean_yaw)).var())

    def _mean(self, poses: np.ndarray) -> np.ndarray:
        yaw = math.atan2(np.sin(4 * poses[:, 2]).mean(), np.cos(4 * poses[:, 2]).mean()) / 4
        return np.array([poses[:, 0].mean(), poses[:, 1].mean(), yaw])

    def evaluate(self, xs: list[np.ndarray], particles: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        """Cost of each candidate plan on the nominal particles ``[N, 6]``: all candidates × particles are rolled out
        together, push by push."""
        c, e = self.cfg, self.env_cfg
        n, p = len(particles), len(xs)
        poses = np.tile(particles[:, :3], (p, 1))  # [P * N, 3]
        models = [self.sim.variant(*s[3:]) for s in particles] * p
        v_w = c.contact_noise**2
        gammas = np.zeros((p, c.pushes))
        bad = np.zeros(p)
        means = [[] for _ in range(p)]
        wrts = [[] for _ in range(p)]
        for k in range(c.pushes):
            stage, controls, init = [], [], []
            for i, x in enumerate(xs):
                angle, off, length = self._decode(x, k)
                center = self._mean(poses[i * n : (i + 1) * n])
                wrt = self.push(center, angle, off, max(length * c.max_length, c.min_length))
                stage.append(wrt)
                wrts[i].append(wrt)
                means[i].append(center)
            length = max(len(w.targets) for w in stage)
            for i, wrt in enumerate(stage):
                padded = np.concatenate([wrt.targets, np.repeat(wrt.targets[-1:], length - len(wrt.targets), 0)])
                controls.append(np.repeat(substep_controls(wrt.start, padded, e.substeps)[None], n, 0))
                init.append(self.sim.initial_states(wrt.start, poses[i * n : (i + 1) * n]))
            blocked = np.concatenate([self._blocked(wrt, poses[i * n : (i + 1) * n]) for i, wrt in enumerate(stage)])
            sd = rollout(models, np.concatenate(init), np.concatenate(controls))
            new = cube_pose(sd[:, -1])
            new[blocked] = poses[blocked]
            contact = np.linalg.norm(new[:, :2] - poses[:, :2], axis=1) > 1e-3
            off_table = np.any(np.abs(new[:, :2]) > np.asarray(e.table_half) - e.cube_half, axis=1)
            for i in range(p):
                s = slice(i * n, (i + 1) * n)
                gammas[i, k] = (self._variance(new[s]) + contact[s].mean() * v_w) / (self._variance(poses[s]) + v_w)
                bad[i] += blocked[s].mean() + off_table[s].mean()
            poses = new
        costs, infos = np.zeros(p), []
        tol_p, tol_y = e.goal_tolerance
        for i in range(p):
            final = self._mean(poses[i * n : (i + 1) * n])
            contraction = float(np.prod(np.exp(-(1.0 - gammas[i]) / max(c.pushes - 1, 1))))
            contraction += c.expansion_weight * float(np.maximum(gammas[i] - 1.0, 0.0).sum())
            dist2 = float(np.sum((final[:2] - self.goal[:2]) ** 2)) / tol_p**2 + (_wrap(final[2] - self.goal[2]) / tol_y) ** 2
            costs[i] = contraction + dist2 + 1e3 * bad[i]
            infos.append({"gammas": gammas[i].tolist(), "final_mean": final, "means": means[i] + [final],
                          "wrts": wrts[i], "final_std": math.sqrt(self._variance(poses[i * n : (i + 1) * n])),
                          "bad": float(bad[i])})
        return costs, infos

    def _blocked(self, wrt: WRT, states: np.ndarray) -> np.ndarray:
        e = self.env_cfg
        rel = states[:, :2] - wrt.start[None, :2]
        c, s = np.cos(states[:, 2]), np.sin(states[:, 2])
        u, v = np.abs(c * rel[:, 0] + s * rel[:, 1]), np.abs(-s * rel[:, 0] + c * rel[:, 1])
        return np.hypot(np.maximum(u - e.cube_half, 0), np.maximum(v - e.cube_half, 0)) < e.pusher_radius + 0.003

    def sample_particles(self) -> np.ndarray:
        c, b = self.cfg, self.belief
        flat, w = b.support(c.support_mass)
        idx = self.rng.choice(flat, size=c.particles, p=w / w.sum())
        s = b.values(idx)
        s[:, 0] += self.rng.uniform(-0.5, 0.5, c.particles) * b.steps[0]
        s[:, 1] += self.rng.uniform(-0.5, 0.5, c.particles) * b.steps[1]
        s[:, 2] = _wrap(s[:, 2] + self.rng.uniform(-0.5, 0.5, c.particles) * b.steps[2])
        return s

    def optimize(self) -> dict:
        """CMA-ES over the K pushes."""
        import cma

        c = self.cfg
        particles = self.sample_particles()
        m = self._mean(particles[:, :3])
        away = math.atan2(m[1] - self.goal[1], m[0] - self.goal[0]) / math.pi  # approach from the side away from goal
        dist = np.linalg.norm(self.goal[:2] - m[:2])
        x0 = np.tile([away, 0.0, min(1.0, dist / (c.pushes * c.max_length))], c.pushes)
        es = cma.CMAEvolutionStrategy(x0, 0.3, {"popsize": c.popsize, "maxiter": c.iterations, "verbose": -9,
                                                 "seed": int(self.rng.integers(1, 2**31 - 1))})
        best = (np.inf, None)
        while not es.stop():
            xs = es.ask()
            costs, infos = self.evaluate(xs, particles)
            es.tell(xs, costs.tolist())
            i = int(np.argmin(costs))
            if costs[i] < best[0]:
                best = (float(costs[i]), infos[i])
        cost, info = best
        info["cost"] = cost
        return info

    # --- decision -------------------------------------------------------------------------------------------------

    def _best_touch(self, w: np.ndarray) -> tuple[WRT, float, float]:
        c, b = self.cfg, self.belief
        flat, weights = b.support(c.support_mass, c.ig_support)
        weights = weights / weights.sum()
        states = b.values(flat)
        best = (None, -np.inf, 0.0)
        for wrt in [self.touch(w, f, o) for f in range(4) for o in c.touch_offsets]:
            pred = self.predict(wrt, states)
            next_values = states.copy()
            next_values[:, :3] = pred.next_pose
            ig = expected_information_gain(weights, self.log_likelihood(pred.obs, pred.obs), b.nearest(next_values))
            score = ig - c.fp_cost * float(weights[pred.blocked | pred.fail].sum())
            if score > best[1]:
                best = (wrt, score, ig)
        return best

    def decide(self) -> tuple[WRT | None, dict]:
        c, b = self.cfg, self.belief
        w = self.map_state()
        h_pose = float(entropy(b.marginal((0, 1, 2))))
        report = {"map": w, "entropy": b.entropy(), "p_goal": self.goal_probability(), "pose_entropy": h_pose}
        if self.plan:
            self._open_loop = True
            k = c.pushes - len(self.plan)
            g = self.plan_info["gammas"][k]
            report["reason"] = f"open loop push {k + 1}/{c.pushes}: planned gamma {g:.2f}, no sensing"
            return self.plan.pop(0), report
        self._open_loop = False
        self.plan_means = []
        if report["p_goal"] >= 1.0 - c.delta:
            report["reason"] = "goal reached in belief"
            return None, report
        if h_pose > c.localize_entropy and self._touches < c.max_touches:
            self._touches += 1
            wrt, score, ig = self._best_touch(w)
            report["reason"] = f"localize: H(pose) {h_pose:.2f} > {c.localize_entropy}, touch IG {ig:.2f}"
            return wrt, report
        self._touches = 0
        info = self.optimize()
        self.plan_info = info
        self.plan = list(info["wrts"])
        self.plan_means = info["means"]
        g = ", ".join(f"{x:.2f}" for x in info["gammas"])
        report["reason"] = f"CMA-ES plan: cost {info['cost']:.1f}, gamma ({g}), final std {100 * info['final_std']:.1f} cm"
        report["plan"] = {k: v for k, v in info.items() if k != "wrts"}
        self._open_loop = True
        return self.plan.pop(0), report

    def update(self, wrt: WRT, observed: np.ndarray) -> float:
        """Open loop: the belief only follows the nominal dynamics T (plus contact noise); sensing touches correct it."""
        if not self._open_loop:
            return super().update(wrt, observed)
        c, b = self.cfg, self.belief
        flat, _ = b.support(c.support_mass, c.update_support)
        states = b.values(flat)
        pred = self.predict(wrt, states)
        next_values = states.copy()
        next_values[:, :3] = pred.next_pose
        b.update(flat, np.zeros(len(flat)), next_values, keep=~pred.fail)
        b.blur()
        return 0.0

    def extra_markers(self) -> list:
        """Planned predicted cube means (cyan) for the remaining open-loop pushes."""
        if not self.plan_means:
            return []
        h = self.env_cfg.cube_half
        done = self.cfg.pushes - len(self.plan)
        return [("box", [0.5 * h, 0.5 * h, 0.002], [m[0], m[1], 0.004], m[2], [0.1, 0.8, 0.9, 0.5])
                for m in self.plan_means[done:]]
