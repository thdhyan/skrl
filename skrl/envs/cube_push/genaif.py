"""GenAIF (Li, Jafari, Matta, arXiv 2610.05692) on the partially observable cube push: a learned generative model
replaces the simulator in the belief update and in planning, and actions minimize the expected free energy.

* Generative model: an MLP trained on MuJoCo rollouts of the WRT library, in the frame of each WRT (cube pose
  relative to the pusher start and push direction, physics, WRT profile) → P(contact), Gaussian (mean, σ) of the
  travel / force features and of the cube displacement. It stands in for the paper's CVAE: one network gives the
  likelihood ℓ_θ(o | w, τ) and the transition T_θ(w' | w, τ).
* Belief: the same grid as WAFR08. The physics levels are the hidden context z and may switch between WRTs with a
  Markov matrix M (stay with probability 1 − ε).
* Decision (MPC, replanned after every WRT): over proposals at w* (touches, probes and the goal pushes the model ranks
  best), drop those with risk P(blocked or off table) > ρ, then take
  ``argmin G = E_b[goal cost of the predicted next pose] − λ IG``, with IG the mutual information between the state
  and the predicted observation (on the next-state belief). The goal-reached stop rule is WAFR08's.
"""
from __future__ import annotations

import dataclasses
import math
import os

import numpy as np
import torch
import torch.nn as nn

from skrl.belief.grid_belief import entropy, expected_information_gain
from skrl.envs.cube_push.sim import FORCE, PUSHER, CubePushCfg, cube_pose, rollout, substep_controls
from skrl.envs.cube_push.wafr08 import WAFR08Cfg, WAFR08CubePlanner, WRT, Prediction

KINDS = ("touch", "probe", "goal")
N_OBS = {"touch": 2, "probe": 5, "goal": 3}


@dataclasses.dataclass
class GenAIFCfg(WAFR08Cfg):
    model_path: str = "runs/cube_push_genaif/model.pt"
    samples_per_profile: int = 5000
    train_steps: int = 8000
    switch_prob: float = 0.02  # ε of the Markov switching of the physics context
    epistemic_weight: float = 2.0  # λ: goal-cost units per nat
    risk: float = 0.2  # ρ
    goal_proposals: int = 6


def _frame(wrt: WRT) -> tuple[np.ndarray, np.ndarray, float]:
    d = wrt.direction
    return d, np.array([-d[1], d[0]]), math.atan2(d[1], d[0])


def _profile(wrt: WRT, planner: WAFR08CubePlanner) -> tuple[float, float]:
    """(speed, pushed length) of the WRT's first segment, which defines its trajectory in its own frame."""
    c = planner.cfg
    if wrt.kind == "touch":
        return c.touch_speed, c.clearance + c.touch_depth
    if wrt.kind == "probe":
        return c.probe_speeds[0], c.clearance + c.probe_lengths[0]
    seg = wrt.segments["push"]
    v = float(np.linalg.norm(wrt.targets[seg.start + 3, :2] - wrt.targets[seg.start + 2, :2]) / planner.env_cfg.control_dt)
    return v, float((wrt.targets[seg.stop - 1, :2] - wrt.start[:2]) @ wrt.direction)


def model_inputs(wrt: WRT, states: np.ndarray, planner: WAFR08CubePlanner) -> np.ndarray:
    """``[K, 12]``: cube position and yaw in the WRT frame, physics, WRT kind and profile."""
    a, l, angle = _frame(wrt)
    rel = states[:, :2] - wrt.start[None, :2]
    yaw = states[:, 2] - angle
    e = planner.env_cfg
    phys = (states[:, 3:6] - np.array([np.mean(e.mu), np.mean(e.mass), np.mean(e.damping)])) / np.array(
        [np.ptp(e.mu), np.ptp(e.mass), np.ptp(e.damping)])
    kind = np.zeros((len(states), 3))
    kind[:, KINDS.index(wrt.kind)] = 1.0
    v, length = _profile(wrt, planner)
    return np.concatenate([rel @ a[:, None] * 10, rel @ l[:, None] * 10, np.sin(4 * yaw)[:, None],
                           np.cos(4 * yaw)[:, None], phys, kind, np.full((len(states), 1), v * 5),
                           np.full((len(states), 1), length * 10)], axis=1)


class GenerativeModel(nn.Module):
    """Inputs → contact logit, means and log-σ of 7 standardized targets (travel, f1, f2, peak, Δa, Δl, Δψ)."""

    def __init__(self, n_in: int = 12, width: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(),
                                 nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 15))
        self.register_buffer("y_mean", torch.zeros(7))
        self.register_buffer("y_std", torch.ones(7))

    def forward(self, x: torch.Tensor):
        out = self.net(x)
        return out[:, 0], out[:, 1:8], out[:, 8:15].clamp(-6, 3)


class GenAIFCubePlanner(WAFR08CubePlanner):
    def __init__(self, env_cfg: CubePushCfg | None = None, cfg: GenAIFCfg | None = None) -> None:
        super().__init__(env_cfg, cfg or GenAIFCfg())
        self.model = GenerativeModel()
        self._sigma = None
        self._p_contact = None
        if os.path.exists(self.cfg.model_path):
            self.model.load_state_dict(torch.load(self.cfg.model_path))
        self.model.eval()

    # --- learning the generative model ----------------------------------------------------------------------------

    def _profiles(self) -> list:
        """WRT constructors ``(e, face, offset) -> WRT``, one per trajectory profile."""
        c = self.cfg
        out = [lambda e, f, o: self.touch(e, f, o), lambda e, f, o: self.probe(e, f)]
        out += [(lambda v, L: lambda e, f, o: self.goal_push(e, f, o, v, L))(v, L) for v in c.goal_speeds
                for L in c.goal_lengths]
        return out

    def dataset(self, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Rollouts of every WRT profile from estimates e and true states w near e: inputs, observation features
        (padded to 5) and the cube displacement in the WRT frame."""
        c, e = self.cfg, self.env_cfg
        rng = np.random.default_rng(seed)
        levels = [np.linspace(*r, 7) for r in (e.mu, e.mass, e.damping)]
        xs, obs, delta = [], [], []
        for make in self._profiles():
            n = c.samples_per_profile
            est = np.stack([rng.uniform(-0.05, 0.25, n), rng.uniform(-0.15, 0.15, n), rng.uniform(-math.pi, math.pi, n)], 1)
            true = est + np.stack([rng.uniform(-0.05, 0.05, n), rng.uniform(-0.05, 0.05, n),
                                   rng.uniform(-math.pi / 4, math.pi / 4, n)], 1)
            phys = np.stack([rng.choice(lv, n) for lv in levels], 1)
            wrts = [make(est[i], rng.integers(4), rng.choice(c.touch_offsets)) for i in range(n)]
            states = np.concatenate([true, phys], 1)
            keep = [i for i in range(n) if not self._blocked(wrts[i], states[i:i + 1])[0]]
            length = max(len(wrts[i].targets) for i in keep)
            controls = np.stack([substep_controls(wrts[i].start, np.concatenate(
                [wrts[i].targets, np.repeat(wrts[i].targets[-1:], length - len(wrts[i].targets), 0)]), e.substeps)
                for i in keep])
            init = np.concatenate([self.sim.initial_states(wrts[i].start, states[i:i + 1, :3]) for i in keep])
            sd = rollout([self.sim.variant(*states[i, 3:]) for i in keep], init, controls)
            for j, i in enumerate(keep):
                wrt = wrts[i]
                steps = sd[j, e.substeps - 1 :: e.substeps][: len(wrt.targets)]
                f = self.features(wrt, steps[:, FORCE], steps[:, PUSHER])
                if wrt.guarded and f[0] == 1.0:
                    nxt = cube_pose(steps[self._contact_step(wrt, steps[:, FORCE])])
                else:
                    nxt = cube_pose(steps[-1])
                a, l, _ = _frame(wrt)
                dxy = nxt[:2] - states[i, :2]
                dyaw = np.mod(nxt[2] - states[i, 2] + math.pi / 4, math.pi / 2) - math.pi / 4
                xs.append(model_inputs(wrt, states[i:i + 1], self)[0])
                obs.append(np.pad(f, (0, 5 - len(f))))
                delta.append([dxy @ a, dxy @ l, dyaw])
        return np.array(xs), np.array(obs), np.array(delta)

    def train_model(self, seed: int = 0, log=print) -> dict:
        x, obs, delta = self.dataset(seed)
        n_obs = (x[:, 7] + 4 * x[:, 8] + 2 * x[:, 9]).astype(int)  # continuous features per kind: 1, 4, 2
        contact = obs[:, 0] == 1.0
        y = np.concatenate([obs[:, 1:5], delta], 1)
        valid = np.zeros_like(y, dtype=bool)
        valid[:, 0] = True  # travel always
        for k in range(1, 4):
            valid[:, k] = contact & (n_obs > k)
        valid[:, 4:] = contact[:, None]  # no contact: the cube stays (Δ = 0 exactly)
        mean = np.array([y[valid[:, k], k].mean() for k in range(7)])
        std = np.array([y[valid[:, k], k].std() + 1e-6 for k in range(7)])
        m = self.model
        m.y_mean.copy_(torch.tensor(mean, dtype=torch.float32))
        m.y_std.copy_(torch.tensor(std, dtype=torch.float32))
        X = torch.tensor(x, dtype=torch.float32)
        Y = torch.tensor((y - mean) / std, dtype=torch.float32)
        V = torch.tensor(valid, dtype=torch.float32)
        C = torch.tensor(contact, dtype=torch.float32)
        perm = torch.randperm(len(X), generator=torch.Generator().manual_seed(seed))
        test = perm[: len(X) // 10]
        train = perm[len(X) // 10 :]
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.cfg.train_steps)
        m.train()

        def loss_fn(idx):
            logit, mu, log_std = m(X[idx])
            bce = nn.functional.binary_cross_entropy_with_logits(logit, C[idx])
            nll = ((0.5 * ((Y[idx] - mu) / log_std.exp()) ** 2 + log_std) * V[idx]).sum() / V[idx].sum()
            return bce + nll, bce, nll

        for step in range(self.cfg.train_steps):
            idx = train[torch.randint(len(train), (512,))]
            loss, _, _ = loss_fn(idx)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            if step % 1000 == 0 or step == self.cfg.train_steps - 1:
                with torch.no_grad():
                    _, bce, nll = loss_fn(test)
                log(f"step {step} test bce {bce:.3f} nll {nll:.3f}")
        m.eval()
        with torch.no_grad():
            logit, mu, _ = m(X[test])
            acc = float(((logit > 0).float() == C[test]).float().mean())
            err = ((mu - Y[test]).abs() * torch.tensor(std, dtype=torch.float32) * V[test]).sum(0) / V[test].sum(0)
        os.makedirs(os.path.dirname(self.cfg.model_path) or ".", exist_ok=True)
        torch.save(m.state_dict(), self.cfg.model_path)
        names = ["travel", "f1", "f2", "peak", "da", "dl", "dyaw"]
        return {"samples": len(X), "contact_acc": acc, **{f"mae_{k}": float(v) for k, v in zip(names, err)}}

    # --- the learned model as T and O -----------------------------------------------------------------------------

    def _blocked(self, wrt: WRT, states: np.ndarray) -> np.ndarray:
        e = self.env_cfg
        rel = states[:, :2] - wrt.start[None, :2]
        c, s = np.cos(states[:, 2]), np.sin(states[:, 2])
        u, v = np.abs(c * rel[:, 0] + s * rel[:, 1]), np.abs(-s * rel[:, 0] + c * rel[:, 1])
        return np.hypot(np.maximum(u - e.cube_half, 0), np.maximum(v - e.cube_half, 0)) < e.pusher_radius + 0.003

    def predict(self, wrt: WRT, states: np.ndarray, physics_free: bool | None = None) -> Prediction:
        e = self.env_cfg
        k, n = len(states), N_OBS[wrt.kind]
        blocked = self._blocked(wrt, states)
        with torch.no_grad():
            logit, mu, log_std = self.model(torch.tensor(model_inputs(wrt, states, self), dtype=torch.float32))
        std_y = self.model.y_std.numpy()
        p = torch.sigmoid(logit).numpy()
        y = mu.numpy() * std_y + self.model.y_mean.numpy()
        sigma = np.exp(log_std.numpy()) * std_y
        contact = p > 0.5
        obs = np.zeros((k, n))
        obs[:, 0] = contact
        obs[:, 1] = y[:, 0]
        obs[:, 2:] = np.where(contact[:, None], y[:, 1 : n - 1], 0.0)
        obs[blocked] = 0.0
        obs[blocked, 0] = 2.0
        a, l, _ = _frame(wrt)
        move = np.where(contact[:, None], y[:, 4:7], 0.0)
        next_pose = states[:, :3].copy()
        next_pose[:, :2] += move[:, :1] * a + move[:, 1:2] * l
        next_pose[:, 2] = np.mod(next_pose[:, 2] + move[:, 2] + math.pi / 4, math.pi / 2) - math.pi / 4
        next_pose[blocked] = states[blocked, :3]
        fail = np.any(np.abs(next_pose[:, :2]) > np.asarray(e.table_half) - e.cube_half, axis=1)
        self._sigma, self._p_contact = sigma[:, : n - 1], np.where(blocked, 0.5, p)
        return Prediction(obs, next_pose, fail, blocked)

    def log_likelihood(self, observed: np.ndarray, predicted: np.ndarray) -> np.ndarray:
        """Sensor noise plus the model's predictive σ; P(contact) from the model."""
        c = self.cfg
        if self._sigma is None or len(self._sigma) != len(predicted):
            return super().log_likelihood(observed, predicted)
        obs = np.atleast_2d(observed)[None]
        pred = predicted[:, None]
        p = np.clip(self._p_contact, c.contact_error, 1 - c.contact_error)[:, None]
        ll = np.where(obs[..., 0] == 1.0, np.log(p), np.log(1 - p))
        ll = np.where((obs[..., 0] == 2.0) | (pred[..., 0] == 2.0),
                      np.where(obs[..., 0] == pred[..., 0], 0.0, math.log(c.contact_error)), ll)
        both = (obs[..., 0] == 1.0) & (pred[..., 0] == 1.0)
        n = pred.shape[-1]
        sens = np.array([c.sigma_travel] + [c.sigma_force[0]] * (n - 2))
        sigma = np.sqrt(sens**2 + self._sigma[:, None, : n - 1] ** 2)
        if n > 2:
            sigma[..., 1:] = sigma[..., 1:] + c.sigma_force[1] * np.abs(pred[..., 2:])
        z = (obs[..., 1:] - pred[..., 1:]) / sigma
        cont = -(0.5 * z**2 + np.log(sigma / sens)).sum(-1)
        ll = ll + np.where(both, cont, 0.0)
        return ll if np.ndim(observed) > 1 else ll[:, 0]

    def update(self, wrt: WRT, observed: np.ndarray) -> float:
        b = self.belief
        # Markov switching of the physics context z before the correction
        phys = b.p.sum(axis=(3, 4, 5), keepdims=True)
        b.set((1 - self.cfg.switch_prob) * b.p + self.cfg.switch_prob * phys / np.prod(b.shape[3:]))
        c = self.cfg
        flat, _ = b.support(c.support_mass, c.update_support)
        states = b.values(flat)
        pred = self.predict(wrt, states)
        ll = self.log_likelihood(observed, pred.obs)
        next_values = states.copy()
        next_values[:, :3] = pred.next_pose
        evidence = b.update(flat, ll, next_values, keep=~pred.fail)
        if wrt.kind != "touch":
            b.blur()
        return evidence

    # --- expected free energy -------------------------------------------------------------------------------------

    def _proposals(self, w: np.ndarray) -> list[WRT]:
        c = self.cfg
        out = [self.touch(w, f, o) for f in range(4) for o in c.touch_offsets]
        out += [self.probe(w, f) for f in range(4)]
        goals = [self.goal_push(w, f, o, v, L) for f in range(4) for o in c.goal_offsets for v in c.goal_speeds
                 for L in c.goal_lengths]
        cost = [float(self._goal_cost(self.predict(g, w[None]).next_pose[0])) for g in goals]
        return out + [goals[i] for i in np.argsort(cost)[: c.goal_proposals]]

    def decide(self) -> tuple[WRT | None, dict]:
        c, b = self.cfg, self.belief
        w = self.map_state()
        report = {"map": w, "entropy": b.entropy(), "p_goal": self.goal_probability(),
                  "pose_entropy": float(entropy(b.marginal((0, 1, 2))))}
        if report["p_goal"] >= 1.0 - c.delta:
            report["reason"] = "goal reached in belief"
            return None, report
        flat, weights = b.support(c.support_mass, c.ig_support)
        weights = weights / weights.sum()
        states = b.values(flat)
        best, best_g, scores = None, np.inf, {}
        for wrt in self._proposals(w):
            pred = self.predict(wrt, states)
            risk = float(weights[pred.blocked | pred.fail].sum())
            if risk > c.risk:
                continue
            pragmatic = float((weights * np.where(pred.blocked | pred.fail, self._goal_cost(states[:, :3]),
                                                  self._goal_cost(pred.next_pose))).sum())
            next_values = states.copy()
            next_values[:, :3] = pred.next_pose
            ig = expected_information_gain(weights, self.log_likelihood(pred.obs, pred.obs), b.nearest(next_values))
            g = pragmatic - c.epistemic_weight * ig
            scores[wrt.name] = (pragmatic, ig, risk)
            if g < best_g:
                best, best_g = wrt, g
        report["scores"] = scores
        if best is None:
            report["reason"] = "every proposal above the risk bound"
            return None, report
        pr, ig, risk = scores[best.name]
        report.update(reason=f"min EFE {best_g:.2f} = cost {pr:.2f} - {c.epistemic_weight:g} x IG {ig:.2f}"
                             f" (risk {risk:.2f})", score=-best_g)
        return best, report
