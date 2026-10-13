"""RSS20 (Wirnshofer, Schmitt, v. Wichert, Burgard) on the partially observable cube push.

* Belief: ``particles`` hypotheses of the cube pose and physics (physics on a 5-level lattice per parameter), each
  simulated in MuJoCo under the executed controller; the weights are multiplied by the likelihood of the robot's
  feedback (the controller's contact / travel / force features), then resampled with jitter when the effective
  sample size drops below half.
* Compression: a 2-component diagonal GMM of the particles (``skrl.belief.torch.gmm_fit``), flattened to
  b̃ = [π | μ | log σ²] over (x, y, cos 4ψ, sin 4ψ, μ, m, k_d), standardized.
* Controller bank around each GMM mode g (heaviest first): 4 guarded touches, 1 force probe towards the goal and the
  3 goal pushes whose simulated outcome from the mode is closest to the goal → 16 controllers.
* Policy: a dueling double DQN (ResNet Q-network of ``belief_dqn/models.py``) on [b̃, goal], trained on the
  controller-level MDP :class:`RSS20MacroEnv`: reward −1 per controller (plus potential shaping γΦ(s') − Φ(s),
  Φ = −distance to goal / tolerance, training only), γ = 0.98, terminal when the cube is within the goal
  tolerance (the task-completion signal), at most ``max_controllers`` per episode.

For rendering and the goal probability the particles are histogrammed onto WAFR08's grid (``self.belief``).
"""
from __future__ import annotations

import dataclasses
import math

import gymnasium
import numpy as np
import torch

from skrl.belief.torch.compression import flatten_gmm, gmm_fit
from skrl.envs.cube_push.env import CubePushEnv, pose_error
from skrl.envs.cube_push.sim import CubePushCfg, wrap_yaw
from skrl.envs.cube_push.wafr08 import WAFR08Cfg, WAFR08CubePlanner, WRT

FEATURE_MEAN = np.array([0.06, 0.0, 0.0, 0.0, 0.55, 0.4, 1.5])
FEATURE_SCALE = np.array([0.1, 0.1, 1.0, 1.0, 0.25, 0.2, 1.5])
OBS_DIM = 2 + 2 * 2 * 7 + 4


@dataclasses.dataclass
class RSS20Cfg(WAFR08Cfg):
    particles: int = 128
    physics_levels: int = 5
    jitter_position: float = 0.003
    jitter_yaw_deg: float = 3.0
    jitter_physics: float = 0.1  # probability of moving one physics level
    goal_pushes: int = 3
    max_controllers: int = 30
    shaping: float = 1.0
    gamma: float = 0.98


class RSS20CubePlanner(WAFR08CubePlanner):
    num_controllers = 2 * (4 + 1 + 3)

    def __init__(self, env_cfg: CubePushCfg | None = None, cfg: RSS20Cfg | None = None, seed: int = 0,
                 policy: torch.nn.Module | None = None) -> None:
        super().__init__(env_cfg, cfg or RSS20Cfg())
        e, c = self.env_cfg, self.cfg
        self.rng = np.random.default_rng(seed)
        self.levels = [np.linspace(*r, c.physics_levels) for r in (e.mu, e.mass, e.damping)]
        self.policy = policy
        self.particles = np.zeros((c.particles, 6))
        self.weights = np.full(c.particles, 1.0 / c.particles)
        self.bank: list[WRT] = []
        self.gmm = None

    # --- particle belief ------------------------------------------------------------------------------------------

    def reset(self, goal: np.ndarray) -> None:
        super().reset(goal)
        e, c, n = self.env_cfg, self.cfg, self.cfg.particles
        self.particles = np.stack([
            self.rng.uniform(*e.cube_x, n), self.rng.uniform(*e.cube_y, n), wrap_yaw(self.rng.uniform(-math.pi, math.pi, n)),
            *(self.rng.choice(lv, n) for lv in self.levels)], 1)
        self.weights = np.full(n, 1.0 / n)
        self._compress()

    def _project(self) -> None:
        """Histogram of the weighted particles on the display grid."""
        b = self.belief
        p = np.zeros(b.p.size)
        np.add.at(p, b.nearest(self.particles), self.weights)
        b.set(p.reshape(b.shape) + 1e-12)

    def _compress(self) -> None:
        x = self.particles
        f = np.stack([x[:, 0], x[:, 1], np.cos(4 * x[:, 2]), np.sin(4 * x[:, 2]), x[:, 3], x[:, 4], x[:, 5]], 1)
        f = (f - FEATURE_MEAN) / FEATURE_SCALE
        gmm = gmm_fit(torch.tensor(f, dtype=torch.float32), torch.tensor(self.weights, dtype=torch.float32), ng=2)
        order = torch.argsort(gmm.hefts, descending=True)
        gmm.hefts, gmm.means, gmm.log_diag_covs = gmm.hefts[order], gmm.means[order], gmm.log_diag_covs[order]
        self.gmm = gmm
        self._project()
        self.bank = self._controllers()

    def mode_states(self) -> np.ndarray:
        """``[2, 6]`` pose + physics at the GMM means (physics snapped to the lattice)."""
        m = self.gmm.means.numpy().astype(np.float64) * FEATURE_SCALE + FEATURE_MEAN
        yaw = np.arctan2(m[:, 3], m[:, 2]) / 4
        phys = [lv[np.abs(lv[None] - m[:, 4 + k, None]).argmin(1)] for k, lv in enumerate(self.levels)]
        return np.stack([m[:, 0], m[:, 1], yaw, *phys], 1)

    def map_state(self) -> np.ndarray:
        return self.mode_states()[0]

    def compressed(self) -> np.ndarray:
        g = self.goal
        goal = [(g[0] - FEATURE_MEAN[0]) / FEATURE_SCALE[0], (g[1] - FEATURE_MEAN[1]) / FEATURE_SCALE[1],
                math.cos(4 * g[2]), math.sin(4 * g[2])]
        return np.concatenate([flatten_gmm(self.gmm).numpy(), goal]).astype(np.float32)

    def _controllers(self) -> list[WRT]:
        bank = []
        for w in self.mode_states():
            touches = [self.touch(w, f, 0.0) for f in range(4)]
            goal_dir = self.goal[:2] - w[:2]
            face = max(range(4), key=lambda f: float(np.dot(self._frame(w, f, 0.0, 0.0)[1], goal_dir)))
            pushes = [x[0] for x in self.goal_candidates(w)[: self.cfg.goal_pushes]]
            pushes += [touches[len(pushes)]] * (self.cfg.goal_pushes - len(pushes))  # too few pushes: fill with touches
            bank += touches + [self.probe(w, face)] + pushes
        return bank

    def goal_candidates(self, w: np.ndarray):
        """Only the faces whose push direction points towards the goal (half the rollouts of WAFR08's)."""
        c = self.cfg
        goal_dir = self.goal[:2] - w[:2]
        faces = [f for f in range(4) if np.dot(self._frame(w, f, 0.0, 0.0)[1], goal_dir) > 0] or list(range(4))
        wrts = [self.goal_push(w, f, o, v, length) for f in faces for o in c.goal_offsets for v in c.goal_speeds
                for length in c.goal_lengths]
        preds = self._predict_many(wrts, w)
        ranked = [(x, p.next_pose[0], float(self._goal_cost(p.next_pose[0]))) for x, p in zip(wrts, preds)
                  if not (p.blocked[0] or p.fail[0])]
        return sorted(ranked, key=lambda x: x[2])

    def update(self, wrt: WRT, observed: np.ndarray) -> float:
        c = self.cfg
        pred = self.predict(wrt, self.particles)
        ll = self.log_likelihood(observed, pred.obs)
        logw = np.log(np.maximum(self.weights, 1e-300)) + np.where(pred.fail, -np.inf, ll)
        if not np.isfinite(logw).any():  # nothing explains the observation: keep the prior weights
            logw = np.log(np.maximum(self.weights, 1e-300))
        top = logw.max()
        w = np.exp(logw - top)
        evidence = float(np.log(w.sum()) + top)
        self.weights = w / w.sum()
        self.particles[:, :3] = pred.next_pose
        ess = 1.0 / np.sum(self.weights**2)
        if ess < c.particles / 2 or wrt.kind != "touch":
            self._resample()
        self._compress()
        return evidence

    def _resample(self) -> None:
        c, n = self.cfg, self.cfg.particles
        u = (self.rng.random() + np.arange(n)) / n
        idx = np.minimum(np.searchsorted(np.cumsum(self.weights), u), n - 1)
        x = self.particles[idx].copy()
        x[:, :2] += self.rng.normal(0, c.jitter_position, (n, 2))
        x[:, 2] = wrap_yaw(x[:, 2] + self.rng.normal(0, math.radians(c.jitter_yaw_deg), n))
        for k, lv in enumerate(self.levels):
            i = np.abs(lv[None] - x[:, 3 + k, None]).argmin(1)
            move = self.rng.random(n) < c.jitter_physics
            i = np.clip(i + move * self.rng.choice([-1, 1], n), 0, len(lv) - 1)
            x[:, 3 + k] = lv[i]
        self.particles, self.weights = x, np.full(n, 1.0 / n)

    # --- policy ---------------------------------------------------------------------------------------------------

    def q_values(self) -> np.ndarray:
        with torch.no_grad():
            q, _ = self.policy.compute({"observations": torch.tensor(self.compressed())[None]})
        return q[0].numpy()

    def decide(self) -> tuple[WRT | None, dict]:
        """Greedy controller of the trained Q-network. Stops only on the task-completion signal (the runner's env)."""
        report = {"map": self.map_state(), "entropy": self.belief.entropy(), "p_goal": self.goal_probability()}
        q = self.q_values()
        a = int(np.argmax(q))
        report.update(action=a, q=q.tolist(), reason=f"DQN: controller {a} (mode {a // 8}, Q {q[a]:.1f}; "
                                                      f"pi {self.gmm.hefts[0]:.2f}/{self.gmm.hefts[1]:.2f})")
        return self.bank[a], report

    def extra_markers(self) -> list:
        """Particles as small magenta spheres, GMM modes as magenta boxes."""
        h = self.env_cfg.cube_half
        out = [("sphere", [0.003, 0, 0], [x, y, 0.004], 0.0, [0.9, 0.1, 0.7, 0.3 + 0.7 * min(1.0, w * len(self.weights) / 4)])
               for (x, y), w in zip(self.particles[:, :2], self.weights)]
        for (x, y, yaw, *_), pi in zip(self.mode_states(), self.gmm.hefts.numpy()):
            out.append(("box", [h * 1.02] * 3, [x, y, h], yaw, [0.9, 0.1, 0.7, 0.1 + 0.3 * float(pi)]))
        return out


class RSS20MacroEnv(gymnasium.Env):
    """Controller-level MDP of RSS20: observation [b̃, goal], action = controller index of the bank."""

    def __init__(self, cfg: RSS20Cfg | None = None, seed: int = 0, env_cfg: CubePushCfg | None = None) -> None:
        self.env = CubePushEnv(env_cfg or CubePushCfg())
        self.env.reset(seed=seed)  # seeds the env's generator; later resets continue it
        self.planner = RSS20CubePlanner(self.env.cfg, cfg, seed=seed)
        self.observation_space = gymnasium.spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
        self.action_space = gymnasium.spaces.Discrete(RSS20CubePlanner.num_controllers)
        self.count = 0

    def _potential(self) -> float:
        return -float(pose_error(self.env.cube_pose(), self.env.goal)[0]) / self.env.cfg.goal_tolerance[0]

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        _, info = self.env.reset(seed=None if seed is None else seed, options=options)
        self.planner.reset(info["goal"])
        self.count = 0
        return self.planner.compressed(), {}

    def step(self, action):
        p, c = self.planner, self.planner.cfg
        before = self._potential()
        wrt = p.bank[int(np.asarray(action).reshape(-1)[0])]
        observed = p.execute(self.env, wrt)
        p.update(wrt, observed)
        self.count += 1
        dist, dyaw = pose_error(self.env.cube_pose(), self.env.goal)
        tol_p, tol_y = self.env.cfg.goal_tolerance
        success = bool(dist < tol_p and dyaw < tol_y)
        off = bool(np.any(np.abs(self.env.cube_pose()[:2]) > np.asarray(self.env.cfg.table_half) - self.env.cfg.cube_half))
        reward = -1.0 + c.shaping * (c.gamma * self._potential() - before) - (10.0 if off else 0.0)
        info = {"success": success, "goal_dist": float(dist)}
        return p.compressed(), reward, success or off, self.count >= c.max_controllers, info
