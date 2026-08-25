# BELIEF TRANSFER — from `Belief-based-pushing` into this skrl fork

Companion doc: [`HANDOFF_ISAACLAB_BELIEF.md`](./HANDOFF_ISAACLAB_BELIEF.md)
(usage guide: running these belief implementations inside Isaac Lab tasks).

---

## 0. Branch / release state (verified)

| Item | State |
|---|---|
| Fork | `github.com/thdhyan/skrl` (origin), working branch `feature/belief-pomdp-manipulation` |
| Base | `d19e5ed9 Release 2.1.0` — identical to **PyPI `skrl==2.1.0`** (latest release) |
| Fork `main` | points at the same Release 2.1.0 commit ⇒ clean PyPI-equivalent base |
| Belief mods | 6 commits on top of 2.1.0 (`9eee9926..06e00c45`): belief package, BeliefDQN, mixin, motion controllers, tests/examples/benchmarks, CartPole belief-PPO smoke proof |
| Sync | local fast-forwarded to `origin/feature/belief-pomdp-manipulation@06e00c45`; docs committed & pushed |

So: **latest released skrl (PyPI 2.1.0) + our belief modifications**, nothing else.

Source repo being transferred FROM:
`~/Projects/ebasa/Belief-based-pushing` (Isaac Lab Franka cube-push,
RSS-20-style particle-filter belief; its own `HANDOFF.md` has full details).

Reference paper: Wirnshofer et al., *Controlling Contact-Rich Manipulation
Under Partial Observability*, RSS 2020.

---

## 1. What this fork ALREADY has (do not re-implement)

### `skrl/belief/`
- `torch/base.py` — `Belief` ABC: `reset / predict / update / compress /
  get_particles / get_weights / get_compressed / get_entropy`.
- `torch/particle_filter.py` — SIR particle filter: log-weights, systematic
  resampling on ESS threshold, cached GMM compression, backend injected via
  `motion_model_fn(particles, control)` and
  `measurement_model_fn(particles, measurement) -> log_likelihoods`.
- `torch/compression.py` — weighted-EM GMM (`gmm_fit`, k-means++ init,
  diagonal covariances), `flatten_gmm -> [hefts | means | log_diag_covs]`,
  `belief_dim(ng, state_dim) = ng + 2·ng·state_dim`.
- `torch/entropy.py` — GMM upper-bound differential entropy (Huber 2008).
- `backends/base.py` — `DynamicsBackend` ABC
  (`step/get_measurement/get_object_pose/get_robot_state/reset`).
- `backends/mujoco_backend.py`, `mjlab_backend.py`, `stub_backend.py`
  (CI-safe), `isaaclab_backend.py` (**interface stub** — callables must be
  supplied).

### Agents
- `agents/torch/belief_mixin.py` — `BeliefAgentMixin`: `_belief_setup /
  _belief_observe / _belief_advance / reset_belief`, TensorBoard tracking of
  entropy + compression time.
- `agents/torch/belief_dqn/` — complete `BeliefDQN` (double-Q, dueling head,
  ε-greedy, replay stores compressed beliefs not raw obs, AMP, polyak target)
  + paper hyperparameters in `BeliefDQN_CFG`.
- `agents/torch/belief_ppo/` — ⚠ **CFG only.** No `belief_ppo.py` yet.
  Interim solution (proven): `examples/contact_rich_uncertainty/cartpole_pomdp_belief_ppo.py`
  runs **standard `PPO` on a belief observation wrapper** — beat baseline PPO
  on hidden-dimension CartPole variants (see `examples/contact_rich_uncertainty/media/`).
- `agents/torch/motion_controllers/` — eTaSL/eTC attractor QP controller
  suite, `build_controller_suite(gmm_means, target_templates, fk, jac, …)`.

### Env wrappers
- `envs/wrappers/torch/pomdp_wrapper.py` — `PomdpWrapper(obs_indices=[…],
  noise=…)` masks chosen dims of ANY gymnasium obs (1 = visible, 0 = hidden)
  + optional sensor noise. `get_full_obs()` still exposes ground truth for
  reward/viz/debug.
- `envs/wrappers/torch/wrapper.py` `wrap_env(...)` normalizes raw gym /
  Isaac Lab envs into skrl's interface.

### Proof artifacts
- `tests/belief/{test_particle_filter,test_compression}.py` (19 tests),
  `tests/agents/torch/test_belief_dqn.py` (8 tests).
- `benchmark_belief_dqn.py`, `verify_belief_compression.py`,
  `runs/belief_compression_analysis/`, `runs/benchmark_belief_dqn/`.
- Examples: `cartpole_pomdp_belief_dqn.py` (agent-owned filter),
  `cartpole_pomdp_belief_ppo.py` (wrapper + standard PPO),
  `peg_in_hole*.py`.

---

## 2. What lives in `Belief-based-pushing` and is worth transferring

| Asset | Location (source repo) | Transfer value |
|---|---|---|
| GPU-batched planar particle filter (contact-gated prediction, asymmetric contact/free-space correction, ESS resample, entropy/ESS diagnostics) | `source/franka_cube_push_project/.../cube_push/mdp/belief.py` | ★ core algorithm; env-side, scales to thousands of parallel Isaac Lab envs — nothing in skrl covers the *batched per-env* case |
| Bimodal prior + "omni-reset" mixture prior (local 78% / secondary tangential mode 17% / uniform 8%) | same file, `reset_planar_particle_belief[_around_cube]` | principled reset diversity for training |
| Compressed 24-float belief vector `[mode0(7)|mode1(7)|mean(2)|goal(2)|Δgoal(2)|diag(3)]` | `_two_component_features` | alternative to EM-GMM compression: deterministic, O(N) weighted 2-means, differentiable-free, no EM iterations — much cheaper inside env step |
| High-level action terms: `BeliefGuidedPushAction` (4-D continuous: steer/anchor/standoff/push through pre-push→push-through IK state machine) and `BeliefDiscretePushAction` (scalar → NC = modes × offset-grid controller bank, 2 Hz) | `.../mdp/actions.py` | the "brittle geometry in low-level controller, policy modulates" trick; directly reusable for any push/drag/manipulate task |
| Asymmetric actor/critic recipe (actor belief-only, critic privileged full state) | `cube_push_env_cfg.py` `BeliefAsymmetricObservationsCfg` | free rsl_rl feature; also portable to skrl via separate critic models fed privileged tensors |
| Dueling Residual DQN utilities + fixed replay `ObservationNormalizer` + discrete-id↔scalar encoding + checkpoint format | `learning/dqn.py` | reference implementation; skrl's `BeliefDQN` supersedes it, but the **normalizer-in-checkpoint** practice must be kept |
| USD belief visualizer (yellow posterior particle cubes + orange mode outlines) + demo/recorder script | `visualization/belief_markers.py`, `scripts/run_belief_demo_franka.py` | port verbatim for Isaac Lab overlays; MP4 recording + `outputs/diagnostics` flow |
| MuJoCo reproduction roadmap (paper-faithful model-based particle rollouts = Phase 4) | `mujoco_repro/DESIGN.md` | maps onto filling `DynamicsBackend.step` properly |

---

## 3. Architecture mapping (decide per use-case)

```
                     ┌──────────────────────────────────────────────┐
                     │            WHERE THE FILTER LIVES            │
                     ├───────────────┬───────────────┬──────────────┤
                     │ A. WRAPPER    │ B. ENV-SIDE   │ C. AGENT-SIDE│
                     │  (skrl today) │  (pushing     │  (skrl       │
                     │               │   repo today) │   BeliefDQN) │
                     ├───────────────┼───────────────┼──────────────┤
   parallel envs     │ 1 env/filter  │ N envs batched│ 1 env/filter │
   engine coupling   │ none          │ Isaac Lab mgr │ DynamicsBknd │
   standard agent?   │ ✔ plain PPO/  │ ✔ plain PPO/  │ ✖ needs      │
                     │   DQN         │   DQN         │  Belief*     │
   matches paper?    │ approx        │ approx        │ closest      │
   status            │ PROVEN        │ PROVEN        │ PROVEN(DQN)  │
                     │ (CartPole)    │ (Franka)      │ (unit tests) │
                     └───────────────┴───────────────┴──────────────┘
```

- **A. Wrapper** — a `gymnasium.Wrapper` owns a `ParticleFilter`; masked obs go
  in, flattened GMM b̃ goes out; any stock skrl agent trains on it. Fastest
  path for ANY task. Already proven here (CartPole belief-PPO/DQN).
- **B. Env-side batched** — filter state as `[num_envs, N, dim]` tensors
  updated once per env step inside the Isaac Lab env; obs term emits the
  compressed vector. Only option that scales to 1024-env GPU training.
  This is the main thing to transfer from the pushing repo.
- **C. Agent-side** — agent owns the filter via `BeliefAgentMixin` +
  `belief_factory`/`DynamicsBackend`; replay stores b̃. Paper-faithful;
  single-env / real-robot regime.

---

## 4. Transfer work items (ordered, independently verifiable)

### T-1 · Port the planar filter as `skrl/belief/torch/planar_batched.py`
Copy `mdp/belief.py` from the pushing repo; strip Isaac Lab imports behind
callables so the math is testable without Isaac Lab:
```python
class PlanarBatchedFilter:
    def __init__(self, num_envs, num_particles=256, device="cuda:0", **cfg): ...
    def reset(self, env_ids, prior_xy, prior_yaw=None): ...      # priors §2.6 of source HANDOFF
    def update(self, ee_xy, ee_delta, measured_contact, goal_xy): ...   # predict+correct+resample
    def compressed(self, goal_xy) -> torch.Tensor                 # [num_envs, 24]
    def particles_world() / posterior_samples() / mode_centers()  # for viz
```
Keep the exact semantics: contact-gated push delta, asymmetric likelihood
strengths (`contact 0.90 / free-space 0.18`, floors `0.03/0.50`),
`uniform_weight_mix`, ESS-resample at 0.55, covariance floor, log-variance
outputs. Verify: unit test vs the original file on synthetic trajectories
(max abs diff < 1e-5), then reuse `tests/belief/test_compression.py` pattern
for the two-mode features.

### T-2 · Isaac Lab adapter (thin)
`skrl/belief/backends/isaaclab_env_adapter.py`: given a manager-based env +
`SceneEntityCfg`s (object/goal/ee), produce `ee_xy`, `ee_delta`,
`measured_contact`, `goal_xy` tensors each step and wire T-1. ~100 lines.
Verify: run inside the Franka push env from the source repo; compare its
24-dim output against the in-repo `compressed_planar_belief` for one episode.

### T-3 · Implement `BeliefPPO` (`skrl/agents/torch/belief_ppo/belief_ppo.py`)
CFG already exists. Mirror `BeliefDQN` structure: rollout memory tensors
`beliefs/next_beliefs/actions/rewards/terminated/log_prob/values` sized by
`self.belief_dim`; `act()` = `_belief_observe()` → policy sample;
`record_transition()` = super() + `_belief_advance()`; `update()` = copy stock
skrl PPO GAE/clip math onto those tensors. Verify: extend the CartPole
belief-PPO wrapper test — agent-owned filter must match wrapper results
within seed noise; loss decreases; memory shapes correct.

### T-4 · Fill or retire `isaaclab_backend.py`
Two honest options: (a) implement the five callables for single-env
model-based particle rollouts (DESIGN.md Phase 4 — expensive, paper-faithful),
or (b) mark it deprecated pointing at A/B patterns above. Decide before
someone builds on the stub.

### T-5 · Port the visualizer
Copy `visualization/belief_markers.py` (only depends on `omni.usd` + `pxr`)
to `examples/contact_rich_uncertainty/usda_belief_markers.py` (or
`skrl/utils/belief_viz/`). Wire `.set_particles(belief.posterior_samples())`
+ `.set_mode_outlines(belief.mode_centers())` per render tick. Add a
matplotlib fallback reading `filter.get_particles()/get_weights()` for
non-Isaac tasks.

### T-6 · Docs
Fold the stale root `handoff.md` ("planning only, no code committed") into
this file or delete it after T-1–T-3 land; keep its callback-contract section
(§"Information the user needs") — still the authoritative list of what an
environment must supply.

---

## 5. Definition of done

1. `pytest tests/belief tests/agents/torch/test_belief_dqn.py -q` green.
2. New `test_planar_batched.py` parity test green (T-1).
3. One Isaac Lab task trained end-to-end with skrl (any of patterns A/B)
   reaching ≥ scripted-baseline success rate; TensorBoard shows
   `Belief/entropy` decaying on successful episodes.
4. Demo video with belief overlays recorded via the ported marker module.
