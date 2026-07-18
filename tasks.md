# Tasks — Belief-Based POMDP Manipulation in skrl

> Branch: `feature/belief-pomdp-manipulation`

## 0. Bootstrap ✅

- [x] Branch `feature/belief-pomdp-manipulation` created, clean tree off `main`.
- [x] `skrl/belief/` public package created with `__init__.py` re-exports.
- [ ] Add `[belief]` / `[belief-mujoco]` / `[belief-mjlab]` extras to `pyproject.toml`.

## 1. Belief interface + particle filter ✅

- [x] `skrl/belief/torch/base.py` — `Belief` ABC.
- [x] `skrl/belief/torch/particle_filter.py` — `ParticleFilter(Belief)`, NP=300, systematic resampling.
- [x] `skrl/belief/torch/compression.py` — `gmm_fit` (EM + k-means++ init), `flatten_gmm`, `belief_dim`. Log-diag-cov layout matching paper §V-A.
- [x] `skrl/belief/torch/entropy.py` — GMM upper-bound differential entropy (Huber 2008).

## 2. Pluggable dynamics backends ✅

- [x] `skrl/belief/backends/base.py` — `DynamicsBackend` ABC.
- [x] `skrl/belief/backends/stub_backend.py` — pure-torch 2-link arm + box, no MuJoCo dep. Unblocks CI.
- [x] `skrl/belief/backends/mujoco_backend.py` — MuJoCo + impedance controller (eq. 3-4). Requires `pip install mujoco`.
- [x] `skrl/belief/backends/mjlab_backend.py` — mjlab vectorised env backend.
- [x] `skrl/belief/backends/isaaclab_backend.py` — callback-injection stub for IsaacLab.

## 3. Low-level motion controllers ✅

- [x] `skrl/agents/torch/motion_controllers/base.py` — `MotionController` ABC.
- [x] `skrl/agents/torch/motion_controllers/etasl_attractor.py` — eTaSL/eTC QP (eq. 7). qpth if available, lstsq+clamp fallback.
- [x] `skrl/agents/torch/motion_controllers/registry.py` — `build_controller_suite(NC = n·NG)`.
- [x] `skrl/agents/torch/motion_controllers/multi_agent_wrapper.py` — `MotionControllerSuite`.

## 4. High-level BeliefDQN agent ✅

- [x] `skrl/agents/torch/belief_dqn/belief_dqn_cfg.py` — `BeliefDQN_CFG`. All paper hyperparams.
- [x] `skrl/agents/torch/belief_dqn/models.py` — `ResNetDuelingQNetwork`. Fixed: `DeterministicMixin` + `Model` explicit init, `self.elu` (not `self.act`), keyword-only `Model.__init__`.
- [x] `skrl/agents/torch/belief_dqn/belief_dqn.py` — `BeliefDQN(Agent)`. Double-Q, dueling, soft target, AMP, TensorBoard.
- [x] `skrl/agents/torch/belief_dqn/__init__.py`.

## 5. Tests ✅ — 27/27 green

- [x] `tests/belief/test_particle_filter.py` — 9 tests: shapes, weights, predict/update/resample, entropy decrease, full cycle.
- [x] `tests/belief/test_compression.py` — 8 tests: GMM fit, cluster recovery, flatten shape, entropy monotone.
- [x] `tests/agents/torch/test_belief_dqn.py` — 8 tests: init, belief_dim, act shape, record_transition, training loop, double-Q, dueling output shape, greedy.

## 6. POMDP wrapper + gymnasium benchmarks ✅

- [x] `skrl/envs/wrappers/torch/pomdp_wrapper.py` — `PomdpWrapper`: hides obs indices, adds noise, exposes `get_full_obs()` for TensorBoard ground-truth logging.
- [x] `examples/contact_rich_uncertainty/cartpole_pomdp_belief_dqn.py` — CartPole-v1 POMDP (hides cart_vel + pole_ang_vel). Analytical dynamics motion model, TensorBoard gt vs belief logging, matplotlib live visualisation (particle cloud + GMM ellipses + ground truth cross).
- [x] `examples/contact_rich_uncertainty/peg_in_hole.py` — stub-backend smoke test (paper task).

## 7. Remaining / next steps

- [ ] `pyproject.toml` — add `[belief]`, `[belief-mujoco]`, `[belief-mjlab]` extras.
- [ ] Pendulum-v1 POMDP example (hide θ̇, discretise torque into NC bins).
- [ ] MountainCar-v0 POMDP example (hide velocity, multi-modal belief naturally).
- [ ] CartPole with custom friction / mass — see `cartpole_custom_dynamics.py` plan below.
- [ ] Comparison script: BeliefDQN (partial obs) vs standard DQN (full obs) vs DQN+MLE (paper baseline).
- [ ] Tests for motion controllers QP feasibility.
- [ ] `docs/source/belief.rst` + CHANGELOG entry.
- [ ] `handoff.md` update.

## 8. CartPole physics extensions (answered below)

CartPole friction and mass are configurable — see §8 answer in handoff.md.
