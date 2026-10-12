# Tasks — Belief, privileged-curriculum and transformer RL in skrl

> Working branch: `feature/privileged-curriculum` (contains `feature/belief-pomdp-manipulation` and the cloud
> branch `claude/2d-multiagent-ppo-grid-n4ld8f`, merged 2026-10-11). Plans and ordering: [`plans.md`](plans.md).
> Test status 2026-10-11: 112 passed (command in `README.md`).

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

## 7b. BeliefPPO agent ✅ (2026-10-10)

- [x] `skrl/agents/torch/belief_ppo/belief_ppo.py` — `BeliefPPO(BeliefAgentMixin, Agent)`, agent-owned filter (pattern C), num_envs=1.
- [x] `skrl/agents/torch/belief_ppo/models.py` — `ResNetCategoricalPolicy`, `ResNetValue` (paper trunk).
- [x] `BeliefPPO_CFG.belief_prior_fn(obs)` — re-seeds filter at each episode start.
- [x] `tests/agents/torch/test_belief_ppo.py` — 8 tests; belief suite 44/44 green.
- [x] `examples/contact_rich_uncertainty/cartpole_pomdp_belief_ppo_agent.py` — CartPole-POMDP smoke (reaches 500; raw-obs baseline ~100).
- [x] Pipeline diagrams + write-up: `docs/belief_ppo/`.
- [ ] Multi-seed comparison C vs A (same MLP) — gap unresolved on 1 seed.
- [ ] Batched (num_envs>1) BeliefPPO or pattern B for Isaac Lab.

## 7c. Transformer policies for MuJoCo ✅ smoke (2026-10-10)

- [x] `skrl/models/torch/transformer.py` — per-dim tokenizer + CLS transformer trunk, Gaussian policy + value.
- [x] `examples/transformer_mujoco/` — PPO on HalfCheetah-v5 / InvertedPendulum-v5 / InvertedDoublePendulum-v5, `--model {transformer,mlp}`, video.
- [x] Multi-seed benchmark, 3 tasks × 3 seeds (HalfCheetah 1M steps): MLP ≥ transformer on all three. Table in the example README.
- [x] Fix: `clip_actions=False` (log_prob of clipped actions overflowed the PPO ratio, NaN in 4/18 runs).
- [ ] Better token design (per GMM component of b̃, or per history time step). No triple-pendulum env in gymnasium (custom MJCF needed).

## 7d. PrivilegedPPO: privileged masking curriculum ✅ (2026-10-10)

- [x] `skrl/agents/torch/privileged_ppo/` — `SoftmaxMaskCurriculum`, encoder z / history estimator ẑ actor, asymmetric critic, `‖sg(z) − ẑ‖² + λ‖z − sg(ẑ)‖²` regularizer.
- [x] `skrl/envs/wrappers/torch/privileged_history_wrapper.py` — obs `[o_t | history | privileged | mask]`, per-episode mask.
- [x] `tests/agents/torch/test_privileged_ppo.py` — 7 tests.
- [x] CartPole POMDP + random pole mass/length sweep, 3 modes × 3 seeds: masked eval curriculum 493, masked-from-start 490, always-visible 118. `docs/privileged_curriculum/`.
- [ ] Harder task where privileged values are not recoverable from a short history; 5+ seeds.
- [ ] Per-dimension masking order and soft blending variants.
- [ ] Isaac Lab: privileged values from the `critic` obs group.

## 7e. Pursuit-evasion self-play + POMDP experiments ✅ (cloud branch, merged 2026-10-11)

- [x] `skrl/envs/pursuit_evasion/` — batched torch 2D grid env, potential fields, obs groups (`full_state`, `global_potential`, `local_potential`).
- [x] `skrl/multi_agents/torch/selfplay/` — generation-based self-play runner, opponent pools / Elo, samplers.
- [x] `PomdpWrapper` fixes (multi-dim obs, negative indices, `full_obs` on reset) + `tests/envs/test_pomdp_wrapper.py`.
- [x] W&B fix: `skrl/utils/tensorboard.py` (no training metrics were synced).
- [x] `experiments/pomdp_and_obs_groups/` — CartPole full/masked/belief and pursuit obs-group studies (offline W&B).
- [x] `wandb sync` the offline runs (15 runs, 2026-10-11).
- [ ] Remove `wandb_offline/` from git history (needs a user-run history rewrite + force-push).
- [x] 100×100 arena, spawn domain randomization, potential-field + trail rendering of rollouts (`skrl/envs/pursuit_evasion/render.py`), per-generation opponent sampler schedule and videos in the self-play runner.

## 7f. Belief markers for visualization ✅ (2026-10-11)

- [x] `skrl/belief/torch/markers.py`: `belief_markers` (particles, weights, GMM clusters, optional world positions via `state_to_position`), `belief_stats` (spread, ESS, number / effective clusters, cluster stds, errors of mean and clusters vs the true state).
- [x] `BeliefAgentMixin.get_belief_markers()`, `cfg.return_belief_markers` (markers in `act()` outputs), W&B/TensorBoard `Belief / *` stats per step.
- [x] `BeliefMarkerServer` (HTTP `GET /belief`) + `fetch_belief_markers` for a sim in another process; `mujoco_add_belief_geoms`.
- [x] CartPole wrapper `info["belief"]`; demo `examples/contact_rich_uncertainty/belief_markers_mujoco_demo.py` (2-panel video).
- [x] Tests: `tests/belief/test_markers.py`, BeliefPPO markers test.
- [ ] Isaac Lab `VisualizationMarkers` client (isaac-sim MCP was down; not tested in Isaac Sim).

## 7g. Privileged curriculum on built-in tasks + history estimator as system ID (2026-10-11)

- [x] `RewardGatedMaskCurriculum`: mask step only while the smoothed return holds its level, back-off on drops, deadline floor.
- [x] Per-group probes: held-out R² of ẑ / z for each privileged group (`cfg.privileged_groups`).
- [x] `examples/privileged_curriculum/gym_privileged_ppo.py`: Pendulum, InvertedPendulum, HalfCheetah, Hopper with hidden velocities + randomized mass / friction / length.
- [x] Fix: evaluation now starts fresh episodes (the skrl vector wrapper resets only once).
- [x] `PrivilegedHistoryWrapper(include_last_action=True)`: observation-action history.
- [x] Sweep on dl (4 tasks × 4 modes × 3–5 seeds) + history ablation (H, actions; curriculum and visible) + λ = 0 ablation: `docs/privileged_curriculum/plot_gym.py`, results in `docs/privileged_curriculum/README.md`. Finding: ẑ recovers velocities, not dynamics (R² ≈ 0–0.1), even when z keeps them (λ = 0).
- [x] Slides (`build_sysid_slides.py`) → [Google Slides](https://docs.google.com/presentation/d/14fJ3Bh9L9cFB9bukCwd-6vyzbkuNAJnS5_WehLsm1Tc/edit); write-up → [Google Doc](https://docs.google.com/document/d/1WaQWwzApMI9jwM4obc2njVuPb743bSdufxjB7laiVXE/edit).
- [ ] Supervised system-ID head (regress e_t from history) and wider dynamics ranges / lower noise.
- [ ] Temporal-transformer estimator vs MLP (see `docs/transformer_rl_robotics.md`).
- [x] Isaac Lab YAML / `Runner` support: `PrivilegedPPO` (strategy presets, curriculum, estimator arch), `BeliefPPO` / `BeliefDQN` (belief section or import-string factory), `PrivilegedHistoryVecWrapper`; example YAMLs in `examples/privileged_curriculum/isaaclab/`, docs in `HANDOFF_ISAACLAB_BELIEF.md` §8.7.
- [x] Isaac Lab smoke run of `skrl_privileged_ppo_cfg.yaml` on a registered task (with debug video): zz-bw, Cartpole
  with hidden velocities (`examples/privileged_curriculum/isaaclab/cartpole_privileged_smoke.py` via `zzbw_smoke.sh`),
  sysid + transformer estimator from the registry YAML, mask p 0 -> 1, checkpoint + 2 clips, frames checked (2026-10-11).

## 7h. Belief-method survey + encoders for Dict observations (2026-10-12)

- [x] Deck comparing the belief of RSS20, WAFR08 (Hsiao et al.), Audit Before You Commit, GenAIF, quasi-static
  belief dynamics (Jankowski) and skrl PrivilegedPPO/BeliefPPO, with equations:
  `docs/belief_methods/build_belief_methods_slides.py` →
  [Google Slides](https://docs.google.com/presentation/d/1Wz_vzRW_W-nKEJHs8J6sakNd3D9xXgZwO-0cHdmWYxU/edit).
- [x] `skrl/models/torch/encoders.py`: per-key encoders for Dict observations (`mlp`, `cnn`, `pointnet`,
  `dinov2` via torch.hub, `clip_image` / `clip_text` via open_clip, `identity`, `register_encoder` or
  `"module:Class"` for any other), `DictEncoder` with `concat` or FiLM fusion, models `EncoderGaussianPolicy`,
  `EncoderCategoricalPolicy`, `EncoderDeterministic` (Runner YAML names). Pretrained backbones frozen by default.
- [x] `DictStandardScaler`: per-key observation preprocessor (standardizes low-dimensional float keys, passes
  images / points / tokens through), YAML `observation_preprocessor: DictStandardScaler`.
- [x] Tests: `tests/models/torch/test_encoders.py` (fake DINO/CLIP backbones, Runner PPO end-to-end, checkpoint).
- [ ] Real DINOv2 / CLIP run: install `open_clip_torch`, torch.hub cache on the server; Isaac Lab camera task.
- [ ] PrivilegedPPO still refuses observation preprocessors and works on a flat Box (no Dict encoders yet).
- [x] MuJoCo cube push with hidden pose + μ, m, k_d (`skrl/envs/cube_push/`): `CubePushEnv` (gymnasium; obs =
  pusher xyz + force xyz + goal pose, action = pusher target; the command is only the wanted final cube pose),
  `CubePushModel` (physics variants, batched `mujoco.rollout` hypotheses that reproduce the env trace exactly).
- [x] WAFR08 (Hsiao et al.) planner `skrl/envs/cube_push/wafr08.py` on `skrl/belief/grid_belief.py` (6-D grid
  belief, multilinear splat, IG on the next-state belief): guarded touches localize the pose, goal pushes run when
  b ∈ BPre (P ≥ 1 − δ, 2× waypoint tolerance), probes (v1/v2 force ≈ μmg + k_d v, peak ≈ m a) only once the pose
  entropy < 1.5 nats. Seeds 0/1/2: 3/3 success in 21/27/22 WRTs (1.3 cm 1.9°, 0.4 cm 4.8°, 0.4 cm 5.5°).
- [x] Belief debug video (`skrl/envs/cube_push/render.py`, `examples/contact_rich_uncertainty/cube_push_wafr08.py`):
  overview / top / pusher-follow + belief plots, belief tiles, MAP, goal + tolerance, table limit, WRT path.
  `runs/cube_push_wafr08/videos/wafr08_seed*.mp4`. Tests `tests/envs/test_cube_push.py` (5).
- [x] Separate deck with the renders (`docs/belief_methods/build_cube_push_wafr08_slides.py`, keyframes in
  `docs/belief_methods/cube_push/`): Google Slides "Cube push WAFR08 belief renders"
  (https://docs.google.com/presentation/d/1xjRNSOlTSZCZv43dooe_DHhNCsMzQ9ptLkDon3rv2Kc/edit) with the three mp4s
  embedded from Drive folder https://drive.google.com/drive/folders/1KVtaCvJHo__JFWIpUJOHmmy6efQpm_dB.
- [ ] Physics levels are coarse (3 per parameter) and μ·m is nearly degenerate from push forces alone; probes rarely
  fire. Next: Audit, GenAIF, quasi-static and RSS20 on the same env.

## 8. CartPole physics extensions (answered below)

CartPole friction and mass are configurable — see §8 answer in handoff.md.
