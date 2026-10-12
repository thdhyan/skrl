# Plans

Working branch: `feature/privileged-curriculum` (merged 2026-10-11 with `feature/belief-pomdp-manipulation` and the
cloud branch `claude/2d-multiagent-ppo-grid-n4ld8f`). Task checklist: [`tasks.md`](tasks.md). Older notes:
`handoff.md`, `HANDOFF_ISAACLAB_BELIEF.md`, `BELIEF_TRANSFER.md`.

Goal: policies for contact-rich robot tasks that act under uncertainty, trained in simulation with privileged
information and deployed without it.

## Where things stand (2026-10-11)

| Workstream | State | Evidence |
|---|---|---|
| A. Belief-based agents | BeliefDQN and BeliefPPO work on CartPole POMDP. One filter per agent, so `num_envs = 1` | `docs/belief_ppo/README.md`, `experiments/pomdp_and_obs_groups/README.md` |
| B. Privileged masking curriculum | PrivilegedPPO works; reward-gated schedule beats time-based on 4 built-in tasks, but always-masked is best on HalfCheetah / InvertedPendulum. The history estimator recovers velocities, not dynamics (system-ID R² ≈ 0–0.1) | `docs/privileged_curriculum/README.md` |
| C. Transformer policies | Per-dimension-token transformer trunk works; MLP ≥ transformer on InvertedPendulum, InvertedDoublePendulum, HalfCheetah | `examples/transformer_mujoco/README.md` |
| D. Pursuit-evasion self-play | 2D grid env, MAPPO/IPPO self-play runner, observation-group study | `examples/pursuit_evasion/README.md` |

## Next plans, in order

### 1. Expose the belief for visualization (belief markers) — done 2026-10-11 (API + stats + server; Isaac Lab client pending)

Today the belief is internal. `BeliefPPO.act()` returns only actions and `log_prob`; the wrapper pattern (A) puts
only `belief_entropy` in `info`. The particle filter already has the data: `get_particles()` `[NP, d]`,
`get_weights()` `[NP]`, and `gmm` (`hefts [NG]`, `means [NG, d]`, `log_diag_covs [NG, d]`). The only place it is
drawn is the matplotlib live view inside `examples/contact_rich_uncertainty/cartpole_pomdp_belief_dqn.py`.

Plan:
- `BeliefAgentMixin.get_belief_markers()` → `{"particles", "weights", "gmm_hefts", "gmm_means", "gmm_log_diag_covs"}`
  (detached, on the agent device). Optional `cfg.return_belief_in_outputs` adds it to `act()` outputs for loggers.
- `BeliefObservationWrapper` (pattern A) puts the same dict in `info["belief"]`, so the env side can draw it.
- A user-supplied `state_to_position(state[..., d]) -> [..., 3]` maps the belief state (joint angles, object pose,
  cart position) to world positions for markers.
- Isaac Lab: draw with `VisualizationMarkers` — small translucent spheres for the top-k particles (scaled by
  weight) and one ellipsoid per GMM component (scale = 2σ of the projected covariance), next to the true state
  marker. Fits the debug-video rule (overview, top-down and follow cameras, labelled panels).
- MuJoCo/gymnasium: same dict rendered as an overlay in the existing video recorder.
- In pattern B (filter inside the Isaac Lab env, batched over envs) the env owns the belief and draws it directly.

### 2. Make the privileged curriculum earn its keep

Status 2026-10-11: built-in-task sweep, history / λ ablations done (dl). The bottleneck is the history estimator:
z keeps the dynamics with λ = 0 (R² 0.9+) but ẑ cannot predict them. Next: supervised system-ID head, temporal
transformer estimator, wider dynamics ranges and lower noise, then Isaac Lab manipulation.

CartPole is too easy: 8 noisy observations recover the velocities, so masked-from-start ties the curriculum.
- Move to a task where the privileged values are hard to infer from a short history: friction / mass / contact
  parameters (peg-in-hole, pushing), or a smaller step budget.
- 5+ seeds; report masked-eval return and estimator error vs true privileged values.
- Try per-dimension masking (mask the easiest privileged dims first) and soft blending `z_mix = (1-α) z + α ẑ`.
- Isaac Lab path: privileged values from the `critic` obs group, history from the `policy` group.

### 3. Belief at scale (pattern B)

Batched particle filter over `[N_envs, NP, d]` inside the env, so stock PPO with many parallel envs can use b̃.
Multi-seed BeliefPPO (pattern C) vs wrapper + PPO (pattern A) with the same network to settle the open gap.

### 4. Combine belief and privileged learning

Use the belief b̃ (or its GMM means) as the estimator input in PrivilegedPPO, and the privileged values as the
regression target. Compare: history estimator vs belief estimator vs both.

### 5. Transformers

Only worth more budget with a better token design: tokens per GMM component of b̃, or per time step of the
history (a temporal transformer as the PrivilegedPPO estimator). The per-dimension design lost to MLPs.

### 5b. Belief methods from the literature + perception encoders

Deck: [Google Slides](https://docs.google.com/presentation/d/1Wz_vzRW_W-nKEJHs8J6sakNd3D9xXgZwO-0cHdmWYxU/edit).
Proposed: one MuJoCo planar cube-push env in skrl (hidden pose + μ, m, k_d), shared particle / grid filters and
deciders for WAFR08 (IG vs failure probability), Audit (retained set + coverage audit), GenAIF (CVAE + EFE-MPC),
quasi-static (γ_k bound, CMA-ES) and RSS20 (BeliefDQN). Encoders for camera / point-cloud / language tasks are in
`skrl/models/torch/encoders.py` (done 2026-10-12); next is a real DINOv2/CLIP run on a server.
Cube-push env + WAFR08 done 2026-10-12 (`skrl/envs/cube_push/`, grid belief in `skrl/belief/grid_belief.py`, belief
video example in `examples/contact_rich_uncertainty/cube_push_wafr08.py`). Next on the same env: Audit, GenAIF,
quasi-static, RSS20.

### 6. Housekeeping

- `pyproject.toml` extras `[belief]`, `[belief-mujoco]`, `[belief-mjlab]`.
- [done] `CLAUDE.md` now documents the uv `.venv`.
- The cloud branch committed offline W&B runs (`experiments/pomdp_and_obs_groups/wandb_offline/`, 2.2 MB);
  synced to W&B and untracked (2026-10-11); still in history until rewritten with `git filter-repo`.
