# HANDOFF — Using the Belief-Based skrl Stack in Isaac Lab Tasks

How to take **any** Isaac Lab task and run a belief-based (POMDP) agent from
this fork (`feature/belief-pomdp-manipulation`, base = PyPI `skrl==2.1.0`).
Companion: [`BELIEF_TRANSFER.md`](./BELIEF_TRANSFER.md) (what is being ported
in from the `Belief-based-pushing` repo and why).

Reference paper: Wirnshofer et al., RSS 2020 — particle filter over object
pose → compress to a small GMM → policy acts on the compressed belief b̃,
never on ground-truth object state.

---

## 1. Prerequisites

```bash
conda activate env_isaaclab            # your Isaac Lab env
pip install -e ~/Projects/skrl         # editable install of THIS branch (do not pip-install PyPI skrl on top)
python -c "import skrl; print(skrl.__version__)"          # 2.1.0 (+ belief modules)
pytest ~/Projects/skrl/tests/belief -q                   # sanity: filter + compression green
```

The 60-second API tour (`skrl/belief/torch/`):

```python
from skrl.belief.torch.particle_filter import ParticleFilter
from skrl.belief.torch.compression import gmm_fit, flatten_gmm, belief_dim, GMMState
from skrl.belief.torch.entropy import gmm_entropy_upper_bound

f = ParticleFilter(
    num_particles=300,             # NP (paper: 300)
    state_dim=3,                   # e.g. object [x, y, yaw]
    num_gmm_components=2,          # NG (paper: 2)
    motion_model_fn=lambda particles, control: ...,        # [NP,dim],[act] -> [NP,dim]
    measurement_model_fn=lambda particles, meas: ...,      # [NP,dim],[obs] -> [NP] log p(z|x)
    device="cuda:0",
)
f.reset(prior_particles)           # [NP, dim], weights uniform
f.predict(action); f.update(raw_obs_visible_dims); f.compress()
b_tilde = f.get_compressed()       # flattened GMM: [hefts | means | log_diag_covs]
particles = f.get_particles(); w = f.get_weights().exp(); H = f.get_entropy()
```

Belief input size for the network: `belief_dim(NG, state_dim) = NG + 2·NG·state_dim`.

---

## 2. Pick one of three integration patterns

| | A. Wrapper + stock agent | B. Env-side batched filter | C. Agent-owned filter |
|---|---|---|---|
| Parallel Isaac envs (`num_envs>1`) | ✖ (one filter instance) | ✔ best (GPU-batched `[N, NP, d]`) | ✖ |
| Code you write today | ~150 lines, works now | port `mdp/belief.py` (~650 lines) | fill backend callables |
| Agent | any stock `PPO`/`DQN` | any stock `PPO`/`DQN` | `BeliefDQN` (`BeliefPPO` pending) |
| Fidelity vs paper | approximate | approximate | closest |

Rule of thumb: **single-env / quick research → A**; **massively parallel
training → B**; **paper reproduction / real robot → C**.

---

## 3. Pattern A — belief observation wrapper + stock skrl agent (works today)

Proven end-to-end on CartPole: `examples/contact_rich_uncertainty/cartpole_pomdp_belief_ppo.py`
and `..._dqn.py`. Adaptation skeleton for an Isaac Lab task (manager-based
Franka push shown; same shape for any task):

> Keep the task's **full observation enabled** (do NOT delete object terms in
> the env config). Your wrapper consumes full obs for the filter and hands the
> policy ONLY b̃ — masking happens here, once, in one place.

```python
import gymnasium, torch, numpy as np
import isaaclab_tasks                         # registers Isaac-*-v0 IDs
from skrl.belief.torch.particle_filter import ParticleFilter
from skrl.belief.torch.compression import flatten_gmm

HIDDEN = [...]                                # indices of GT object dims in the flat obs
STATE_DIM = 3                                 # belief over object [x, y, yaw]

def make_motion_model(process_noise: float = 0.003):
    def motion(particles, control):           # [NP, dim], unused control for free objects
        return particles + process_noise * torch.randn_like(particles)
    return motion

def make_measurement_model():
    def measurement(particles, obs):          # obs = VISIBLE dims only, single env
        # example: distance-based contact likelihood
        d = torch.linalg.norm(particles[:, :2] - obs_ee_xy(obs), dim=-1)
        return -0.5 * (d / 0.03).square()
    return measurement

class BeliefObservation(gymnasium.Wrapper):
    """full obs in -> compressed particle-filter belief out."""
    def __init__(self, env, nparticles=300, ng=2):
        super().__init__(env)
        core = env.unwrapped                  # ManagerBasedRLEnf -> .scene entities
        self._belief = ParticleFilter(
            num_particles=nparticles, state_dim=STATE_DIM, num_gmm_components=ng,
            motion_model_fn=make_motion_model(None),
            measurement_model_fn=make_measurement_model(),
            device=env.device)
        self.observation_space = gymnasium.spaces.Box(
            -np.inf, np.inf, (ng + 2 * ng * STATE_DIM,), np.float32)

    def _visible(self, obs_t):
        keep = [i for i in range(obs_t.shape[-1]) if i not in HIDDEN]
        return obs_t[..., keep]

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        prior = self._sample_prior(info)      # NEVER from GT object pose
        self._belief.reset(prior)
        self._belief.compress()
        return flatten_gmm(self._belief.gmm).unsqueeze(0), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._belief.predict(action.flatten())          # motion uses chosen action
        self._belief.update(self._visible(obs)[0])
        self._belief.compress()
        return flatten_gmm(self._belief.gmm).unsqueeze(0), reward, terminated, truncated, info

    # viz hook — call every render frame
    def belief_viz(self):
        return self._belief.get_particles(), self._belief.get_weights().exp(), self._belief.gmm
```

Then train with a **completely stock** skrl agent — copy any
`examples/isaaclab/torch_*_direct_*_ppo.py` and change only the spaces/models:

```python
from skrl.agents.torch.ppo import PPO, PPO_CFG
from skrl.envs.wrappers.torch import wrap_env
from skrl.trainers.torch import SequentialTrainer

env = gym.make("Isaac-Push-Cube-Franka-v0", ...)
env = BeliefObservation(env)
env  = wrap_env(env)                                   # plain-gymnasium path is fine here

cfg = PPO_CFG(rollouts=16, learning_epochs=8, mini_batches=2,
              discount_factor=0.99, gae_lambda=0.95,
              learning_rate=5e-4, grad_norm_clip=0.5,
              entropy_loss_scale=0.004,
              rewards_shaper=None)
# models: copy from the isaaclab PPO example; input size = belief_dim(2, STATE_DIM)
agent = PPO(models={"policy": policy, "value": value},
            memory=torch_memory, observation_space=env.observation_space,
            action_space=env.action_space, device=env.device, cfg=cfg)

trainer = SequentialTrainer(cfg={"timesteps": 100_000, "headless": True}, env=env, agents=agent)
trainer.train()
```

DQN variant: swap in `BeliefDQN` **or** stock `DQN` — with the wrapper both
work identically because the belief is already the observation.

Baseline for comparison: same env wrapped with
`skrl.envs.wrappers.torch.pomdp_wrapper.PomdpWrapper(obs_indices=HIDDEN)`
(masked raw obs, no belief) — this is exactly the A/B the CartPole example ran
(`--baseline` flag), including heavy/light-pole sweeps under
`examples/contact_rich_uncertainty/media/`.

---

## 4. Pattern B — env-side batched filter (for `num_envs` in the hundreds/thousands)

This is what the source repo runs in production. Transfer source:
`~/Projects/ebasa/Belief-based-pushing/source/franka_cube_push_project/.../cube_push/mdp/belief.py`
(full algorithm spec: its `HANDOFF.md` §2).

Steps inside YOUR task package (no skrl agent changes needed):

1. Copy `belief.py` next to your task's `mdp/` (it imports only
   `isaaclab.assets/managers/sensors`; retarget the three `SceneEntityCfg`s +
   workspace clamps to your scene).
2. OmniReset-style starts + belief priors at reset:
   - Object placement: use the shared mixture sampler via
     `skrl.belief.torch.isaaclab_omnireset.reset_object_pose_omnireset`
     as an `EventTerm(mode="reset", params={"object_cfg": SceneEntityCfg(...),
     "cfg": OmniResetCfg(goal_xy=...)})`. This reproduces the source repo's
     reaching/corridor/near-goal reset mixture (0.35/0.30/0.35) and stores
     mode ids on `env._skrl_omnireset_mode_ids` (log with
     `omnireset_mode_fractions(env)`).
   - Belief prior: add a second reset event calling
     `reset_planar_particle_belief` (bimodal task-knowledge prior) or
     `reset_planar_particle_belief_around_cube` (local+secondary+uniform
     mixture — recommended; pairs with omni-reset placement).
   - Do NOT try to do this with `OmniResetWrapper` here: Isaac Lab vectorized
     envs auto-reset internally, so wrapper-level resets fire only once. The
     wrapper (`skrl/envs/wrappers/torch/omnireset_wrapper.py`) is for
     single-env/gymnasium/MuJoCo-style tasks where Python sees every reset.
3. Add the observation term:
   ```python
   compressed_belief = ObsTerm(func=my_mdp.compressed_planar_belief, params={...})
   ```
   (params dict = the tuned defaults in the source repo's
   `cube_push_env_cfg.py:190`; output is `[N, 24]`).
4. Delete ground-truth object terms from the **policy** obs group; optionally
   clone them into a separate `critic` group (privileged asymmetric training —
   rsl_rl supports this natively; in skrl just feed the critic model the
   privileged tensors via `states`).
5. Train with stock `PPO`/`DQN` exactly like any Isaac Lab example.

Why this scales: filter state is `[num_envs, NP, 2]` tensors updated once per
step, fully batched on GPU; the env simply exposes a bigger obs vector.

---

## 5. Pattern C — agent-owned filter (`BeliefDQN` + `DynamicsBackend`)

Paper-faithful: the AGENT owns the filter; the env only supplies measurements.

```python
from skrl.belief.backends.isaaclab_backend import IsaacLabBackend
from skrl.belief.torch.particle_filter import ParticleFilter
from skrl.agents.torch.belief_dqn import BeliefDQN, BeliefDQN_CFG

backend = IsaacLabBackend(
    step_fn=...,           # propagate NP ghost states n_substeps (model-based rollouts)
    obs_fn=...,            # log N(q_r; sigma_enc=1deg) per particle from real env joints
    pose_fn=...,           # GT object pose [7] (drives sim / reward only)
    robot_state_fn=..., reset_fn=..., state_dim=nq + nv, device="cuda:0")

def belief_factory():
    return ParticleFilter(num_particles=300, state_dim=backend.state_dim,
                          num_gmm_components=2,
                          motion_model_fn=backend.step,
                          measurement_model_fn=backend.get_measurement,
                          device="cuda:0")

cfg = BeliefDQN_CFG(belief_factory=belief_factory,
                    discount_factor=0.98, polyak=0.005, learning_rate=1e-4,
                    l2_reg=0.01, batch_size=64, exploration_scheduler=eps_sched)
agent = BeliefDQN(models={"q_network": qnet, "target_q_network": target_qnet},
                  memory=RandomMemory(memory_size=100_000, device="cuda:0"),
                  observation_space=env.observation_space,     # raw joint encoders
                  action_space=gymnasium.spaces.Discrete(NC), device="cuda:0", cfg=cfg)
```

Lifecycle handled by `BeliefAgentMixin`: act→compress b̃→Q(b̃);
record_transition→predict(action)/update(next_obs)/compress; replay stores
`(b̃, a, r, b̃′, done)`; TB tracks `Belief/entropy`, `Belief/compress_ms`.
Reset each episode with `agent.reset_belief(prior_particles)`.

Single-env only. For `n_substeps`: high-level 2 Hz over 200 Hz controller ⇒ 100.
On-policy twin (`BeliefPPO`) is CFG-only today — until T-3 lands, use Pattern A
with stock `PPO` (identical math, wrapper-side belief).

---

## 6. Visualizing the belief in Isaac Sim

Port (verbatim-safe) `visualization/belief_markers.py` from the source repo —
depends only on `omni.usd` + `pxr`:

```python
viz = BeliefCloudUsdVisualizer(particle_count=120, cube_side_length=0.0406)
# every render frame, from ANY pattern:
particles, weights, _ = env.belief_viz()                 # A/C
viz.set_particles(sample_posterior(particles, weights, 120))   # yellow cubes
viz.set_mode_outlines(torch.stack([gmm.means[0], gmm.means[1]], 0))  # orange outlines
```

Use **posterior-weighted sampling** for display (raw clouds look static
because low-weight stragglers linger). Record MP4s with the source repo's
`scripts/run_belief_demo_franka.py --record` pattern (`outputs/recordings/`),
and always log scalars: belief entropy, ESS ratio, contact signal,
belief-error vs GT (`scripts/analyze_belief_rollout.py` shows the report
format under `outputs/diagnostics/`). Non-viewport fallback: matplotlib
scatter colored by weights + GMM ellipses from `GMMState`.

---

## 7. Gotchas (each one has bitten before)

1. **Log-variance convention**: b̃ covariances are `log var` everywhere
   (`flatten_gmm` AND the source repo's 24-dim vector). Don't feed exp'd
   values into the net or compare checkpoints across conventions.
2. **Normalizer travels with DQN checkpoints**: fit mean/std from replay once,
   save INSIDE the checkpoint, reload verbatim at play time (source repo's
   `ObservationNormalizer` pattern).
3. Observation space changes when you add the wrapper — rebuild models with
   `belief_dim(NG, state_dim)`, not the env's raw obs size.
4. Filter priors must NOT read ground truth (bimodal/mixture priors exist for
   this). GT is allowed ONLY for physics/reward/critic/viz.
5. Encoder noise std = 1 degree IN RADIANS (0.0174533) wherever the paper's
   sensor model applies.
6. `PomdpWrapper` masks by zeroing + optional noise — fine for baselines, but
   zeroed dims are still "known zeros"; the belief wrapper (A) or env-side
   compression (B) is the honest POMDP.
7. Multi-env + Pattern A silently filters only ONE trajectory — if
   `num_envs > 1`, switch to Pattern B.
8. After editing anything in `skrl/belief`, rerun
   `pytest tests/belief tests/agents/torch/test_belief_dqn.py -q` and
   `python verify_belief_compression.py`.

---

## 8. Running inside Isaac Lab's registered-env YAML flow (`isaaclab.sh train`)

Isaac Lab's default skrl workflow is: task registered via `gym.register(...,
skrl_cfg_entry_point="...:skrl_ppo_cfg.yaml")` → `./isaaclab.sh train
--rl_library skrl --task <TASK>` → `SkrlVecEnvWrapper(env)` →
`skrl.utils.runner.torch.Runner(env, agent_cfg_from_yaml)` → `runner.run()`.
This flow works with the belief system **unchanged**, because of one verified
fact about the bridge:

> `IsaacLabWrapper` (in this fork, `skrl/envs/wrappers/torch/isaaclab_envs.py`)
> reads obs **groups by name**: `observations["policy"]` becomes the agent's
> observation, and an optional `"critic"` group becomes the agent's privileged
> `states` (`state_space = single_observation_space["critic"]`). So the
> asymmetric belief setup maps 1:1 onto Isaac Lab's group mechanism.

Where the filter lives in this flow: **Pattern B** (filter inside the env,
belief vector as a "policy"-group obs term) trains with stock YAML PPO as-is
and scales to many envs. Since 2026-10-11 the YAML `Runner` also builds
`BeliefPPO` / `BeliefDQN` (Pattern C, agent-owned filter, `num_envs = 1`) and
`PrivilegedPPO` (history estimator); see §8.7.

### 8.1 Observation groups (the part you asked about)

In a manager-based env cfg, group names = attribute names on your
`ObservationsCfg`. Reuse the proven groups from the source repo
(`cube_push_env_cfg.py`, `BeliefAsymmetricObservationsCfg`):

```python
@configclass
class BeliefObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):                    # -> observations["policy"]
        actions         = ObsTerm(func=mdp.last_action)
        noisy_joint_pos = ObsTerm(func=mdp.noisy_joint_pos, params={"std": 0.0174533})
        joint_vel       = ObsTerm(func=mdp.joint_vel_rel)
        ee_position     = ObsTerm(func=mdp.ee_frame_pos)
        compressed_belief = ObsTerm(func=mdp.compressed_planar_belief, params={...})   # 24-dim b̃
        ee_to_belief    = ObsTerm(func=mdp.ee_to_belief_mean, params={"num_particles": 256})
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    policy: PolicyCfg = PolicyCfg()               # NO ground-truth object pose here

    @configclass
    class CriticCfg(ObsGroup):                    # -> observations["critic"] (privileged)
        joint_pos       = ObsTerm(func=mdp.joint_pos_rel)
        red_cube_pose   = ObsTerm(func=mdp.red_cube_pose)      # GT allowed: critic-only
        red_cube_vel    = ObsTerm(func=mdp.red_cube_velocity)
        goal_position   = ObsTerm(func=mdp.goal_position)
        cube_to_goal    = ObsTerm(func=mdp.cube_to_goal)
        ee_position     = ObsTerm(func=mdp.ee_frame_pos)
        ...
    critic: CriticCfg = CriticCfg()
```

Rules that make it work end-to-end:

1. Group names MUST be exactly `policy` and `critic` (the wrapper looks those up).
2. Actor terms never read ground truth; all GT terms go in `critic`.
3. `concatenate_terms = True` per group; each group flattens to one Box space.
4. The particle-filter update happens inside the `compressed_planar_belief`
   term call (once per step, guarded by `common_step_counter`) — no trainer
   involvement; auto-resets are handled by the omni-reset EventTerm + prior
   event from §4.
5. If you only need symmetric training, omit the `critic` group entirely:
   `state()` returns `None` and every model just uses `OBSERVATIONS`.

### 8.2 Register the task with a skrl entry point

In your task package `__init__.py` (same file that registers `-v0` IDs):

```python
import os
agents_dir = os.path.join(os.path.dirname(__file__), "agents")

gym.register(
    id="Isaac-Belief-Push-Cube-Franka-IK-HighLevel-OmniReset-PPO-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cube_push_env_cfg:FrankaCubePushBeliefIKHighLevelOmniResetPPOEnvCfg",
        "skrl_cfg_entry_point": f"{agents_dir}:skrl_belief_ppo_cfg.yaml",   # <- add this
    },
)
```

(Isaac Lab's own registry uses `"skrl_cfg_entry_point": f"{__name__}:skrl_ppo_cfg.yaml"`.)

### 8.3 The YAML (`agents/skrl_belief_ppo_cfg.yaml`)

Same schema as any Isaac Lab skrl yaml; two belief-specific changes: input
sizes match the belief-augmented policy group, and the value net consumes the
privileged group via `input: STATES`:

```yaml
seed: 42

models:
  separate: False
  policy:
    class: GaussianMixin
    clip_actions: false
    clip_log_std: true
    min_log_std: -20.0
    max_log_std: 2.0
    initial_log_std: 0.0
    network:
      - name: net
        input: OBSERVATIONS          # flattened "policy" group (belief + proprioception)
        layers: [256, 128, 64]
        activations: elu
    output: ACTIONS
  value:
    class: DeterministicMixin
    clip_actions: false
    network:
      - name: net
        input: STATES                # flattened "critic" group (privileged full state)
        layers: [256, 128, 64]
        activations: elu
    output: ONE

memory:
  class: RandomMemory
  memory_size: -1

agent:
  class: PPO                         # stock agents only in the Runner (by design here)
  rollouts: 32
  learning_epochs: 5
  mini_batches: 4
  discount_factor: 0.99
  lambda: 0.95
  learning_rate: 5.0e-4
  learning_rate_scheduler: KLAdaptiveLR
  learning_rate_scheduler_kwargs:
    kl_threshold: 0.01
  state_preprocessor: RunningStandardScaler
  state_preprocessor_kwargs:
    size: null                       # derive from env
  grad_norm_clip: 1.0
  ratio_clip: 0.2
  value_clip: 0.2
  clip_predicted_values: true
  entropy_loss_scale: 0.004
  value_loss_scale: 2.0
  time_limit_bootstrap: false
  experiment:
    directory: "franka_belief_push_skrl"
    experiment_name: ""
    write_interval: auto
    checkpoint_interval: auto

trainer:
  class: SequentialTrainer
  timesteps: 16000                   # max_iterations * rollouts
  environment_info: log
```

Hyperparameters mirror the source repo's rsl_rl config (`PPORunnerCfg`) for
comparability. For the discrete DQN variant swap the models to
CategoricalMixin over the same `OBSERVATIONS` input and set
`agent.class: DQN`.

### 8.4 Run / play

```bash
conda activate env_isaaclab && cd ~/Projects/IsaacLab
# training (YAML picked up automatically via skrl_cfg_entry_point):
./isaaclab.sh train --rl_library skrl --task Isaac-Belief-Push-Cube-Franka-IK-HighLevel-OmniReset-PPO-v0 \
    --headless --num_envs 1024 --max_iterations 500
# play:
./isaaclab.sh play  --rl_library skrl --task <TASK>-Play-v0 \
    --checkpoint logs/skrl/franka_belief_push_skrl/<run>/checkpoints/agent_<N>.pt
# (the deprecated-but-working script path: scripts/reinforcement_learning/skrl/train.py;
#  it pins SKRL_VERSION >= 2.1.0 — exactly this branch's base)
```

Logs → `logs/skrl/<experiment.directory>/<timestamp>_ppo_torch_.../`
(`params/env.yaml`, `params/agent.yaml`, TensorBoard, checkpoints).

### 8.5 Belief viz inside this flow

The filter lives in the env, so visualize from env state, not the agent:
during a play loop read `env.unwrapped._rss20_planar_belief` (or the
portable helpers `belief_posterior_samples_world` /
`belief_mode_centers_world`) each render tick and feed the ported
`BeliefCloudUsdVisualizer` (§6). Mode ids from omni-reset land on
`env.unwrapped._skrl_omnireset_mode_ids` for per-mode success breakdowns.

### 8.6 Flow-specific gotchas

- `agent.class: BeliefPPO` / `BeliefDQN` work in the `Runner` (§8.7), but the
  agent tracks one filter: run them with `--num_envs 1`. For many envs keep
  the filter in the env (Pattern B) and use stock PPO.
- The value net's `STATES` input exists ONLY if the env defines a `critic`
  group; otherwise set `input: OBSERVATIONS`.
- `state_preprocessor` normalizes the privileged states; keep it enabled —
  raw cube pose magnitudes differ wildly from proprioception scales.
- Isaac Lab auto-resets internally, so `OmniResetWrapper` (§3) must NOT be
  stacked on registered envs — use the `reset_object_pose_omnireset`
  EventTerm (§4). Wrapper-level resets fire once at startup only.

### 8.7 Belief and privileged agents from YAML (`Runner` extensions)

`skrl/utils/runner/torch/runner.py` now knows `PrivilegedPPO`, `BeliefPPO`
and `BeliefDQN` and their models (`PrivilegedGaussianPolicy`,
`PrivilegedCategoricalPolicy`, `PrivilegedValue`, `ResNetCategoricalPolicy`,
`ResNetValue`, `ResNetDuelingQNetwork`). Any model class can also be named
as `"package.module:Class"`. Example configs:
`examples/privileged_curriculum/isaaclab/skrl_privileged_ppo_cfg.yaml` and
`skrl_belief_ppo_cfg.yaml`.

Register and run them like any skrl YAML, selected with `--agent <entry point
key>`. Isaac Lab 3.0's `--algorithm` accepts only `AMP`/`PPO`/`IPPO`/`MAPPO`, so
use `--agent`; the log directory is named after `agent.class`
(`privilegedppo`):

```python
gym.register(id="Isaac-<Task>-v0", ..., kwargs={
    "env_cfg_entry_point": ...,
    "skrl_privileged_ppo_cfg_entry_point": f"{agents.__name__}:skrl_privileged_ppo_cfg.yaml",
    "skrl_belief_ppo_cfg_entry_point": f"{agents.__name__}:skrl_belief_ppo_cfg.yaml",
})
```

```bash
python scripts/reinforcement_learning/skrl/train.py --task Isaac-<Task>-v0 \
    --agent skrl_privileged_ppo_cfg_entry_point --headless
python scripts/reinforcement_learning/skrl/train.py --task Isaac-<Task>-v0 --num_envs 1 \
    --agent skrl_belief_ppo_cfg_entry_point --headless
```

As a module in your own script: `Runner(SkrlVecEnvWrapper(env), Runner.load_cfg_from_yaml(path)).run()`.
A `privileged_fn` can also be a Python callable put into the loaded dict.

Isaac Lab smoke run: `examples/privileged_curriculum/isaaclab/cartpole_privileged_smoke.py`
(direct Cartpole with the joint velocities hidden and used as privileged
values; YAML resolved from the gym registry like `train.py --agent`; 3-panel
debug video with the mask state). Run it on zz-bw with `zzbw_smoke.sh` (pe SIF,
Isaac Lab 3.0, this fork on `PYTHONPATH`), never on the laptop (an Isaac Lab
start crashed it).

What the YAML selects:

| Key | Values |
|---|---|
| `agent.strategy` (PrivilegedPPO) | `roa`, `roa_lam0`, `sysid`, `contrastive`, `whiten`, `osi`, `rma`, `uposi` (`skrl.agents.torch.privileged_ppo.STRATEGIES`); explicit keys override the preset |
| `agent.curriculum` | `{class: SoftmaxMaskCurriculum \| RewardGatedMaskCurriculum \| FixedMaskCurriculum \| TwoPhaseMaskCurriculum, ...kwargs}` |
| `models.policy.estimator` | history estimator: `mlp`, `conv1d`, `lstm`, `transformer` (causal); `latent_dim`, `hidden` |
| `privileged_history` | wraps the env in `PrivilegedHistoryVecWrapper` (GPU, all envs): `history_length`, `include_last_action`, `privileged_indices`, `privileged_fn: "module:function"`; privileged values default to the `critic` group |
| `belief` (BeliefPPO/DQN) | particle filter: `class` (`ParticleFilter` or `"module:Class"`), `num_particles`, `state_dim`, `num_gmm_components`, `n_substeps`, `sigma_enc`, `backend: {class: "module:Backend", ...}` (gets `env` / `device` if its signature has them) |
| `agent.belief_factory`, `belief_prior_fn`, `belief_true_state_fn`, `belief_state_to_position` | `"module:function"` import strings |
| belief `models` | `ResNet*` over the compressed belief b̃ (input size from the filter, or `agent.belief_dim`), or a custom `"module:Class"` |

Limits: LSTM / transformer apply to the history estimator (the encoder of
the privileged values and the belief b̃ have no time axis, so they are
MLP / ResNet). `observation_preprocessor` must stay `null` for PrivilegedPPO
(it would rescale the mask flag). Tests:
`tests/utils/test_runner_privileged_belief.py` (fake Isaac Lab env with
same-step auto-reset and a `critic` state).

---

## 9. Verification checklist (definition of "it works")

- [ ] `tests/belief`, `test_belief_dqn.py` green.
- [ ] CartPole smoke proof reproduces: belief-PPO beats `--baseline` PPO on
      hidden-dim variants (see `media/reward_curves.png`).
- [ ] Isaac Lab task: policy obs contains NO ground-truth object dims
      (grep your obs group); training curves improve vs PomdpWrapper baseline.
- [ ] TensorBoard: `Belief/entropy` decays on successful episodes;
      ESS ratio oscillates around the resample threshold, not pinned at 1.
- [ ] Viewport overlay: yellow posterior cloud converges to the true object
      within a few contacts; orange outlines sit on the two modes.
