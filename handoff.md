# Handoff — Belief-Based POMDP Manipulation in skrl

## Status

**Branch:** `feature/belief-pomdp-manipulation` (created from `main`, clean tree).
**State at handoff time:** planning only — no code committed yet.
**Paper:** Wirnshofer et al., *Controlling Contact-Rich Manipulation Under Partial Observability*, RSS 2020 (`https://www.roboticsproceedings.org/rss16/p023.pdf`).

## What this branch delivers

- A new public module `skrl/belief/` (torch backend) with:
  - `Belief` ABC
  - `ParticleFilter` (torch, backend-agnostic)
  - GMM / QGMM compression (Kim et al. 2017 for SE(3) orientations)
  - GMM upper-bound differential entropy (Huber et al. 2008)
- A pluggable `DynamicsBackend` interface with three impls:
  - MuJoCo (primary)
  - mjlab
  - pure-torch stub (CI / smoke tests; 2-link arm + box on table)
  - IsaacLab stub (interface only; user fills in their env)
- A new skrl agent `skrl/agents/torch/belief_dqn/` — DQN with:
  - ResNet trunk (1 + 4 ResNet blocks of 64, ELU) per paper §V-A
  - Dueling head
  - Double Q-learning
  - Soft target update
  - Mixed-precision aware
  - Input: GMM-flattened compressed belief (hefts, means, log-diag-covs)
  - Output: discrete action in `{0, ..., NC-1}`
- A new skrl multi-agent `skrl/agents/torch/motion_controllers/`:
  - eTaSL/eTC attractor controllers (paper eq. 7) as a QP via `qpth` (with `cvxpy` fallback)
  - `build_controller_suite(...)` to generate `NC = n · NG` controllers
  - Gripper open/close variants for PR2-Hammer-style tasks
- An example: `examples/contact_rich_uncertainty/peg_in_hole.py` (smoke-test pipeline)
- Tests under `tests/belief/` and `tests/agents/torch/test_belief_dqn.py`, `test_motion_controllers.py`
- Docs: `docs/source/belief.rst`, CHANGELOG entry

## What's NOT in this branch (explicit)

- Full reproduction of all five paper benchmarks. Only peg-in-hole ships; the other four (Bridge, Bunny-Gap, PR2-Hammer, Active-Localization) reuse the same building blocks but are out of scope.
- IsaacLab backend implementation — the user is setting up the IsaacSim/IsaacLab side separately and will provide pose callbacks. The branch ships the interface stub only.
- Real-robot KUKA iiwa experiment (paper §V-E). Infrastructure is in place; the user runs that.
- Sim-to-real transfer tooling.

## Information the user needs to provide for belief estimation

When the agent runs against a real / non-MuJoCo sim, the following callables
must be supplied via `cfg.belief_factory` and the `DynamicsBackend` ABC:

### From the simulation (per env step)

1. **`get_measurement()`** — joint encoder reading `q_r` (numpy or torch,
   shape `[n_r]`). The filter treats this as `N(q_r, Σ_enc)` with
   `Σ_enc = σ_nom² I`, `σ_nom = 1 deg` by default. **This is the only
   measurement the filter uses** — no cameras, no force sensors.

2. **`get_object_pose()`** — ground-truth object pose in world frame, shape
   `[7]` (xyz + quaternion) or `[4,4]` SE(3) matrix. The belief is
   read from the **particles** for visualisation, not from this. This
   callback is only used by `DynamicsBackend.step()` to drive the sim
   forward, and (optionally) by reward shaping.

3. **`step(control, n_substeps)`** — applies the impedance-controller
   reference for `n_substeps` inner ticks and returns the new `q_r` and
   object pose. `n_substeps = 100` when the high-level policy runs at
   2 Hz over a 200 Hz controller.

### From the user at agent construction

4. **`prior_factory(task_name)`** — returns a callable that, given a
   random seed, produces the initial particle set `X_0` of shape
   `[NP, nq + nv]`. The paper uses a coarse multi-modal prior (Fig. 1).

5. **`target_pose_templates(task_name)`** — list of `n` SE(3) target
   poses `t_j` placed around the object. Hand-engineered (paper §IV-B).
   Example for peg-in-hole: 4 approach angles × 1 insertion axis.

6. **`goal_distance(x, x_goal)`** — function used by the termination
   criterion. The paper uses Euclidean distance for `x ∈ ℝ^n` (eq. 8).

### From the user at runtime (read-only, for viz)

7. **`belief.get_particles()`** — shape `[NP, nq + nv]`. The user's
   sim-side viz can call this every render frame to overlay particle
   clouds and GMM ellipsoids. The agent does not gate on this.

## Architecture summary

```
[env: isaaclab/mjlab/mujoco/stub]          [skrl: belief_dqn agent]
        │                                          │
        │  obs (joint encoders)                    │  owns:
        │  object pose (gt, for step)              │    Belief
        │  ───────────────────────────────────────► │      └─ ParticleFilter
        │                                          │      └─ compress() → b̃
        │  reward, terminated, truncated           │
        │  ◄─────────────────────────────────────── │
        │                                          │  Q(b̃, a) → a*
        │  step(action, n_substeps)                │  passes a to controller suite
        │  ◄─────────────────────────────────────── │  which runs n_substeps of
        │                                          │  impedance QP
```

The agent owns the filter and the Q-network. The env owns the sim
backend. The boundary between them is the joint-encoder observation —
the same boundary the paper uses.

## How to extend to the other four benchmarks

All four reuse the same building blocks:

- **Bridge** — drop in a `bridge_backend`, narrow-passage `target_pose_templates`,
  higher `NC` (paper uses 20), increase `particle_count` to 500 to handle
  the narrow-passage multi-modality.
- **Bunny-Gap** — non-convex object, need a more accurate contact-dyn
  backend; same `belief_dqn` agent.
- **PR2-Hammer** — set `gripper_variants=True` in the controller registry;
  paper NC = 16.
- **Active-Localization** — use the entropy-based termination
  (eq. 9 of the paper) from `skrl/belief/torch/entropy.py`; constant
  negative reward, episode abort at 50 actions.

## Open questions deferred to implementation

- Exact EM initialisation for the QGMM (paper uses k-medoids, fine for
  NG=2; revisit if NG>4).
- Resampling scheme (systematic vs. multinomial). The paper does not
  specify; we default to systematic.
- Particle starvation guard (paper §V-D suggests adding a penalty; not
  implemented in this iteration).

## File pointers (relative to repo root)

- Plan: `~/.claude/plans/agile-wandering-lantern.md`
- Tasks: `./tasks.md`
- (this file): `./handoff.md`
- Branch: `feature/belief-pomdp-manipulation`

## Next action for whoever picks this up

Start with §1 and §2 of `tasks.md` — `Belief` ABC, `ParticleFilter`,
stub `DynamicsBackend`. The stub unblocks tests; the real backends
land in parallel. Do **not** start the `BeliefDQN` agent until
`compress()` and the stub backend are green — the agent's input shape
depends on the compression layout.
