# POMDP masking and observation groups: effect on training

Two controlled experiments run on CPU (Python 3.13, torch 2.14, wandb 0.30), logged to Weights & Biases in
**offline** mode (the cloud container could not reach `api.wandb.ai`). The offline runs are in `wandb_offline/`; upload
them with:

```bash
wandb login
wandb sync --project skrl-cartpole-pomdp  experiments/pomdp_and_obs_groups/wandb_offline/cartpole/offline-run-*
wandb sync --project skrl-pursuit-evasion experiments/pomdp_and_obs_groups/wandb_offline/pursuit/offline-run-*
```

Runs are grouped by condition (`group` = `full_obs` / `masked` / `masked_belief` and `obs_ablation_<mode>`), so the
W&B "group by" view gives mean ± spread curves directly. `results.json` holds the raw curves, `analyze.py` prints the
tables below (run it next to the `cartpole/` and `pursuit/` run directories that `run_cartpole.sh` and
`run_pursuit.sh` create).

## 1. CartPole POMDP: masking vs belief (PPO, 30k steps, 3 seeds)

`examples/contact_rich_uncertainty/cartpole_pomdp_belief_ppo.py`. Identical PPO hyper-parameters; `PomdpWrapper` with
sensor noise 0.02 in every condition.

| Condition | Policy input | Final return | Mean return over training | First step with return >= 195 |
|---|---|---|---|---|
| `full_obs` | raw obs, nothing hidden | **500.0 ± 0.0** | 378.5 ± 15.6 | 4.2k / 5.1k / 2.4k |
| `masked` | raw obs, cart velocity + pole angular velocity zeroed | 48.6 ± 2.2 | 49.5 ± 1.0 | never (best 107) |
| `masked_belief` | GMM-compressed particle-filter belief over the masked obs | 276.2 ± 58.8 | 246.8 ± 13.4 | 3.0k / 3.0k / 3.6k |

*Final* = mean of the last 10 % of logged points. W&B metric: `Reward / Total reward (mean)`.

* Masking the two velocities makes the task unlearnable for a memoryless PPO policy: it plateaus at ~50 because it
  cannot tell which way the pole is moving.
* The belief restores most of the performance: it crosses 195 as fast as full observation and reaches 500 in every
  seed, but is less stable late in training (final 276 ± 59 vs 500). The filter's motion-model noise and the
  2-component GMM compression lose information that full observation has.

## 2. Pursuit-evasion observation groups (MAPPO, 2 seeds)

`examples/pursuit_evasion/torch_pursuit_evasion_selfplay.py`: 12x12 grid, 2 pursuers, capture on the same cell,
50-step episodes, 32 envs, 6 generations x 800 steps. Only the pursuers learn (`--learning-teams pursuers`) against
the frozen random evader, so `obs_mode` is the only variable.

| `obs_mode` | Obs size | Capture rate in training, gen 1 -> 6 | Eval capture, gen 1 -> 6 | s / gen |
|---|---|---|---|---|
| `full_state` | 18 | 0.41, 0.72, 0.96, 1.00, 1.00, **1.00** | 0.72, then 1.00 from gen 2 | 4 |
| `global_potential` | 720 (5 x 12 x 12) | 0.36, 0.40, 0.50, 0.61, 0.62, 0.64 | 0.31 -> 0.98 | 6 |
| `local_potential` (partial obs) | 104 (4 x 5 x 5 + 4) | 0.31, 0.33, 0.38, 0.44, 0.53, 0.65 | 0.38 -> 0.86 | 5 |

W&B metrics: `pursuers/train_win_rate`, `pursuers/eval_score` (evaluation against the random evader).

* `full_state` learns fastest: 18 numbers already contain everything the reward depends on.
* `global_potential` holds the same information spread over 720 inputs; with an MLP it learns about 3x slower. A CNN
  on `env.observation_image_shape` should close part of that gap.
* `local_potential` is the genuinely partially observable group (5x5 window, radius 2): slowest early on, but still
  improving steadily and close to `global_potential` by generation 6.
* Training capture rates are lower than evaluation ones because training includes exploration noise.

## Testing done alongside

* `PomdpWrapper` masking tests (`tests/envs/test_pomdp_wrapper.py`): hidden dims exactly zero, noise only on visible
  dims, `info["full_obs"]` on reset and step, multi-dimensional observations, index validation, masked tensors reaching
  skrl. They exposed (and the branch fixes) a crash on multi-dimensional observations, a wrong `visible_indices` with
  negative indices, and a missing `full_obs` on reset.
* The CartPole belief example initialised its particle prior from the noise-free ground truth; it now uses the noisy
  masked observation.
* skrl's W&B integration (`experiment.wandb=True`) logged **no training metrics**: skrl's `SummaryWriter` bound
  tensorboard's `EventFileWriter` before `wandb.init` patched it. Fixed in `skrl/utils/tensorboard.py`.

## Caveats

* Small budgets (30k steps, 6 generations) and 2-3 seeds: the ordering is clear, the exact gaps are not precise.
* The pursuit-evasion comparison trains against a random evader. In self-play the observation group also shapes the
  opponent, which this does not measure.
