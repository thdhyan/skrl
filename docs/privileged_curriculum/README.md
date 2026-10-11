# PrivilegedPPO: privileged-observation masking curriculum

Code: `skrl/agents/torch/privileged_ppo/`, `skrl/envs/wrappers/torch/privileged_history_wrapper.py`,
example `examples/privileged_curriculum/cartpole_privileged_ppo.py`, tests `tests/agents/torch/test_privileged_ppo.py`.

The actor acts on an encoder latent `z` of the privileged values in unmasked episodes and on a history-estimator
latent `ẑ` in masked ones. The critic always sees the privileged values. `SoftmaxMaskCurriculum` ramps the
per-episode mask probability from 0 to 1. After each PPO update an extra pass minimizes
`||sg(z) − ẑ||² + 0.1·||z − sg(ẑ)||²`. Pipeline diagram: `diagrams/01_pipeline.dot`.

## Benchmark (2026-10-10)

Task: CartPole-v1 POMDP. Cart velocity and pole angular velocity are hidden, observation noise std 0.02, history
of 8 observations. Pole mass and length are randomized per episode. Privileged values: both velocities, pole mass
and pole length.

Settings: 300k env steps (8 envs), rollouts 128, 8 epochs, 4 mini-batches, lr 1e-3, 3 seeds (42, 43, 44). Curriculum:
`start=0.1, end=0.7, sharpness=10`. Evaluation: 20 deterministic episodes per seed, with the privileged values
masked (deployment setting) or visible. Max return is 500. About 5–7 min per run on the laptop, with 8 runs in parallel.

| Training mode | Eval masked (seeds 42/43/44) | Mean masked | Eval visible | Mean visible |
|---|---|---|---|---|
| curriculum (visible → masked) | 500.0 / 500.0 / 480.2 | 493.4 | 500 / 500 / 500 | 500 |
| masked from start (history only) | 500.0 / 468.6 / 500.0 | 489.5 | 500 / 500 / 500 | 500 |
| always visible | 105.2 / 142.1 / 107.3 | 118.2 | 500 / 500 / 500 | 500 |

![sweep](diagrams/02_sweep.png)

Regenerate the plot with
`python docs/privileged_curriculum/plot.py runs/privileged_curriculum docs/privileged_curriculum/diagrams/02_sweep.png`.

What the results show:

- The curriculum works as designed: at the end of training the deployable (masked) actor reaches 480–500.
- The always-visible policy fails when masked (about 118). Its estimator is still trained by the auxiliary loss,
  but the actor never learns to act on `ẑ`, so the masking curriculum is required for deployment.
- On this task the curriculum is **not** better than training masked from the start (493 vs 490 masked, within
  seed noise). The history of 8 noisy observations is enough to recover the hidden velocities, and 300k steps
  are enough for history-only PPO. The plot shows the visible policy learning fastest (it reaches 500 at about 10k
  trainer steps). The curriculum curve stays close to it while the mask probability is low, then becomes noisy and
  drops (200–500) while the mask probability ramps and the actor switches to `ẑ`. The estimator loss of the
  curriculum runs only falls once most episodes are masked (after about 25k trainer steps).
- To show a benefit from the curriculum, a harder task is needed: one where the privileged values cannot be
  recovered cheaply from a short history (for example contact or friction parameters), or a smaller training
  budget. 3 seeds is also too few to separate 493 from 490.

Raw logs: `runs/bench_logs/priv_*.log`; TensorBoard runs: `runs/privileged_curriculum/<mode>_seed<seed>/` (not committed).
