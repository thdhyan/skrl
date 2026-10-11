# Transformer policies for PPO on MuJoCo tasks

PPO (skrl 2.x, PyTorch) with Transformer-based policy and value models instead of MLPs, trained on
gymnasium MuJoCo tasks: `HalfCheetah-v5`, `InvertedPendulum-v5`, `InvertedDoublePendulum-v5`.

Note on the "3 inverted pendulum" task: gymnasium has no triple-pendulum environment. The closest
registered task is `InvertedDoublePendulum-v5`, which is what the example uses.

## Architecture

Code: `skrl/models/torch/transformer.py`

- `ObsTokenizer`: each observation scalar becomes one token. A shared `Linear(1, d_model)` embeds the value,
  and a learned per-dimension embedding tells the transformer which observation the token encodes.
- `TransformerEncoderTrunk`: tokens plus a learned [CLS] token, passed through `nn.TransformerEncoder`
  (`batch_first=True`, `norm_first=True`), followed by a LayerNorm. Returns the CLS embedding `[B, d_model]`.
- `TransformerGaussianPolicy(GaussianMixin, Model)`: trunk, then a linear mean head and a state-independent
  `log_std` parameter (as in the skrl pendulum example).
- `TransformerValue(DeterministicMixin, Model)`: trunk, then a linear value head.

Defaults: `d_model=64`, `n_heads=4`, `n_layers=2`, `dim_ff=128`, `dropout=0.0`. The models are not exported
from `skrl/models/torch/__init__.py`; import them from `skrl.models.torch.transformer`.

The `mlp` baseline (`--model mlp`) uses the sizes of the skrl pendulum example: two hidden layers of 64 units
with ReLU, a linear mean head, and the same `log_std` parameter.

## How to run

From the repository root:

```bash
# Transformer policy/value, InvertedPendulum, 100k environment steps, 8 envs, eval video at the end
python examples/transformer_mujoco/torch_mujoco_transformer_ppo.py --task InvertedPendulum-v5 --model transformer --video

# MLP baseline
python examples/transformer_mujoco/torch_mujoco_transformer_ppo.py --task InvertedPendulum-v5 --model mlp --video

python examples/transformer_mujoco/torch_mujoco_transformer_ppo.py --task HalfCheetah-v5 --model transformer --timesteps 1000000
```

Options: `--task {HalfCheetah-v5,InvertedPendulum-v5,InvertedDoublePendulum-v5}`, `--model {transformer,mlp}`,
`--timesteps` (total environment steps summed over envs; the trainer counts vectorized steps internally),
`--num-envs` (default 8), `--seed` (default 42), `--video`.

Outputs go to `runs/transformer_mujoco/<task>_<model>/`:

- TensorBoard event file and `checkpoints/` (`agent_<step>.pt`, `best_agent.pt`), written by skrl's experiment config.
- `videos/<task>-episode-0.mp4` when `--video` is set. The eval rollout uses the deterministic mean action,
  clipped to the action space, and is recorded with gymnasium `RecordVideo`.

Environment notes:

- Rendering needs `MUJOCO_GL`. On the development laptop `MUJOCO_GL=egl` fails with an EGLError (even with the
  Mesa vendor forced), and `osmesa` is not functional. `glfw` works with `DISPLAY` set, so the script defaults to
  `glfw`. Override with `MUJOCO_GL=<backend>` if needed.
- `RecordVideo` needs `moviepy` (1.x API; `moviepy==1.0.3` was installed into the venv).

## Smoke-test results

Hardware: laptop with an RTX 4060 (8 GB, shared), default skrl device. Common settings: 8 envs,
`rollouts=512` per env, 10 epochs, 32 mini-batches, lr 3e-4, gamma 0.99, GAE lambda 0.95, ratio clip 0.2, seed 42.
These are short smoke runs, not converged results.

"Final mean return" is the last `Reward / Total reward (mean)` value in TensorBoard, which is the mean return of
episodes completed in the last update window (stochastic policy). "Eval return" is the single deterministic
episode recorded in the video.

| Task | Model | Env steps | Wall time | Final mean return (TB) | Last-5-logged mean | Eval return (video) |
|---|---|---|---|---|---|---|
| InvertedPendulum-v5 | transformer | 20k | ~31 s | 43.0 | 45.4 | 67.0 |
| InvertedPendulum-v5 | mlp | 20k | ~15 s | 119.0 | 69.8 | 154.0 |
| InvertedDoublePendulum-v5 | transformer | 20k | ~25 s | 82.0 | 135.4 | 127.6 |
| HalfCheetah-v5 | transformer | 50k | ~63 s | -287.3 | -249.6 | -189.5 |

Wall time includes environment creation, training and the eval video. Runs are single-seed, so the table does not
support any claim that the transformer beats the MLP. The 20k-step budget is far below what PPO needs to converge
on these tasks. HalfCheetah at 50k steps is still negative for the transformer, which is expected this early but
is not evidence either way.

Videos (in the repo tree, under `runs/`, not committed):

- `runs/transformer_mujoco/InvertedPendulum-v5_transformer/videos/InvertedPendulum-v5-episode-0.mp4`
- `runs/transformer_mujoco/InvertedPendulum-v5_mlp/videos/InvertedPendulum-v5-episode-0.mp4`
- `runs/transformer_mujoco/InvertedDoublePendulum-v5_transformer/videos/InvertedDoublePendulum-v5-episode-0.mp4`
  (only 17 frames: the episode terminates early at this training budget)
- `runs/transformer_mujoco/HalfCheetah-v5_transformer/videos/HalfCheetah-v5-episode-0.mp4`

Frame checks: the extracted PNGs show the pendulum and the cheetah clearly (not black frames).

## Not done / open items

- No hyperparameter tuning and no multi-seed runs. Results are single-seed smoke runs only.
- No comparison at convergence budgets (typically 1M+ steps for these tasks).
- No history/memory tokens: the tokenizer is one token per current observation dimension.
- The Transformer models are not wired into skrl's model instantiators or the model registry; they are used
  through the explicit classes in `skrl/models/torch/transformer.py`.
- No triple-pendulum task (see the note at the top).
