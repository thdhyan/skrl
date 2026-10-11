# SKRL Project - Development Environment

## Python Environment (uv)

The project uses a **uv** virtual environment at `.venv/` in the repo root (Python 3.10). There is no conda env;
an older version of this file named a `belief` conda env that does not exist.

### Key Packages (Installed, 2026-10-11)

| Package | Version | Notes |
|---------|---------|-------|
| Python | 3.10.20 | - |
| PyTorch | 2.14.1+cu126 | CUDA 12.6 enabled (RTX 4060 laptop, 8 GB) |
| Gymnasium | 1.4.0 | Environment interfaces |
| MuJoCo | 3.15.0 | Physics simulation (`MUJOCO_GL=egl` for headless rendering) |
| skrl | 2.1.0 (editable) | `/home/thakk100/Projects/skrl` |
| wandb | 0.30.0 | Entity `thakk100-dhyan-home`, credentials in `~/.netrc` |

### Setup and use

```bash
cd /home/thakk100/Projects/skrl
uv venv .venv --python 3.10                     # once
uv pip install -e . pytest "gymnasium[mujoco]" tensorboard matplotlib imageio imageio-ffmpeg wandb
uv pip install <package>                         # add a package (installs into .venv)
.venv/bin/python <script.py>                      # run without activating
source .venv/bin/activate                         # or activate the env
```

### Running Tests

```bash
.venv/bin/python -m pytest tests/                 # all tests
.venv/bin/python -m pytest -q tests/belief tests/agents/torch/test_belief_dqn.py tests/agents/torch/test_belief_ppo.py \
    tests/agents/torch/test_privileged_ppo.py tests/models tests/envs/test_pomdp_wrapper.py \
    tests/envs/test_pursuit_evasion.py tests/multi_agents/torch/test_selfplay.py tests/utils/test_tensorboard_writer.py
```

## Long runs on `dl`

Long sweeps run on `ssh dl` (64 CPU cores, root disk nearly full) in `~/Projects/skrl` with a CPU-only uv venv
(`.venv`, torch `+cpu`): small MLP PPO runs are faster there on CPU than on the laptop GPU. Pattern: a jobs file
with one command per line, `tmux new -d -s <name> "xargs -P 16 -I{} sh -c '{}' < jobs.txt"` with
`OMP_NUM_THREADS=2`, then `rsync -az --exclude checkpoints dl:~/Projects/skrl/runs/<dir>/ runs/<dir>/`.

## skills

This project uses the following Claude skills:

| Skill | Command | Purpose |
|-------|---------|---------|
| graphify | `/graphify` | Knowledge graph generation |

See `~/.claude/skills/graphify/SKILL.md` for skill documentation.

## Development Notes

- The project is installed in editable mode (`uv pip install -e .`)
- Changes to the codebase are immediately reflected without reinstallation
- CUDA-enabled PyTorch is available for GPU acceleration
- MuJoCo 3.9.0 is licensed software; ensure proper licensing for your use case

## Related Branches

- `feature/belief-pomdp-manipulation` — Current work on belief-based POMDP manipulation agent
- `develop` — Main development branch (for PRs)
- `main` — Release branch
