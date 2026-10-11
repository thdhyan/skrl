"""Plot and tabulate the transformer vs MLP sweep: return per task/model (mean ± min/max over seeds).

    .venv/bin/python examples/transformer_mujoco/plot_benchmark.py runs/transformer_mujoco docs/belief_ppo/diagrams/06_transformer_benchmark.png
"""
import glob
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TAG = "Reward / Total reward (mean)"
TASKS = ["InvertedPendulum-v5", "InvertedDoublePendulum-v5", "HalfCheetah-v5"]
MODELS = ["transformer", "mlp"]

root, out = sys.argv[1], sys.argv[2]
fig, axes = plt.subplots(1, len(TASKS), figsize=(16, 4.5), dpi=130)
rows = []
for ax, task in zip(axes, TASKS):
    for model in MODELS:
        series = []
        for run in sorted(glob.glob(os.path.join(root, f"{task}_{model}_seed*"))):
            acc = EventAccumulator(run)
            acc.Reload()
            if TAG in acc.Tags()["scalars"]:
                series.append(acc.Scalars(TAG))
        if not series:
            continue
        # seeds log at different steps (an update with no finished episode logs nothing): interpolate on a grid
        steps = np.linspace(0, min(s[-1].step for s in series), 200)
        vals = np.array([np.interp(steps, [e.step for e in s], [e.value for e in s]) for s in series])
        line = ax.plot(steps, vals.mean(0), label=f"{model} (n={len(series)})", linewidth=1.6)[0]
        ax.fill_between(steps, vals.min(0), vals.max(0), color=line.get_color(), alpha=0.15)
        # final = mean of the last 5 logged points per seed, then mean/std over seeds
        final = np.array([np.mean([e.value for e in s[-5:]]) for s in series])
        rows.append((task, model, len(series), int(steps[-1]), final.mean(), final.std()))
    ax.set_title(task)
    ax.set_xlabel("trainer steps (× num_envs = env steps)")
    ax.set_ylabel("episode return (mean)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
fig.suptitle("PPO on MuJoCo: transformer vs MLP policy/value (mean, min/max over seeds)")
fig.tight_layout()
fig.savefig(out)
print(out)
print("| Task | Model | Seeds | Last trainer step | Final return (last 5 logs, mean ± std over seeds) |")
print("|---|---|---|---|---|")
for task, model, n, step, mean, std in rows:
    print(f"| {task} | {model} | {n} | {step} | {mean:.1f} ± {std:.1f} |")
