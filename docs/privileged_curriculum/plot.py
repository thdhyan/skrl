"""Plot the privileged-curriculum sweep: return per mode (mean ± min/max over seeds), mask probability, estimator loss.

    .venv/bin/python docs/privileged_curriculum/plot.py runs/privileged_curriculum docs/privileged_curriculum/diagrams/02_sweep.png
"""
import glob
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TAGS = [
    ("Reward / Total reward (mean)", "episode return (mean)"),
    ("Curriculum / Mask probability", "mask probability"),
    ("Loss / Estimator loss", "estimator loss ||sg(z) − ẑ||²"),
]
MODES = {"curriculum": "curriculum (visible → masked)", "masked": "masked from start", "visible": "always visible"}

root, out = sys.argv[1], sys.argv[2]
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), dpi=130)
for mode, label in MODES.items():
    runs = sorted(glob.glob(os.path.join(root, f"{mode}_seed*")))
    if not runs:
        continue
    accs = []
    for run in runs:
        acc = EventAccumulator(run)
        acc.Reload()
        accs.append(acc)
    for ax, (tag, ylabel) in zip(axes, TAGS):
        series = [acc.Scalars(tag) for acc in accs if tag in acc.Tags()["scalars"]]
        if not series:
            continue
        # seeds log at different steps (an update with no finished episode logs no return): interpolate on a grid
        steps = np.linspace(0, min(s[-1].step for s in series), 300)
        vals = np.array([np.interp(steps, [e.step for e in s], [e.value for e in s]) for s in series])
        line = ax.plot(steps, vals.mean(0), label=f"{label} (n={len(series)})", linewidth=1.6)[0]
        ax.fill_between(steps, vals.min(0), vals.max(0), color=line.get_color(), alpha=0.15)
        ax.set_ylabel(ylabel)
for ax in axes:
    ax.set_xlabel("trainer steps (× num_envs = env steps)")
    ax.grid(alpha=0.3)
axes[0].legend(fontsize=8, loc="upper left")
axes[2].set_yscale("log")
fig.suptitle("PrivilegedPPO, CartPole-v1 POMDP + random pole mass/length (training curves use each episode's own mask)")
fig.tight_layout()
fig.savefig(out)
print(out)
