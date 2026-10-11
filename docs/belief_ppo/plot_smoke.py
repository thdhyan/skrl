"""Plot CartPole-POMDP smoke runs (TensorBoard events) → learning-curve PNG + markdown table.

    .venv/bin/python docs/belief_ppo/plot_smoke.py runs/belief_ppo_smoke docs/belief_ppo/diagrams/04_smoke_curves.png
"""
import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TAG = "Reward / Total reward (mean)"
LABELS = {
    "C_BeliefPPO": "C: BeliefPPO (agent-owned filter, ResNet)",
    "A_WrapperPPO": "A: belief wrapper + stock PPO (MLP)",
    "Base_RawObsPPO": "baseline: stock PPO on raw masked obs",
}

root, out = sys.argv[1], sys.argv[2]
fig, ax = plt.subplots(figsize=(9, 4.8), dpi=150)
rows = []
for name, label in LABELS.items():
    events = glob.glob(os.path.join(root, name, "**", "events.out.tfevents.*"), recursive=True)
    if not events:
        continue
    acc = EventAccumulator(os.path.dirname(sorted(events)[-1]))
    acc.Reload()
    if TAG not in acc.Tags()["scalars"]:
        continue
    sc = acc.Scalars(TAG)
    steps, vals = [s.step for s in sc], [s.value for s in sc]
    ax.plot(steps, vals, label=label, linewidth=1.8)
    last = vals[-10:]
    rows.append((label, steps[-1], max(vals), sum(last) / len(last)))

ax.set_xlabel("env steps")
ax.set_ylabel("episode return (mean)")
ax.set_title("CartPole-v1 POMDP (cart vel + pole ang. vel hidden), 20k steps, seed 42")
ax.grid(alpha=0.3)
ax.legend(loc="upper left", fontsize=9)
fig.tight_layout()
fig.savefig(out)

print("| run | last step | max return | mean of last 10 logs |")
print("|---|---|---|---|")
for r in rows:
    print(f"| {r[0]} | {r[1]} | {r[2]:.1f} | {r[3]:.1f} |")
