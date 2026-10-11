"""Figures and a results table for the built-in-task sweep of ``examples/privileged_curriculum/gym_privileged_ppo.py``.

    .venv/bin/python docs/privileged_curriculum/plot_gym.py runs/privileged_gym docs/privileged_curriculum/diagrams

Writes ``03_gym_returns.png`` (masked / visible eval return per task and mode), ``04_probe_r2.png`` (held-out R² of
each privileged group from ẑ and z) and ``05_gate_traces.png`` (mask probability and training return over training,
time-based vs reward-gated), and prints a markdown table.
"""
import collections
import glob
import json
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODES = ["curriculum", "gated", "masked", "visible"]
COLORS = {"curriculum": "#1f77b4", "gated": "#2ca02c", "masked": "#ff7f0e", "visible": "#7f7f7f"}

root, out = sys.argv[1], sys.argv[2]
runs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(root, "*.json")))]
tasks = sorted({r["task"] for r in runs})
by = collections.defaultdict(list)
for r in runs:
    by[r["task"], r["mode"]].append(r)

# returns
fig, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 4.2), dpi=130, squeeze=False)
for ax, task in zip(axes[0], tasks):
    for j, key in enumerate(["return_masked", "return_visible"]):
        for i, mode in enumerate(MODES):
            vals = [r[key] for r in by[task, mode]]
            if not vals:
                continue
            x = i + (j - 0.5) * 0.38
            ax.bar(x, np.mean(vals), 0.36, color=COLORS[mode], alpha=1.0 if j == 0 else 0.4,
                   label=("masked eval" if j == 0 else "visible eval") if i == 0 else None)
            ax.scatter([x] * len(vals), vals, s=10, c="black", zorder=3)
    ax.set_xticks(range(len(MODES)), MODES)
    ax.set_title(task)
    ax.grid(axis="y", alpha=0.3)
axes[0][0].set_ylabel("deterministic eval return (dots: seeds)")
axes[0][0].legend(fontsize=8)
fig.suptitle("PrivilegedPPO on built-in tasks: solid = privileged masked (deployable), light = visible")
fig.tight_layout()
fig.savefig(os.path.join(out, "03_gym_returns.png"))

# probe R²
fig, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 4.2), dpi=130, squeeze=False)
for ax, task in zip(axes[0], tasks):
    keys = sorted({k for r in runs if r["task"] == task for k in r["probe_r2"]})
    groups = list(dict.fromkeys(k.split(" R2 ")[1] for k in keys))
    rows = []
    for latent in ("ẑ", "z"):
        for mode in MODES:
            vals = [[r["probe_r2"].get(f"{latent} R2 {g}", np.nan) for g in groups] for r in by[task, mode]]
            if vals:
                rows.append((f"{latent} ({mode})", np.nanmean(vals, axis=0)))
    data = np.array([v for _, v in rows])
    ax.imshow(np.clip(data, 0, 1), cmap="viridis", vmin=0, vmax=1, aspect="auto")
    for (i, j), v in np.ndenumerate(data):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="white" if v < 0.6 else "black", fontsize=8)
    ax.set_xticks(range(len(groups)), groups)
    ax.set_yticks(range(len(rows)), [n for n, _ in rows], fontsize=8)
    ax.set_title(task)
fig.suptitle("Held-out R² of a probe predicting each privileged group (ẑ: history estimator, z: encoder)")
fig.tight_layout()
fig.savefig(os.path.join(out, "04_probe_r2.png"))

# mask traces: curriculum vs gated, first seed of each task
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

fig, axes = plt.subplots(2, len(tasks), figsize=(5 * len(tasks), 6), dpi=130, squeeze=False)
for col, task in enumerate(tasks):
    for mode in ("curriculum", "gated"):
        dirs = sorted(glob.glob(os.path.join(root, f"{task}_{mode}_seed*")))
        dirs = [d for d in dirs if os.path.isdir(d)]
        for k, d in enumerate(dirs):
            acc = EventAccumulator(d)
            acc.Reload()
            tags = acc.Tags()["scalars"]
            for row, tag in enumerate(["Curriculum / Mask probability", "Reward / Total reward (mean)"]):
                if tag in tags:
                    s = acc.Scalars(tag)
                    axes[row][col].plot([e.step for e in s], [e.value for e in s], color=COLORS[mode], alpha=0.8,
                                        linewidth=1.2, label=mode if k == 0 else None)
    axes[0][col].set_title(task)
    axes[0][col].set_ylabel("mask probability")
    axes[1][col].set_ylabel("training return (mean)")
    axes[1][col].set_xlabel("trainer steps")
    for row in range(2):
        axes[row][col].grid(alpha=0.3)
axes[0][0].legend(fontsize=8)
fig.suptitle("Time-based softmax schedule vs reward-gated schedule (mask step only while the return holds)")
fig.tight_layout()
fig.savefig(os.path.join(out, "05_gate_traces.png"))

# table
print("| Task | Mode | n | Eval masked | Eval visible | ẑ R² per group | Final gate p |")
print("|---|---|---|---|---|---|---|")
for task in tasks:
    for mode in MODES:
        rs = by[task, mode]
        if not rs:
            continue
        keys = [k for k in rs[0]["probe_r2"] if k.startswith("ẑ")]
        r2 = ", ".join(f"{k.split(' R2 ')[1]} {np.mean([r['probe_r2'][k] for r in rs]):.2f}" for k in keys)
        gate = f"{np.mean([r['final_mask_probability'] for r in rs]):.2f}" if mode == "gated" else ""
        masked = [r["return_masked"] for r in rs]
        visible = [r["return_visible"] for r in rs]
        print(f"| {task} | {mode} | {len(rs)} | {np.mean(masked):.1f} ± {np.std(masked):.1f} | "
              f"{np.mean(visible):.1f} ± {np.std(visible):.1f} | {r2} | {gate} |")

# history-estimator ablations: ẑ R² per group vs history length H / observation-action history (+a).
# <root>/ablation: curriculum mode, <root>/ablation_visible: visible mode (the H = 8 baseline comes from the main sweep)
variants = [(8, False), (8, True), (32, False), (32, True)]
names = [f"H={h}" + (" +a" if a else "") for h, a in variants]
for subdir, mode, png in (("ablation", "curriculum", "06_history_ablation.png"),
                          ("ablation_visible", "visible", "07_history_ablation_visible.png")):
    ablation = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(root, subdir, "*.json")))]
    if not ablation:
        continue
    ablation += [dict(r, history=r.get("history", 8), history_actions=False) for r in runs if r["mode"] == mode]
    ab_tasks = sorted({r["task"] for r in ablation if r["history_actions"] or r["history"] != 8})
    fig, axes = plt.subplots(1, len(ab_tasks), figsize=(5 * len(ab_tasks), 4.2), dpi=130, squeeze=False)
    print(f"\n| Task ({mode}) | History | n | Eval masked | ẑ R² per group | z R² per group |")
    print("|---|---|---|---|---|---|")
    for ax, task in zip(axes[0], ab_tasks):
        rs_task = [r for r in ablation if r["task"] == task]
        groups = [k.split(" R2 ")[1] for k in rs_task[0]["probe_r2"] if k.startswith("ẑ")]
        for v, ((h, a), name) in enumerate(zip(variants, names)):
            rs = [r for r in rs_task if r["history"] == h and r["history_actions"] == a]
            if not rs:
                continue
            for g, group in enumerate(groups):
                vals = [r["probe_r2"][f"ẑ R2 {group}"] for r in rs]
                x = g + (v - 1.5) * 0.18
                ax.bar(x, np.mean(vals), 0.17, color=plt.cm.viridis(v / 3), label=name if g == 0 else None)
                ax.scatter([x] * len(vals), vals, s=8, c="black", zorder=3)
            r2 = lambda latent: ", ".join(  # noqa: E731
                f"{g} {np.mean([r['probe_r2'][f'{latent} R2 {g}'] for r in rs]):.2f}" for g in groups)
            masked = [r["return_masked"] for r in rs]
            print(f"| {task} | {name} | {len(rs)} | {np.mean(masked):.1f} ± {np.std(masked):.1f} | {r2('ẑ')} | {r2('z')} |")
        ax.set_xticks(range(len(groups)), groups)
        ax.set_ylim(-0.1, 1.0)
        ax.set_title(task)
        ax.grid(axis="y", alpha=0.3)
    axes[0][0].set_ylabel("held-out R² from ẑ")
    axes[0][0].legend(fontsize=8)
    fig.suptitle(f"History estimator as system ID ({mode} mode): history length H and observation-action history (+a)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, png))

# encoder regularizer off (λ = 0): does z keep the dynamics, and does ẑ follow?
lam0 = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(root, "ablation_lam0", "*.json")))]
if lam0:
    print("\n| Task | Mode | λ | n | Eval masked | Eval visible | ẑ R² per group | z R² per group |")
    print("|---|---|---|---|---|---|---|---|")
    for task in sorted({r["task"] for r in lam0}):
        for mode in ("curriculum", "visible"):
            for lam, source in ((0.1, runs), (0.0, lam0)):
                rs = [r for r in source if r["task"] == task and r["mode"] == mode]
                if not rs:
                    continue
                groups = [k.split(" R2 ")[1] for k in rs[0]["probe_r2"] if k.startswith("ẑ")]
                r2 = lambda latent: ", ".join(  # noqa: E731
                    f"{g} {np.mean([r['probe_r2'][f'{latent} R2 {g}'] for r in rs]):.2f}" for g in groups)
                m, v = [r["return_masked"] for r in rs], [r["return_visible"] for r in rs]
                print(f"| {task} | {mode} | {lam} | {len(rs)} | {np.mean(m):.1f} ± {np.std(m):.1f} | "
                      f"{np.mean(v):.1f} ± {np.std(v):.1f} | {r2('ẑ')} | {r2('z')} |")
