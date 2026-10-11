import glob, json, os, sys
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

EXP = os.path.dirname(os.path.abspath(__file__))
out = {"cartpole": {}, "pursuit": {}}

# ---- CartPole POMDP
for cond in ["full_obs", "masked", "masked_belief"]:
    seeds = []
    for d in sorted(glob.glob(f"{EXP}/cartpole/runs/{cond}_s*")):
        files = glob.glob(f"{d}/**/events.out.tfevents*", recursive=True)
        if not files:
            continue
        acc = EventAccumulator(os.path.dirname(files[0]), size_guidance={"scalars": 0}); acc.Reload()
        tag = "Reward / Total reward (mean)"
        if tag not in acc.Tags()["scalars"]:
            continue
        ev = acc.Scalars(tag)
        steps = np.array([e.step for e in ev]); vals = np.array([e.value for e in ev])
        seeds.append({"seed": d.rsplit("_s", 1)[1], "steps": steps.tolist(), "values": vals.tolist(),
                      "final": float(vals[-max(1, len(vals) // 10):].mean()), "best": float(vals.max()),
                      "auc": float(vals.mean()),
                      "first_ge_195": int(steps[np.argmax(vals >= 195)]) if (vals >= 195).any() else None})
    out["cartpole"][cond] = seeds

# ---- pursuit obs ablation
for mode in ["full_state", "global_potential", "local_potential"]:
    seeds = []
    for d in sorted(glob.glob(f"{EXP}/pursuit/runs/{mode}_s*")):
        path = f"{d}/history.json"
        if not os.path.exists(path):
            continue
        hist = json.load(open(path))
        tr = [r["teams"]["pursuers"]["train_win_rate"] for r in hist]
        ev = [np.mean(list(r["teams"]["pursuers"]["eval_win_rate"].values())) for r in hist]
        el = [r["elapsed_s"] for r in hist]
        seeds.append({"seed": d.rsplit("_s", 1)[1], "train_capture": tr, "eval_capture": [float(x) for x in ev], "elapsed": el})
    out["pursuit"][mode] = seeds

json.dump(out, open(f"{EXP}/results.json", "w"), indent=1)
print("== CartPole (episode return, 30k steps; mean ± std over seeds)")
for cond, seeds in out["cartpole"].items():
    if seeds:
        f = [s["final"] for s in seeds]; a = [s["auc"] for s in seeds]; b = [s["best"] for s in seeds]
        hit = [s["first_ge_195"] for s in seeds]
        print(f"{cond:14s} n={len(seeds)} final {np.mean(f):6.1f}±{np.std(f):5.1f} | mean-over-training {np.mean(a):6.1f}±{np.std(a):5.1f} | best {np.mean(b):6.1f} | first step >=195: {hit}")
print("== Pursuit (pursuers' capture rate vs random evader; mean over seeds per generation)")
for mode, seeds in out["pursuit"].items():
    if seeds:
        tr = np.array([s["train_capture"] for s in seeds]); ev = np.array([s["eval_capture"] for s in seeds])
        print(f"{mode:17s} n={len(seeds)} train {np.round(tr.mean(0), 3).tolist()} | eval {np.round(ev.mean(0), 3).tolist()} | s/gen {np.mean([np.mean(s['elapsed']) for s in seeds]):.0f}")
