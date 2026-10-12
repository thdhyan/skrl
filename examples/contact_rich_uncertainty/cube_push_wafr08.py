"""WAFR08 belief-based execution on the partially observable MuJoCo cube push, with a debug video of the belief.

Hidden: the cube's initial pose (x, y, yaw) and its friction μ, mass m and damping k_d. The planner is given only
the wanted final cube pose. Each iteration it either stops (goal reached in belief), executes a goal push whose belief
pre-image holds the belief, or executes the most informative touch/probe (IG - c FP), then updates the grid belief
with the observed contact, travel and forces.

    MUJOCO_GL=egl python examples/contact_rich_uncertainty/cube_push_wafr08.py --seed 1
"""
import argparse
import json
import math
import os
import time

import numpy as np

from skrl.envs.cube_push import CubePushEnv, pose_error
from skrl.envs.cube_push.render import BeliefVideo
from skrl.envs.cube_push.wafr08 import WAFR08CubePlanner

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--out", default="runs/cube_push_wafr08")
parser.add_argument("--no-video", action="store_true")
args = parser.parse_args()

env = CubePushEnv()
_, info = env.reset(seed=args.seed)
planner = WAFR08CubePlanner(env.cfg)
planner.reset(info["goal"])
state = {"i": 0, "wrt": "-", "reason": "prior", "p_pre": 0.0}


def fmt(pose):
    return f"({100 * pose[0]:5.1f}, {100 * pose[1]:5.1f} cm, {math.degrees(pose[2]):5.1f}°)"


def status(phase: str) -> str:
    b, w = planner.belief, planner.map_state()
    return (
        f"WAFR08 cube push  seed {args.seed}  t {env.steps * env.cfg.control_dt:5.1f} s  WRT #{state['i']}: "
        f"{state['wrt']}  [{phase}]\n"
        f"why: {state['reason']}   H(b) {b.entropy():.2f} nats   P(goal) {planner.goal_probability():.2f}   "
        f"P(BPre) {state['p_pre']:.2f}\n"
        f"true {fmt(env.cube_pose())} mu {env.physics[0]:.2f} m {env.physics[1]:.2f} k_d {env.physics[2]:.1f} | "
        f"MAP {fmt(w)} mu {w[3]:.2f} m {w[4]:.2f} k_d {w[5]:.1f} | goal {fmt(planner.goal)}"
    )


os.makedirs(os.path.join(args.out, "videos"), exist_ok=True)
video_path = os.path.join(args.out, "videos", f"wafr08_seed{args.seed}.mp4")
video = None if args.no_video else BeliefVideo(env, planner, status, video_path)
log = []
t0 = time.time()
if video:
    video.hold(1.5)
for i in range(planner.cfg.max_wrts):
    wrt, report = planner.decide()
    state.update(i=i, reason=report["reason"], p_pre=report.get("p_preimage", 0.0))
    if wrt is None:
        break
    state["wrt"] = wrt.name
    if video:
        video.wrt = wrt
        video.hold(0.6)
    observed = planner.execute(env, wrt, video.on_step if video else None)
    planner.update(wrt, observed)
    dist, dyaw = pose_error(env.cube_pose(), planner.goal)
    log.append({"wrt": wrt.name, "reason": report["reason"], "observed": observed.tolist(),
                "entropy": planner.belief.entropy(), "p_goal": planner.goal_probability(),
                "map": planner.map_state().tolist(), "true_pose": env.cube_pose().tolist(),
                "goal_dist": float(dist), "goal_dyaw_deg": math.degrees(float(dyaw))})
    print(f"{i:2d} {wrt.name:40s} {report['reason'][:55]:55s} H {log[-1]['entropy']:.2f} "
          f"P(goal) {log[-1]['p_goal']:.2f} err {100 * dist:.1f} cm {math.degrees(dyaw):.1f}°", flush=True)
    if video:
        video.wrt = None
        video.history.append((i + 1, log[-1]["entropy"], log[-1]["p_goal"]))
        video.refresh()
        video.hold(0.8)

dist, dyaw = pose_error(env.cube_pose(), planner.goal)
tol_p, tol_y = env.cfg.goal_tolerance
result = {
    "seed": args.seed, "wrts": len(log), "sim_time_s": env.steps * env.cfg.control_dt, "wall_s": time.time() - t0,
    "success": bool(dist < tol_p and dyaw < tol_y), "goal_dist_cm": 100 * float(dist),
    "goal_dyaw_deg": math.degrees(float(dyaw)), "stopped": state["reason"], "physics": env.physics.tolist(),
    "goal": planner.goal.tolist(), "final_pose": env.cube_pose().tolist(), "log": log,
}
print({k: v for k, v in result.items() if k != "log"})
with open(os.path.join(args.out, f"seed{args.seed}.json"), "w") as f:
    json.dump(result, f, indent=1)
if video:
    state.update(wrt="-", reason=f"stop: {state['reason']}; success {result['success']}")
    last = video.hold(3.0)
    video.close()
    import imageio.v2 as imageio

    imageio.imwrite(video_path.replace(".mp4", "_final.png"), last)
    print("video", video_path, video.n_frames, "frames")
