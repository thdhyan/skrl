"""Belief-space methods on the partially observable MuJoCo cube push, with a debug video of the belief.

Hidden: the cube's initial pose (x, y, yaw) and its friction μ, mass m and damping k_d. The planner is given only
the wanted final cube pose. Each iteration it picks a controller (touch, force probe or goal push) from its belief,
executes it and updates the belief with the observed contact, travel and forces.

* ``wafr08``: stop when the goal holds in belief, goal push whose belief pre-image holds the belief, else IG - c FP.
* ``audit``: fixed probe schedule until the retained set is tight, scenario-minimax commit, audited with the truth.
* ``genaif``: learned generative model (runs/cube_push_genaif/model.pt, trained if missing), expected free energy.
* ``quasistatic``: localize by touches, then a CMA-ES optimized open-loop push sequence (variance-reducing pushes).
* ``rss20``: particle belief, GMM compression, DQN over a 16-controller bank (``--policy``, from
  ``cube_push_rss20_train.py``); stops on the task-completion signal.

    MUJOCO_GL=egl python examples/contact_rich_uncertainty/cube_push_belief.py --method wafr08 --seed 1
"""
import argparse
import json
import math
import os
import time

import numpy as np

from skrl.envs.cube_push import CubePushEnv, pose_error
from skrl.envs.cube_push.render import BeliefVideo
from skrl.envs.cube_push.sim import CubePushCfg
from skrl.envs.cube_push.wafr08 import WAFR08CubePlanner

METHODS = {"wafr08": "WAFR08", "audit": "Audit", "genaif": "GenAIF", "quasistatic": "Quasi-static", "rss20": "RSS20"}
parser = argparse.ArgumentParser()
parser.add_argument("--method", choices=list(METHODS), default="wafr08")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--pusher", choices=["gantry", "arm", "polar", "artic", "franka", "ur5e"], default="gantry",
                    help="low level: gantry slides, SCARA arm, cylindrical arm, articulated arm, Franka, or UR5e under IK "
                    "(same EE interface)")
parser.add_argument("--out", default=None, help="default runs/cube_push[_arm]_<method>")
parser.add_argument("--policy", default="runs/cube_push_rss20/ddqn/checkpoints/best_agent.pt", help="rss20 only")
parser.add_argument("--no-video", action="store_true")
args = parser.parse_args()
args.out = args.out or (f"runs/cube_push_{args.method}" if args.pusher == "gantry"
                        else f"runs/cube_push_{args.pusher}_{args.method}")

env = CubePushEnv(CubePushCfg(pusher=args.pusher))
_, info = env.reset(seed=args.seed)
if args.method == "audit":
    from skrl.envs.cube_push.audit import AuditCubePlanner

    planner = AuditCubePlanner(env.cfg)
elif args.method == "genaif":
    from skrl.envs.cube_push.genaif import GenAIFCubePlanner

    planner = GenAIFCubePlanner(env.cfg)
    if not os.path.exists(planner.cfg.model_path):
        planner.train_model()
elif args.method == "quasistatic":
    from skrl.envs.cube_push.quasistatic import QuasiStaticCubePlanner

    planner = QuasiStaticCubePlanner(env.cfg, seed=args.seed)
elif args.method == "rss20":
    import gymnasium
    import torch

    from skrl.agents.torch.belief_dqn.models import ResNetDuelingQNetwork
    from skrl.envs.cube_push.rss20 import OBS_DIM, RSS20CubePlanner

    policy = ResNetDuelingQNetwork(gymnasium.spaces.Box(-np.inf, np.inf, (OBS_DIM,)),
                                   gymnasium.spaces.Discrete(RSS20CubePlanner.num_controllers), "cpu")
    policy.load_state_dict(torch.load(args.policy, map_location="cpu")["q_network"])
    planner = RSS20CubePlanner(env.cfg, seed=args.seed, policy=policy.eval())
else:
    planner = WAFR08CubePlanner(env.cfg)
planner.reset(info["goal"])
max_steps = planner.cfg.max_controllers if args.method == "rss20" else planner.cfg.max_wrts
state = {"i": 0, "wrt": "-", "reason": "prior", "p_pre": 0.0}


def fmt(pose):
    return f"({100 * pose[0]:5.1f}, {100 * pose[1]:5.1f} cm, {math.degrees(pose[2]):5.1f}°)"


def status(phase: str) -> str:
    b, w = planner.belief, planner.map_state()
    return (
        f"{METHODS[args.method]} cube push  seed {args.seed}  t {env.steps * env.cfg.control_dt:5.1f} s  WRT #{state['i']}: "
        f"{state['wrt']}  [{phase}]\n"
        f"why: {state['reason']}   H(b) {b.entropy():.2f} nats   P(goal) {planner.goal_probability():.2f}   "
        f"P(BPre) {state['p_pre']:.2f}\n"
        f"true {fmt(env.cube_pose())} mu {env.physics[0]:.2f} m {env.physics[1]:.2f} k_d {env.physics[2]:.1f} | "
        f"MAP {fmt(w)} mu {w[3]:.2f} m {w[4]:.2f} k_d {w[5]:.1f} | goal {fmt(planner.goal)}"
    )


os.makedirs(os.path.join(args.out, "videos"), exist_ok=True)
video_path = os.path.join(args.out, "videos", f"{args.method}_seed{args.seed}.mp4")
video = None if args.no_video else BeliefVideo(env, planner, status, video_path)
log = []
t0 = time.time()
if video:
    video.hold(1.5)
tol_p, tol_y = env.cfg.goal_tolerance
for i in range(max_steps):
    if args.method == "audit":
        planner.true_state = np.concatenate([env.cube_pose(), env.physics])  # for the audit only
    wrt, report = planner.decide()
    state.update(i=i, reason=report["reason"], p_pre=report.get("p_preimage", 0.0))
    if wrt is None:
        break
    state["wrt"] = wrt.name
    if video:
        video.wrt = wrt
        video.hold(0.6)
    observed = planner.execute(env, wrt, video.on_step if video else None)
    if args.method == "audit" and wrt.kind == "goal":
        planner.audit_outcome(env.cube_pose())
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
    if args.method == "rss20" and dist < tol_p and dyaw < tol_y:
        state["reason"] = "task-completion signal"
        break

dist, dyaw = pose_error(env.cube_pose(), planner.goal)
result = {
    "method": args.method, "pusher": args.pusher, "seed": args.seed, "wrts": len(log), "sim_time_s": env.steps * env.cfg.control_dt, "wall_s": time.time() - t0,
    "success": bool(dist < tol_p and dyaw < tol_y), "goal_dist_cm": 100 * float(dist),
    "goal_dyaw_deg": math.degrees(float(dyaw)), "stopped": state["reason"], "physics": env.physics.tolist(),
    "goal": planner.goal.tolist(), "final_pose": env.cube_pose().tolist(), "log": log,
}
if args.method == "audit":
    result["audit"] = planner.audit_summary()
    result["audits"] = planner.audits
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
