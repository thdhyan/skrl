"""Self-play on the 2D grid pursuit-evasion environment (n pursuers vs. 1 evader) with MAPPO/IPPO.

Examples::

    # default: 100x100 arena, randomized spawns, a different opponent sampler every generation, debug videos
    python examples/pursuit_evasion/torch_pursuit_evasion_selfplay.py --sampler-schedule last uniform hedge \
        --checkpoint-dir runs/pursuit_evasion/demo

    # 3 pursuers on a 10x10 grid, local potential-field observations, hedge opponent sampling
    python examples/pursuit_evasion/torch_pursuit_evasion_selfplay.py --n-pursuers 3 --grid 10 10 \\
        --obs-mode local_potential --sampler hedge --generations 20 --timesteps-per-generation 3000

    # adaptive population-based self-play: 4 concurrent learners per team with population-based training
    python examples/pursuit_evasion/torch_pursuit_evasion_selfplay.py --sampler adaptive_pbt --population-size 4

    # quick CPU check (a few seconds)
    python examples/pursuit_evasion/torch_pursuit_evasion_selfplay.py --generations 2 \\
        --timesteps-per-generation 64 --num-envs 8 --eval-episodes 8 --device cpu
"""

import argparse
import json

from skrl.envs.pursuit_evasion import OBSERVATION_BUILDERS, PursuitEvasionEnv
from skrl.envs.wrappers.torch import wrap_env
from skrl.multi_agents.torch.selfplay import SAMPLERS, SelfPlayCfg, SelfPlayRunner
from skrl.utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # environment
    parser.add_argument("--n-pursuers", type=int, default=3, help="number of pursuers")
    parser.add_argument("--grid", type=int, nargs=2, default=(100, 100), metavar=("H", "W"), help="grid size")
    parser.add_argument("--num-envs", type=int, default=64, help="number of parallel sub-environments")
    parser.add_argument("--max-steps", type=int, default=None, help="maximum episode length (default: 2 * (H + W))")
    parser.add_argument("--capture-radius", type=int, default=1, help="0: same cell, 1: adjacent, ...")
    parser.add_argument("--obstacle-density", type=float, default=0.0, help="fraction of random static obstacles")
    parser.add_argument("--spawn-distance", type=int, nargs=2, default=(5, 60), metavar=("MIN", "MAX"),
                        help="per-episode minimum evader-pursuer spawn distance is drawn from this range")
    parser.add_argument("--spawn-cluster-prob", type=float, default=0.5, help="probability of clustered pursuers")
    parser.add_argument("--spawn-cluster-radius", type=int, default=5, help="half-size of the pursuer cluster")
    parser.add_argument("--obs-mode", default="full_state", choices=sorted(OBSERVATION_BUILDERS))
    parser.add_argument("--local-radius", type=int, default=2, help="radius of the local potential window")
    # self-play
    parser.add_argument("--generations", type=int, default=10, help="number of generations (G)")
    parser.add_argument("--timesteps-per-generation", type=int, default=2000, help="env steps per generation")
    parser.add_argument("--sampler", default="uniform", choices=sorted(SAMPLERS), help="opponent sampling strategy")
    parser.add_argument("--sampler-kwargs", type=json.loads, default={}, help='e.g. \'{"eta": 0.2, "top_k": 3}\'')
    parser.add_argument("--sampler-schedule", nargs="+", default=None, choices=sorted(SAMPLERS),
                        help="sampler per generation, cycled (kwargs: --sampler-kwargs '{\"hedge\": {...}}')")
    parser.add_argument("--scheme", default="alternating", choices=["alternating", "simultaneous"])
    parser.add_argument("--first-learner", default="pursuers", choices=["pursuers", "evader"])
    parser.add_argument(
        "--learning-teams", nargs="+", default=None, choices=["pursuers", "evader"], help="default: both teams"
    )
    parser.add_argument("--algorithm", default="mappo", choices=["mappo", "ippo"])
    parser.add_argument("--eval-episodes", type=int, default=64, help="evaluation episodes per opponent")
    parser.add_argument("--eval-max-opponents", type=int, default=4)
    parser.add_argument("--pbt", default=None, choices=["on", "off"], help="default: on for adaptive_pbt only")
    parser.add_argument(
        "--population-size", type=int, default=None, help="learners per team (default: 4 for adaptive_pbt, else 1)"
    )
    parser.add_argument("--population-snapshots", default="all", choices=["all", "best"])
    parser.add_argument("--pbt-truncation", type=float, default=0.25, help="fraction replaced by truncation selection")
    parser.add_argument("--rollouts", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--entropy", type=float, default=0.01, help="entropy loss scale")
    parser.add_argument("--hidden", type=int, nargs="+", default=(128, 128), help="MLP hidden sizes")
    # misc
    parser.add_argument("--checkpoint-dir", default=None, help="write snapshots and history.json here")
    parser.add_argument("--video-dir", default=None, help="debug videos (default: <checkpoint-dir>/videos)")
    parser.add_argument("--video-interval", type=int, default=1, help="generations between debug videos")
    parser.add_argument("--wandb", action="store_true", help="log every generation to Weights & Biases")
    parser.add_argument("--wandb-project", default="skrl-pursuit-evasion")
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    env = PursuitEvasionEnv(
        num_envs=args.num_envs,
        height=args.grid[0],
        width=args.grid[1],
        num_pursuers=args.n_pursuers,
        max_steps=args.max_steps or 2 * (args.grid[0] + args.grid[1]),
        spawn_min_distance_range=tuple(args.spawn_distance),
        spawn_cluster_prob=args.spawn_cluster_prob,
        spawn_cluster_radius=args.spawn_cluster_radius,
        # repulsion radii in cells: scale with the arena so the fields stay informative on large grids
        d0_pp=max(2.5, max(args.grid) / 20),
        d0_ep=max(4.0, max(args.grid) / 10),
        capture_radius=args.capture_radius,
        obstacle_density=args.obstacle_density,
        obs_mode=args.obs_mode,
        local_radius=args.local_radius,
        device=args.device,
        seed=args.seed,
    )
    env = wrap_env(env, wrapper="pursuit-evasion")
    print(f"observation size: {env.observation_space('evader').shape}, state size: {env.state_space('evader').shape}")
    if args.grid[0] * args.grid[1] <= 400:
        print(env.render())

    cfg = SelfPlayCfg(
        generations=args.generations,
        timesteps_per_generation=args.timesteps_per_generation,
        algorithm=args.algorithm,
        sampler=args.sampler,
        sampler_kwargs=args.sampler_kwargs,
        sampler_schedule=args.sampler_schedule,
        training_scheme=args.scheme,
        first_learner=args.first_learner,
        learning_teams=args.learning_teams,
        eval_episodes=args.eval_episodes,
        eval_max_opponents=args.eval_max_opponents,
        pbt=None if args.pbt is None else args.pbt == "on",
        population_size=args.population_size,
        population_snapshots=args.population_snapshots,
        pbt_truncation=args.pbt_truncation,
        hidden_sizes=tuple(args.hidden),
        agent_cfg={
            "rollouts": args.rollouts,
            "learning_rate": args.learning_rate,
            "entropy_loss_scale": args.entropy,
        },
        checkpoint_dir=args.checkpoint_dir,
        video_dir=args.video_dir or (args.checkpoint_dir and f"{args.checkpoint_dir}/videos"),
        video_interval=args.video_interval,
        seed=args.seed,
        wandb=args.wandb,
        wandb_kwargs={"project": args.wandb_project, "group": args.wandb_group, "name": args.wandb_name},
    )
    runner = SelfPlayRunner(env, cfg)
    history = runner.run()
    runner.close()

    print("\nfinal Elo ratings")
    for team, pool in runner.pools.items():
        print(f"  {team}: " + ", ".join(f"g{runner._entry_label(entry)}={entry.rating:.0f}" for entry in pool))
    last = history[-1]["teams"]
    print("last generation:", {team: info["eval_win_rate"] for team, info in last.items()})


if __name__ == "__main__":
    main()
