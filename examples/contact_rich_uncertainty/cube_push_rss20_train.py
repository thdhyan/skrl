"""Train the RSS20 controller-selection policy (dueling double DQN on the compressed particle belief) for the cube push.

Each environment step executes one controller of the 16-controller bank in MuJoCo and updates the particle belief,
so the environments run in parallel processes; rollouts inside each process use ``CUBE_PUSH_THREADS`` threads.

    CUBE_PUSH_THREADS=1 OMP_NUM_THREADS=1 python examples/contact_rich_uncertainty/cube_push_rss20_train.py \
        --num-envs 48 --timesteps 1500
"""
import argparse
import math

import gymnasium

from skrl.agents.torch.belief_dqn.models import ResNetDuelingQNetwork
from skrl.agents.torch.ddqn import DDQN, DDQN_CFG
from skrl.envs.cube_push.rss20 import RSS20MacroEnv
from skrl.envs.cube_push.sim import CubePushCfg
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

parser = argparse.ArgumentParser()
parser.add_argument("--num-envs", type=int, default=48)
parser.add_argument("--timesteps", type=int, default=1500, help="vector steps (num-envs transitions each)")
parser.add_argument("--out", default="runs/cube_push_rss20")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--pusher", choices=["gantry", "arm"], default="gantry")
args = parser.parse_args()

set_seed(args.seed)


def make(i):
    return lambda: RSS20MacroEnv(seed=1000 * args.seed + i, env_cfg=CubePushCfg(pusher=args.pusher))


if __name__ == "__main__":
    env = gymnasium.vector.AsyncVectorEnv([make(i) for i in range(args.num_envs)],
                                          autoreset_mode=gymnasium.vector.AutoresetMode.SAME_STEP)
    env = wrap_env(env)
    device = env.device

    memory = RandomMemory(memory_size=100_000, num_envs=env.num_envs, device=device)
    models = {role: ResNetDuelingQNetwork(env.observation_space, env.action_space, device)
              for role in ("q_network", "target_q_network")}

    cfg = DDQN_CFG()
    cfg.discount_factor = 0.98
    cfg.batch_size = 128
    cfg.gradient_steps = 8
    cfg.learning_rate = 3e-4
    cfg.learning_starts = 50
    cfg.target_update_interval = 5
    cfg.polyak = 0.05
    cfg.exploration_scheduler = lambda t, total: max(math.exp(-5 * t / (0.6 * total)), 0.05)
    cfg.experiment.directory = args.out
    cfg.experiment.experiment_name = "ddqn"
    cfg.experiment.write_interval = 20
    cfg.experiment.checkpoint_interval = 250

    agent = DDQN(models=models, memory=memory, cfg=cfg, observation_space=env.observation_space,
                 state_space=env.state_space, action_space=env.action_space, device=device)
    SequentialTrainer(cfg={"timesteps": args.timesteps, "headless": True}, env=env, agents=agent).train()
