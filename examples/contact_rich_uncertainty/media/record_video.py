import argparse
import sys
import os

sys.path.insert(0, "/home/thakk100/Projects/Humanoids26/skrl/examples/contact_rich_uncertainty")
import numpy as np
import torch
import gymnasium
import imageio

import cartpole_pomdp_belief_ppo as M
from skrl.agents.torch.ppo import PPO, PPO_CFG
from skrl.envs.wrappers.torch import wrap_env
from skrl.envs.wrappers.torch.pomdp_wrapper import PomdpWrapper
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.utils import set_seed

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--baseline", action="store_true")
parser.add_argument("--out", required=True)
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--max-steps", type=int, default=500)
parser.add_argument(
    "--env-max-episode-steps",
    type=int,
    default=500,
    help="Override CartPole-v1's built-in 500-step TimeLimit, e.g. 2000, to show sustained "
    "balance beyond the standard cutoff (proves stability, not just hitting the default cap).",
)
parser.add_argument("--masscart", type=float, default=M.MASSCART)
parser.add_argument("--masspole", type=float, default=M.MASSPOLE)
args = parser.parse_args()

# Seed torch/numpy/random globally -- without this, the particle filter's process
# noise and resampling (torch.randn draws) are NOT reproducible across runs even
# with the same --seed, since --seed only seeds the env's own RNG. This was the
# cause of the light-pole rollout reward=10 anomaly: an unseeded torch RNG made
# every invocation a different noise realization.
set_seed(args.seed)

device = "cpu"
base_env = gymnasium.make("CartPole-v1", render_mode="rgb_array", max_episode_steps=args.env_max_episode_steps)
M.apply_cartpole_physics(base_env, args.masscart, args.masspole)
env = PomdpWrapper(base_env, hidden_obs_indices=M.HIDDEN, obs_noise_std=0.02, seed=args.seed)
if not args.baseline:
    env = M.BeliefObservationWrapper(env, device=device)
wenv = wrap_env(env)

memory = RandomMemory(memory_size=256, num_envs=wenv.num_envs, device=wenv.device)
models = {
    "policy": M.Policy(wenv.observation_space, wenv.state_space, wenv.action_space, wenv.device),
    "value": M.Value(wenv.observation_space, wenv.state_space, wenv.action_space, wenv.device),
}
cfg = PPO_CFG()
cfg.observation_preprocessor = RunningStandardScaler
cfg.observation_preprocessor_kwargs = {"size": wenv.observation_space, "device": wenv.device}
cfg.value_preprocessor = RunningStandardScaler
cfg.value_preprocessor_kwargs = {"size": 1, "device": wenv.device}
cfg.experiment.write_interval = 0
cfg.experiment.checkpoint_interval = 0

agent = PPO(
    models=models, memory=memory, cfg=cfg,
    observation_space=wenv.observation_space, state_space=wenv.state_space,
    action_space=wenv.action_space, device=wenv.device,
)
agent.init()
agent.load(args.checkpoint)

def greedy_action(obs):
    inputs = {"observations": agent._observation_preprocessor(obs), "states": agent._state_preprocessor(None)}
    logits = agent.policy.compute(inputs, role="policy")[0]
    return logits.argmax(dim=-1, keepdim=True)

frames = []
obs, info = wenv.reset()
ep_reward = 0.0
for step in range(args.max_steps):
    frame = base_env.render()
    frames.append(frame)
    action = greedy_action(obs)
    obs, reward, terminated, truncated, info = wenv.step(action)
    ep_reward += float(reward)
    if bool(terminated) or bool(truncated):
        frames.append(base_env.render())
        break

print(f"episode reward = {ep_reward:.1f}, steps = {len(frames)}")
os.makedirs(os.path.dirname(args.out), exist_ok=True)
imageio.mimsave(args.out, frames, fps=50)
print("saved", args.out)
