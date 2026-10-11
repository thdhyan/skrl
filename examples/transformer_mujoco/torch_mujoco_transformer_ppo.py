"""PPO on MuJoCo tasks with Transformer (or MLP baseline) policy/value models.

Usage (from the repository root)::

    python examples/transformer_mujoco/torch_mujoco_transformer_ppo.py --task InvertedPendulum-v5 --model transformer
    python examples/transformer_mujoco/torch_mujoco_transformer_ppo.py --task InvertedPendulum-v5 --model mlp --video

Logs and checkpoints go to ``runs/transformer_mujoco/<task>_<model>_seed<seed>/``; with ``--video`` an eval rollout
(mp4) is recorded to ``runs/transformer_mujoco/<task>_<model>_seed<seed>/videos/``.
"""

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "glfw")  # offscreen rendering backend for --video (see README)

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from skrl.agents.torch.ppo import PPO, PPO_CFG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.models.torch.transformer import TransformerGaussianPolicy, TransformerValue
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed


TASKS = ["HalfCheetah-v5", "InvertedPendulum-v5", "InvertedDoublePendulum-v5"]

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="InvertedPendulum-v5", choices=TASKS)
parser.add_argument("--model", type=str, default="transformer", choices=["transformer", "mlp"])
parser.add_argument("--timesteps", type=int, default=100000, help="Total environment steps (summed over envs)")
parser.add_argument("--num-envs", type=int, default=8, help="Number of vectorized environments")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--video", action="store_true", help="Record an eval rollout (mp4) after training")
args = parser.parse_args()

set_seed(args.seed)

run_dir = f"runs/transformer_mujoco/{args.task}_{args.model}_seed{args.seed}"


# MLP baseline with the same sizes as the skrl pendulum example (2 hidden layers of 64, ReLU)
class MLPGaussianPolicy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        GaussianMixin.__init__(self, clip_actions=clip_actions, reduction="sum")
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_actions),
        )
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        return self.net(inputs["observations"]), {"log_std": self.log_std_parameter}


class MLPValue(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self)
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def compute(self, inputs, role):
        return self.net(inputs["observations"]), {}


# create the vectorized environment and wrap it for skrl
env = gym.make_vec(args.task, num_envs=args.num_envs, vectorization_mode="sync")
env = wrap_env(env)
device = env.device

memory = RandomMemory(memory_size=512, num_envs=env.num_envs, device=device)

if args.model == "transformer":
    policy_cls, value_cls = TransformerGaussianPolicy, TransformerValue
else:
    policy_cls, value_cls = MLPGaussianPolicy, MLPValue

models = {}
# no action clipping in the model: GaussianMixin computes log_prob on the clipped action, so a mean outside
# the bounds gives log_probs of -1e2..-1e3 and the PPO ratio overflows to inf/NaN. MuJoCo clamps to ctrlrange.
models["policy"] = policy_cls(env.observation_space, env.action_space, device, clip_actions=False)
models["value"] = value_cls(env.observation_space, env.action_space, device)

cfg = PPO_CFG()
cfg.rollouts = 512  # steps per env before each update
cfg.learning_epochs = 10
cfg.mini_batches = 32
cfg.discount_factor = 0.99
cfg.gae_lambda = 0.95
cfg.learning_rate = 3e-4
cfg.grad_norm_clip = 0.5
cfg.ratio_clip = 0.2
cfg.value_clip = 0.2
cfg.entropy_loss_scale = 0.0
cfg.value_loss_scale = 0.5
cfg.kl_threshold = 0
cfg.observation_preprocessor = RunningStandardScaler
cfg.observation_preprocessor_kwargs = {"size": env.observation_space, "device": device}
cfg.value_preprocessor = RunningStandardScaler
cfg.value_preprocessor_kwargs = {"size": 1, "device": device}
cfg.experiment.write_interval = "auto"
cfg.experiment.checkpoint_interval = "auto"
cfg.experiment.directory = os.path.dirname(run_dir)
cfg.experiment.experiment_name = os.path.basename(run_dir)

agent = PPO(
    models=models,
    memory=memory,
    cfg=cfg,
    observation_space=env.observation_space,
    state_space=env.state_space,
    action_space=env.action_space,
    device=device,
)

# the trainer counts vectorized steps, so divide the total env steps by the number of envs
trainer = SequentialTrainer(
    cfg={"timesteps": max(1, args.timesteps // args.num_envs), "headless": True, "disable_progressbar": True},
    env=env,
    agents=agent,
)
trainer.train()
env.close()


if args.video:
    video_dir = os.path.join(run_dir, "videos")
    # HalfCheetah's default camera is static and the cheetah leaves the frame: use its tracking camera
    camera = {"camera_name": "track"} if args.task == "HalfCheetah-v5" else {}
    eval_env = gym.make(args.task, render_mode="rgb_array", **camera)
    eval_env = gym.wrappers.RecordVideo(eval_env, video_folder=video_dir, name_prefix=args.task)
    obs, _ = eval_env.reset(seed=args.seed)
    episode_return, done = 0.0, False
    policy = agent.policy
    policy.eval()
    with torch.no_grad():
        while not done:
            x = agent._observation_preprocessor(torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
            actions, _ = policy.compute({"observations": x}, role="policy")  # deterministic: mean actions
            action = np.clip(actions[0].cpu().numpy(), eval_env.action_space.low, eval_env.action_space.high)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            episode_return += float(reward)
            done = terminated or truncated
    eval_env.close()
    print(f"[eval] {args.task} {args.model}: episode return {episode_return:.2f}, video in {video_dir}")
