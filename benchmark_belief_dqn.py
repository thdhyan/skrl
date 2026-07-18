"""Benchmark: BeliefDQN vs DQN vs PPO on CartPole POMDP.

Trains all three agents on same task for comparison:
- BeliefDQN: Uses particle filter belief
- DQN: Baseline, processes observations directly
- PPO: Policy gradient baseline

Logs to separate TensorBoard runs for comparison.
"""
from __future__ import annotations

import argparse
import time
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import gymnasium

from torch.utils.tensorboard import SummaryWriter

from skrl.agents.torch.dqn import DQN, DQN_CFG
from skrl.agents.torch.ppo import PPO, PPO_CFG
from skrl.agents.torch.belief_dqn import BeliefDQN, BeliefDQN_CFG, ResNetDuelingQNetwork
from skrl.belief.torch.particle_filter import ParticleFilter
from skrl.memories.torch import RandomMemory
from skrl.envs.wrappers.torch.pomdp_wrapper import PomdpWrapper
from skrl.models.torch import Model

import math

# -----------------------------------------------------------------------
# CartPole config (same as example)
# -----------------------------------------------------------------------
STATE_DIM = 4
OBS_DIM = 4
HIDDEN = [1, 3]
VISIBLE = [0, 2]
NG = 2
NP = 200
NC = 2
BELIEF_DIM = NG + 2 * NG * STATE_DIM

GRAVITY = 9.8
MASSCART = 1.0
MASSPOLE = 0.1
TOTAL_MASS = MASSCART + MASSPOLE
LENGTH = 0.5
POLEMASS_LENGTH = MASSPOLE * LENGTH
FORCE_MAG = 10.0
TAU = 0.02


def cartpole_step(state: torch.Tensor, action_force: float, noise_std: float = 0.01) -> torch.Tensor:
    squeeze = state.dim() == 1
    if squeeze:
        state = state.unsqueeze(0)

    x      = state[:, 0]
    x_dot  = state[:, 1]
    theta  = state[:, 2]
    theta_dot = state[:, 3]

    force = action_force
    costheta = torch.cos(theta)
    sintheta = torch.sin(theta)

    temp = (force + POLEMASS_LENGTH * theta_dot ** 2 * sintheta) / TOTAL_MASS
    theta_acc = (GRAVITY * sintheta - costheta * temp) / (
        LENGTH * (4.0 / 3.0 - MASSPOLE * costheta ** 2 / TOTAL_MASS)
    )
    x_acc = temp - POLEMASS_LENGTH * theta_acc * costheta / TOTAL_MASS

    x_new       = x     + TAU * x_dot
    x_dot_new   = x_dot + TAU * x_acc
    theta_new   = theta + TAU * theta_dot
    theta_dot_new = theta_dot + TAU * theta_acc

    next_state = torch.stack([x_new, x_dot_new, theta_new, theta_dot_new], dim=-1)
    noise = torch.randn_like(next_state) * noise_std
    next_state = next_state + noise

    if squeeze:
        return next_state.squeeze(0)
    return next_state


def make_motion_model(action_force_getter, noise_std: float = 0.02):
    def motion_model(particles: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        force = action_force_getter()
        return cartpole_step(particles, force, noise_std=noise_std)
    return motion_model


def make_measurement_model(sigma_pos: float = 0.05, sigma_angle: float = 0.02):
    def measurement_model(particles: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        pos_err   = (particles[:, 0] - obs[0]) / sigma_pos
        angle_err = (particles[:, 2] - obs[2]) / sigma_angle
        log_ll = -0.5 * (pos_err ** 2 + angle_err ** 2)
        log_ll -= math.log(sigma_pos * math.sqrt(2 * math.pi))
        log_ll -= math.log(sigma_angle * math.sqrt(2 * math.pi))
        return log_ll
    return measurement_model


# -----------------------------------------------------------------------
# Network definitions
# -----------------------------------------------------------------------
class SimpleQNetwork(Model):
    def __init__(self, obs_dim, action_dim, hidden_size=128, device="cpu"):
        super().__init__(device=device)
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )

    def forward(self, obs):
        return self.net(obs)


class SimpleActorCritic(Model):
    def __init__(self, obs_dim, action_dim, hidden_size=128, device="cpu"):
        super().__init__(device=device)
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_size, action_dim)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, obs):
        x = self.shared(obs)
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value


# -----------------------------------------------------------------------
# Training functions
# -----------------------------------------------------------------------
def train_belief_dqn(
    timesteps: int = 30_000,
    log_dir: str = "runs/benchmark_belief_dqn",
    device: str = "cpu",
    seed: int = 42,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    writer = SummaryWriter(log_dir=log_dir)
    start_time = time.time()

    base_env = gymnasium.make("CartPole-v1")
    env = PomdpWrapper(base_env, hidden_obs_indices=HIDDEN, obs_noise_std=0.02, seed=seed)

    current_action_force = [0.0]
    motion_model = make_motion_model(lambda: current_action_force[0], noise_std=0.05)
    measurement_model = make_measurement_model(sigma_pos=0.05, sigma_angle=0.02)

    def belief_factory():
        return ParticleFilter(
            num_particles=NP,
            state_dim=STATE_DIM,
            num_gmm_components=NG,
            motion_model_fn=motion_model,
            measurement_model_fn=measurement_model,
            device=device,
            resample_threshold=0.5,
        )

    obs_space = env.observation_space
    action_space = env.action_space
    belief_space = gymnasium.spaces.Box(
        low=-np.inf, high=np.inf, shape=(BELIEF_DIM,), dtype=np.float32
    )

    q_net   = ResNetDuelingQNetwork(belief_space, action_space, device, resnet_blocks=4, width=64, dueling=True)
    tgt_net = ResNetDuelingQNetwork(belief_space, action_space, device, resnet_blocks=4, width=64, dueling=True)
    memory = RandomMemory(memory_size=20_000, num_envs=1, device=device)

    cfg = BeliefDQN_CFG(
        belief_factory=belief_factory,
        num_controllers=NC,
        gmm_components=NG,
        batch_size=64,
        discount_factor=0.99,
        learning_rate=1e-4,
        l2_reg=0.01,
        random_timesteps=1_000,
        learning_starts=1_000,
        update_interval=1,
        target_update_interval=100,
        exploration_scheduler=lambda t, T: max(0.05, 1.0 - t / (T * 0.5)),
        n_substeps=1,
    )

    agent = BeliefDQN(
        models={"q_network": q_net, "target_q_network": tgt_net},
        memory=memory,
        observation_space=obs_space,
        action_space=action_space,
        device=device,
        cfg=cfg,
    )
    agent.init()

    global_step = 0
    episode = 0
    rewards = []

    while global_step < timesteps:
        episode += 1
        obs_np, _ = env.reset(seed=seed + episode)
        obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)

        full_obs = env.get_full_obs()
        prior = torch.zeros(NP, STATE_DIM, device=device)
        prior[:, 0] = torch.tensor(full_obs[0]).expand(NP) + torch.randn(NP) * 0.3
        prior[:, 1] = torch.randn(NP) * 1.0
        prior[:, 2] = torch.tensor(full_obs[2]).expand(NP) + torch.randn(NP) * 0.05
        prior[:, 3] = torch.randn(NP) * 0.5
        agent.reset_belief(prior)

        done = False
        ep_reward = 0.0
        ep_steps = 0

        while not done and global_step < timesteps:
            action_t, _ = agent.act(obs_t, None, timestep=global_step, timesteps=timesteps)
            action = int(action_t.item())
            current_action_force[0] = FORCE_MAG if action == 1 else -FORCE_MAG

            next_obs_np, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            full_obs = env.get_full_obs()

            next_obs_t = torch.tensor(next_obs_np, dtype=torch.float32, device=device).unsqueeze(0)
            reward_t  = torch.tensor([[float(reward)]], device=device)
            term_t    = torch.tensor([[terminated]], device=device)
            trunc_t   = torch.tensor([[truncated]], device=device)

            agent.record_transition(
                observations=obs_t,
                states=None,
                actions=action_t,
                rewards=reward_t,
                next_observations=next_obs_t,
                next_states=None,
                terminated=term_t,
                truncated=trunc_t,
                infos=info,
                timestep=global_step,
                timesteps=timesteps,
            )
            agent.post_interaction(timestep=global_step, timesteps=timesteps)

            obs_t = next_obs_t
            ep_reward += reward
            ep_steps += 1
            global_step += 1

        writer.add_scalar("Reward/episode_reward", ep_reward, episode)
        rewards.append(ep_reward)

        if episode % 10 == 0:
            print(f"BeliefDQN ep={episode:5d}  reward={ep_reward:.0f}  steps={global_step}")

    env.close()
    writer.close()
    elapsed = time.time() - start_time

    return {
        "agent": "BeliefDQN",
        "episodes": episode,
        "timesteps": global_step,
        "rewards": rewards,
        "time_sec": elapsed,
        "avg_reward": np.mean(rewards[-10:]) if len(rewards) > 0 else 0,
    }


def train_dqn(
    timesteps: int = 30_000,
    log_dir: str = "runs/benchmark_dqn",
    device: str = "cpu",
    seed: int = 42,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    writer = SummaryWriter(log_dir=log_dir)
    start_time = time.time()

    base_env = gymnasium.make("CartPole-v1")
    env = PomdpWrapper(base_env, hidden_obs_indices=HIDDEN, obs_noise_std=0.02, seed=seed)

    obs_space = env.observation_space
    action_space = env.action_space

    q_net = SimpleQNetwork(OBS_DIM, action_space.n, hidden_size=128, device=device).to(device)
    tgt_net = SimpleQNetwork(OBS_DIM, action_space.n, hidden_size=128, device=device).to(device)
    memory = RandomMemory(memory_size=20_000, num_envs=1, device=device)

    cfg = DQN_CFG(
        batch_size=64,
        discount_factor=0.99,
        learning_rate=1e-4,
        random_timesteps=1_000,
        learning_starts=1_000,
        update_interval=1,
        target_update_interval=100,
        exploration_scheduler=lambda t, T: max(0.05, 1.0 - t / (T * 0.5)),
    )

    agent = DQN(
        models={"q_network": q_net, "target_q_network": tgt_net},
        memory=memory,
        observation_space=obs_space,
        action_space=action_space,
        device=device,
        cfg=cfg,
    )
    agent.init()

    global_step = 0
    episode = 0
    rewards = []

    while global_step < timesteps:
        episode += 1
        obs_np, _ = env.reset(seed=seed + episode)
        obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)

        done = False
        ep_reward = 0.0

        while not done and global_step < timesteps:
            action_t, _ = agent.act(obs_t, None, timestep=global_step, timesteps=timesteps)
            action = int(action_t.item())

            next_obs_np, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            next_obs_t = torch.tensor(next_obs_np, dtype=torch.float32, device=device).unsqueeze(0)
            reward_t  = torch.tensor([[float(reward)]], device=device)
            term_t    = torch.tensor([[terminated]], device=device)
            trunc_t   = torch.tensor([[truncated]], device=device)

            agent.record_transition(
                observations=obs_t,
                states=None,
                actions=action_t,
                rewards=reward_t,
                next_observations=next_obs_t,
                next_states=None,
                terminated=term_t,
                truncated=trunc_t,
                infos=info,
                timestep=global_step,
                timesteps=timesteps,
            )
            agent.post_interaction(timestep=global_step, timesteps=timesteps)

            obs_t = next_obs_t
            ep_reward += reward
            global_step += 1

        writer.add_scalar("Reward/episode_reward", ep_reward, episode)
        rewards.append(ep_reward)

        if episode % 10 == 0:
            print(f"DQN      ep={episode:5d}  reward={ep_reward:.0f}  steps={global_step}")

    env.close()
    writer.close()
    elapsed = time.time() - start_time

    return {
        "agent": "DQN",
        "episodes": episode,
        "timesteps": global_step,
        "rewards": rewards,
        "time_sec": elapsed,
        "avg_reward": np.mean(rewards[-10:]) if len(rewards) > 0 else 0,
    }


def train_ppo(
    timesteps: int = 30_000,
    log_dir: str = "runs/benchmark_ppo",
    device: str = "cpu",
    seed: int = 42,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    writer = SummaryWriter(log_dir=log_dir)
    start_time = time.time()

    base_env = gymnasium.make("CartPole-v1")
    env = PomdpWrapper(base_env, hidden_obs_indices=HIDDEN, obs_noise_std=0.02, seed=seed)

    obs_space = env.observation_space
    action_space = env.action_space

    model = SimpleActorCritic(OBS_DIM, action_space.n, hidden_size=128, device=device).to(device)

    cfg = PPO_CFG(
        batch_size=32,
        mini_batches=4,
        discount_factor=0.99,
        learning_rate=1e-4,
        learning_epochs=3,
        entropy_loss_scale=0.01,
        value_loss_scale=0.5,
        random_timesteps=500,
        learning_starts=500,
    )

    agent = PPO(
        models={"policy": model},
        memory=None,
        observation_space=obs_space,
        action_space=action_space,
        device=device,
        cfg=cfg,
    )
    agent.init()

    global_step = 0
    episode = 0
    rewards = []

    while global_step < timesteps:
        episode += 1
        obs_np, _ = env.reset(seed=seed + episode)
        obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)

        done = False
        ep_reward = 0.0

        while not done and global_step < timesteps:
            action_t, _ = agent.act(obs_t, None, timestep=global_step, timesteps=timesteps)
            action = int(action_t.item())

            next_obs_np, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            next_obs_t = torch.tensor(next_obs_np, dtype=torch.float32, device=device).unsqueeze(0)
            reward_t  = torch.tensor([[float(reward)]], device=device)
            term_t    = torch.tensor([[terminated]], device=device)
            trunc_t   = torch.tensor([[truncated]], device=device)

            agent.record_transition(
                observations=obs_t,
                states=None,
                actions=action_t,
                rewards=reward_t,
                next_observations=next_obs_t,
                next_states=None,
                terminated=term_t,
                truncated=trunc_t,
                infos=info,
                timestep=global_step,
                timesteps=timesteps,
            )
            agent.post_interaction(timestep=global_step, timesteps=timesteps)

            obs_t = next_obs_t
            ep_reward += reward
            global_step += 1

        writer.add_scalar("Reward/episode_reward", ep_reward, episode)
        rewards.append(ep_reward)

        if episode % 10 == 0:
            print(f"PPO      ep={episode:5d}  reward={ep_reward:.0f}  steps={global_step}")

    env.close()
    writer.close()
    elapsed = time.time() - start_time

    return {
        "agent": "PPO",
        "episodes": episode,
        "timesteps": global_step,
        "rewards": rewards,
        "time_sec": elapsed,
        "avg_reward": np.mean(rewards[-10:]) if len(rewards) > 0 else 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps",  type=int,  default=30_000)
    parser.add_argument("--device",     default="cpu")
    parser.add_argument("--seed",       type=int,  default=42)
    args = parser.parse_args()

    print("=" * 70)
    print("BENCHMARK: BeliefDQN vs DQN vs PPO on CartPole POMDP")
    print("=" * 70)

    print("\n[1/3] Training BeliefDQN...")
    belief_dqn_result = train_belief_dqn(
        timesteps=args.timesteps,
        log_dir="runs/benchmark_belief_dqn",
        device=args.device,
        seed=args.seed,
    )

    print("\n[2/3] Training DQN...")
    dqn_result = train_dqn(
        timesteps=args.timesteps,
        log_dir="runs/benchmark_dqn",
        device=args.device,
        seed=args.seed,
    )

    print("\n[3/3] Training PPO...")
    ppo_result = train_ppo(
        timesteps=args.timesteps,
        log_dir="runs/benchmark_ppo",
        device=args.device,
        seed=args.seed,
    )

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\n{'Agent':<15} {'Episodes':<12} {'Avg Reward':<15} {'Time (s)':<12}")
    print("-" * 70)
    for result in [belief_dqn_result, dqn_result, ppo_result]:
        print(f"{result['agent']:<15} {result['episodes']:<12} {result['avg_reward']:<15.1f} {result['time_sec']:<12.1f}")

    print("\n" + "=" * 70)
    print("TensorBoard: tensorboard --logdir runs/")
    print("=" * 70)
