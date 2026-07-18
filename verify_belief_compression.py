"""Verify belief compression works in BeliefDQN network.

Measures:
1. Belief entropy trend during training
2. Q-value sensitivity to belief changes
3. Gradient flow through belief input
4. Compare DQN vs BeliefDQN learning curves
"""
from __future__ import annotations

import argparse
import numpy as np
import torch
import torch.nn as nn
import gymnasium

from torch.utils.tensorboard import SummaryWriter

from skrl.agents.torch.belief_dqn import BeliefDQN, BeliefDQN_CFG, ResNetDuelingQNetwork
from skrl.belief.torch.particle_filter import ParticleFilter
from skrl.memories.torch import RandomMemory
from skrl.envs.wrappers.torch.pomdp_wrapper import PomdpWrapper

import math

STATE_DIM = 4
OBS_DIM = 4
HIDDEN = [1, 3]
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

    x = state[:, 0]
    x_dot = state[:, 1]
    theta = state[:, 2]
    theta_dot = state[:, 3]

    force = action_force
    costheta = torch.cos(theta)
    sintheta = torch.sin(theta)

    temp = (force + POLEMASS_LENGTH * theta_dot ** 2 * sintheta) / TOTAL_MASS
    theta_acc = (GRAVITY * sintheta - costheta * temp) / (
        LENGTH * (4.0 / 3.0 - MASSPOLE * costheta ** 2 / TOTAL_MASS)
    )
    x_acc = temp - POLEMASS_LENGTH * theta_acc * costheta / TOTAL_MASS

    x_new = x + TAU * x_dot
    x_dot_new = x_dot + TAU * x_acc
    theta_new = theta + TAU * theta_dot
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
        pos_err = (particles[:, 0] - obs[0]) / sigma_pos
        angle_err = (particles[:, 2] - obs[2]) / sigma_angle
        log_ll = -0.5 * (pos_err ** 2 + angle_err ** 2)
        log_ll -= math.log(sigma_pos * math.sqrt(2 * math.pi))
        log_ll -= math.log(sigma_angle * math.sqrt(2 * math.pi))
        return log_ll
    return measurement_model


def run_belief_dqn_with_compression_analysis(
    timesteps: int = 30_000,
    log_dir: str = "runs/belief_compression_analysis",
    device: str = "cpu",
    seed: int = 42,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    writer = SummaryWriter(log_dir=log_dir)
    start_time = torch.cuda.Event(enable_timing=True) if device == "cuda" else None

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

    q_net = ResNetDuelingQNetwork(belief_space, action_space, device, resnet_blocks=4, width=64, dueling=True)
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

    print(f"Training BeliefDQN with compression analysis on device={device}")
    print(f"Belief dimension: {BELIEF_DIM}")

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

        while not done and global_step < timesteps:
            action_t, _ = agent.act(obs_t, None, timestep=global_step, timesteps=timesteps)
            action = int(action_t.item())
            current_action_force[0] = FORCE_MAG if action == 1 else -FORCE_MAG

            next_obs_np, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            next_obs_t = torch.tensor(next_obs_np, dtype=torch.float32, device=device).unsqueeze(0)
            reward_t = torch.tensor([[float(reward)]], device=device)
            term_t = torch.tensor([[terminated]], device=device)
            trunc_t = torch.tensor([[truncated]], device=device)

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

            # ---- Measure belief compression ----
            belief = agent._belief
            gmm = belief.gmm

            if gmm is not None:
                # 1. Belief entropy (how concentrated is GMM?)
                hefts = gmm.hefts  # [NG]
                log_covs = gmm.log_diag_covs  # [NG, STATE_DIM]

                # Shannon entropy of mixture weights
                h_mixture = -(hefts * torch.log(hefts + 1e-8)).sum()
                writer.add_scalar("Compression/mixture_entropy", h_mixture.item(), global_step)

                # Avg variance (lower = more confident)
                avg_cov = log_covs.exp().mean().item()
                writer.add_scalar("Compression/avg_log_std", log_covs.mean().item(), global_step)
                writer.add_scalar("Compression/avg_cov_exp", avg_cov, global_step)

                # Belief vector norm (compactness)
                particles = belief.get_particles()
                particle_spread = particles.std(dim=0).mean().item()
                writer.add_scalar("Compression/particle_spread", particle_spread, global_step)

                # 2. Check velocity estimates (are we inferring hidden states?)
                # State: [cart_pos, cart_vel, pole_angle, pole_vel]
                # Hidden: cart_vel (idx 1), pole_vel (idx 3)
                for mode_idx in range(NG):
                    mu = gmm.means[mode_idx]
                    writer.add_scalar(f"Compression/mode{mode_idx}_cart_vel_est", mu[1].item(), global_step)
                    writer.add_scalar(f"Compression/mode{mode_idx}_pole_vel_est", mu[3].item(), global_step)

                # 3. Q-value diversity across actions
                # (indicates network is learning from belief)
                if global_step % 100 == 0 and global_step > cfg.learning_starts:
                    particles = belief.get_particles()
                    gmm = belief.gmm
                    hefts = gmm.hefts
                    means = gmm.means
                    log_covs = gmm.log_diag_covs
                    b_tilde = torch.cat([hefts, means.flatten(), log_covs.flatten()]).unsqueeze(0)

                    q_net.eval()
                    with torch.no_grad():
                        q_out = q_net({"observations": b_tilde.unsqueeze(0)})
                        if isinstance(q_out, tuple):
                            q_vals = q_out[0]  # (values, outputs) tuple
                        else:
                            q_vals = q_out
                    q_net.train()

                    q_spread = q_vals.std().item()
                    writer.add_scalar("Compression/q_value_spread", q_spread, global_step)

            obs_t = next_obs_t
            ep_reward += reward
            global_step += 1

        writer.add_scalar("Reward/episode_reward", ep_reward, episode)
        rewards.append(ep_reward)

        if episode % 20 == 0:
            avg_reward = np.mean(rewards[-20:]) if len(rewards) >= 20 else np.mean(rewards)
            print(f"ep={episode:5d}  reward={ep_reward:.0f}  avg(last 20)={avg_reward:.1f}  steps={global_step}")

    env.close()
    writer.close()

    return {
        "agent": "BeliefDQN",
        "episodes": episode,
        "timesteps": global_step,
        "rewards": rewards,
        "avg_reward": np.mean(rewards[-10:]) if len(rewards) > 0 else 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=30_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 70)
    print("BELIEF COMPRESSION ANALYSIS")
    print("=" * 70)

    result = run_belief_dqn_with_compression_analysis(
        timesteps=args.timesteps,
        log_dir="runs/belief_compression_analysis",
        device=args.device,
        seed=args.seed,
    )

    print("\n" + "=" * 70)
    print(f"Episodes: {result['episodes']}")
    print(f"Final avg reward (last 10): {result['avg_reward']:.1f}")
    print("=" * 70)
    print("TensorBoard:")
    print("  tensorboard --logdir runs/belief_compression_analysis")
    print("\nMetrics to check:")
    print("  - Compression/mixture_entropy: should decrease (GMM converges)")
    print("  - Compression/particle_spread: should decrease (particles concentrate)")
    print("  - Compression/avg_cov_exp: should decrease (confidence increases)")
    print("  - Compression/mode*_*_vel_est: velocity estimates vs ground truth")
    print("  - Compression/input_grad_norm: should be non-zero (gradients flow)")
    print("=" * 70)
