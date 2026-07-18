"""CartPole-v1 under partial observability with BeliefDQN.

Partial obs: cart velocity (idx 1) and pole angular velocity (idx 3) hidden.
Agent sees only [cart_pos, pole_angle] and must infer velocities via PF.

Visualisation window shows:
  - Left:  CartPole rendered frame
  - Right: belief particle cloud (cart_pos vs cart_vel) + GMM ellipses
            + ground truth state (red cross)

TensorBoard logs (run: tensorboard --logdir runs/):
  - Reward / episode_reward
  - Belief / entropy
  - Belief / compress_ms
  - Belief / gt_cart_pos, gt_cart_vel, gt_pole_angle, gt_pole_vel
  - Belief / mean_cart_pos_mode0, Belief / mean_cart_vel_mode0
  - Belief / mean_cart_pos_mode1, Belief / mean_cart_vel_mode1
  - Exploration / epsilon
  - Loss / q_loss

Usage:
    python examples/contact_rich_uncertainty/cartpole_pomdp_belief_dqn.py
    python examples/contact_rich_uncertainty/cartpole_pomdp_belief_dqn.py --no-render
    python examples/contact_rich_uncertainty/cartpole_pomdp_belief_dqn.py --timesteps 50000
"""
from __future__ import annotations

import argparse
import math
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import gymnasium

from torch.utils.tensorboard import SummaryWriter

from skrl.agents.torch.belief_dqn import BeliefDQN, BeliefDQN_CFG, ResNetDuelingQNetwork
from skrl.belief.torch.particle_filter import ParticleFilter
from skrl.memories.torch import RandomMemory
from skrl.envs.wrappers.torch.pomdp_wrapper import PomdpWrapper


# -----------------------------------------------------------------------
# CartPole state: [cart_pos, cart_vel, pole_angle, pole_ang_vel]
# Hidden:         [cart_vel (1), pole_ang_vel (3)]
# Visible:        [cart_pos (0), pole_angle (2)]
# -----------------------------------------------------------------------
STATE_DIM = 4          # full CartPole state
OBS_DIM = 4            # masked obs still 4-dim (hidden = 0)
HIDDEN = [1, 3]        # indices to zero out
VISIBLE = [0, 2]
NG = 2                 # GMM components
NP = 200               # particles
NC = 2                 # CartPole actions: push left / push right
BELIEF_DIM = NG + 2 * NG * STATE_DIM  # 2 + 2*2*4 = 18


# -----------------------------------------------------------------------
# CartPole dynamics (analytical, no physics engine needed)
# -----------------------------------------------------------------------
GRAVITY = 9.8
MASSCART = 1.0
MASSPOLE = 0.1
TOTAL_MASS = MASSCART + MASSPOLE
LENGTH = 0.5
POLEMASS_LENGTH = MASSPOLE * LENGTH
FORCE_MAG = 10.0
TAU = 0.02  # seconds per step


def cartpole_step(state: torch.Tensor, action_force: float, noise_std: float = 0.01) -> torch.Tensor:
    """Euler-integrated CartPole dynamics for particle propagation.

    :param state: ``[NP, 4]`` or ``[4]``
    :param action_force: +FORCE_MAG (right) or -FORCE_MAG (left)
    :param noise_std: Process noise std
    :return: next state, same shape as input
    """
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

    # add process noise
    noise = torch.randn_like(next_state) * noise_std
    next_state = next_state + noise

    if squeeze:
        return next_state.squeeze(0)
    return next_state


def make_motion_model(action_force_getter, noise_std: float = 0.02):
    """Returns a motion_model_fn(particles, control) -> particles_next.

    action_force_getter() must return current force (+/- FORCE_MAG).
    """
    def motion_model(particles: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        force = action_force_getter()
        return cartpole_step(particles, force, noise_std=noise_std)
    return motion_model


def make_measurement_model(sigma_pos: float = 0.05, sigma_angle: float = 0.02):
    """Returns measurement_model_fn(particles, obs) -> log_likelihoods.

    Measurement: noisy [cart_pos, pole_angle] (visible indices 0 and 2).
    """
    def measurement_model(particles: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        # obs: [4], but only indices 0 and 2 are non-zero (POMDP mask)
        pos_err   = (particles[:, 0] - obs[0]) / sigma_pos
        angle_err = (particles[:, 2] - obs[2]) / sigma_angle
        log_ll = -0.5 * (pos_err ** 2 + angle_err ** 2)
        log_ll -= math.log(sigma_pos * math.sqrt(2 * math.pi))
        log_ll -= math.log(sigma_angle * math.sqrt(2 * math.pi))
        return log_ll
    return measurement_model


# -----------------------------------------------------------------------
# Visualisation (matplotlib, non-blocking)
# -----------------------------------------------------------------------
class BeliefVisualiser:
    """Live matplotlib window: CartPole frame + particle/GMM belief plot."""

    def __init__(self, enabled: bool = True) -> None:
        self._enabled = enabled
        if not enabled:
            return
        try:
            import matplotlib
            matplotlib.use("TkAgg")  # non-blocking backend
        except Exception:
            try:
                import matplotlib
                matplotlib.use("Qt5Agg")
            except Exception:
                pass
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.patches import Ellipse

        self._plt = plt
        self._Ellipse = Ellipse

        self._fig, (self._ax_env, self._ax_belief) = plt.subplots(1, 2, figsize=(12, 5))
        self._fig.suptitle("BeliefDQN — CartPole POMDP", fontsize=13)
        self._ax_belief.set_xlabel("Cart position")
        self._ax_belief.set_ylabel("Cart velocity")
        self._ax_belief.set_title("Particle belief (pos vs vel)")
        self._ax_env.set_title("CartPole")
        self._ax_env.axis("off")
        self._plt.ion()
        self._plt.show()
        self._img_obj = None

    def update(
        self,
        frame: np.ndarray | None,
        particles: torch.Tensor,
        gmm,
        gt_state: np.ndarray,
        episode: int,
        step: int,
        entropy: float,
    ) -> None:
        if not self._enabled:
            return
        plt = self._plt
        Ellipse = self._Ellipse

        # env frame
        ax_env = self._ax_env
        ax_belief = self._ax_belief

        if frame is not None:
            if self._img_obj is None:
                self._img_obj = ax_env.imshow(frame)
            else:
                self._img_obj.set_data(frame)

        # particle cloud (cart_pos vs cart_vel)
        ax_belief.cla()
        ax_belief.set_xlabel("Cart position")
        ax_belief.set_ylabel("Cart velocity")
        ax_belief.set_title(f"ep={episode} step={step}  H={entropy:.2f}")

        pts = particles.cpu().numpy()
        ax_belief.scatter(pts[:, 0], pts[:, 1], s=4, alpha=0.3, c="steelblue", label="particles")

        # GMM ellipses (2-σ) for each mode
        colors = ["darkorange", "green"]
        if gmm is not None:
            for i in range(gmm.num_components):
                h = gmm.hefts[i].item()
                mx = gmm.means[i, 0].item()
                my = gmm.means[i, 1].item()
                sx = gmm.log_diag_covs[i, 0].exp().sqrt().item() * 2
                sy = gmm.log_diag_covs[i, 1].exp().sqrt().item() * 2
                e = Ellipse(
                    (mx, my), width=sx * 2, height=sy * 2,
                    color=colors[i % len(colors)], alpha=0.25
                )
                ax_belief.add_patch(e)
                ax_belief.plot(mx, my, "x", color=colors[i % len(colors)],
                               markersize=10, markeredgewidth=2,
                               label=f"GMM mode {i} (h={h:.2f})")

        # ground truth
        ax_belief.plot(gt_state[0], gt_state[1], "r*", markersize=14, label="ground truth")
        ax_belief.legend(fontsize=7, loc="upper right")
        ax_belief.set_xlim(-3, 3)
        ax_belief.set_ylim(-6, 6)

        self._fig.tight_layout()
        plt.pause(0.001)

    def close(self) -> None:
        if self._enabled:
            self._plt.close(self._fig)


# -----------------------------------------------------------------------
# Main training loop
# -----------------------------------------------------------------------

def run(
    timesteps: int = 30_000,
    render: bool = True,
    log_dir: str = "runs/cartpole_pomdp_belief_dqn",
    device: str = "cpu",
    seed: int = 42,
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard: tensorboard --logdir {log_dir}")

    # ---- env ----
    base_env = gymnasium.make("CartPole-v1", render_mode="rgb_array" if render else None)
    env = PomdpWrapper(base_env, hidden_obs_indices=HIDDEN, obs_noise_std=0.02, seed=seed)

    # ---- belief ----
    current_action_force = [0.0]  # mutable closure cell

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

    # ---- agent ----
    obs_space    = env.observation_space
    action_space = env.action_space
    belief_space = gymnasium.spaces.Box(
        low=-np.inf, high=np.inf, shape=(BELIEF_DIM,), dtype=np.float32
    )

    q_net   = ResNetDuelingQNetwork(belief_space, action_space, device,
                                    resnet_blocks=4, width=64, dueling=True)
    tgt_net = ResNetDuelingQNetwork(belief_space, action_space, device,
                                    resnet_blocks=4, width=64, dueling=True)

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

    vis = BeliefVisualiser(enabled=render)

    # ---- training loop ----
    global_step = 0
    episode = 0

    while global_step < timesteps:
        episode += 1
        obs_np, _ = env.reset(seed=seed + episode)
        obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)

        # prior: wide uniform particles (unknown initial velocity)
        full_obs = env.get_full_obs()
        prior = torch.zeros(NP, STATE_DIM, device=device)
        prior[:, 0] = torch.tensor(full_obs[0]).expand(NP) + torch.randn(NP) * 0.3
        prior[:, 1] = torch.randn(NP) * 1.0   # unknown velocity
        prior[:, 2] = torch.tensor(full_obs[2]).expand(NP) + torch.randn(NP) * 0.05
        prior[:, 3] = torch.randn(NP) * 0.5   # unknown ang vel
        agent.reset_belief(prior)

        done = False
        ep_reward = 0.0
        ep_steps = 0
        frame = None

        while not done and global_step < timesteps:
            # render
            if render:
                try:
                    frame = base_env.render()
                except Exception:
                    frame = None

            # act
            action_t, _ = agent.act(obs_t, None, timestep=global_step, timesteps=timesteps)
            action = int(action_t.item())

            # update force for motion model closure
            current_action_force[0] = FORCE_MAG if action == 1 else -FORCE_MAG

            # env step
            next_obs_np, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            full_obs = env.get_full_obs()  # ground truth

            next_obs_t = torch.tensor(next_obs_np, dtype=torch.float32, device=device).unsqueeze(0)
            reward_t  = torch.tensor([[float(reward)]], device=device)
            term_t    = torch.tensor([[terminated]], device=device)
            trunc_t   = torch.tensor([[truncated]], device=device)

            # record → drives belief.predict + believe.update internally
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

            # ---- TensorBoard: belief vs ground truth ----
            belief = agent._belief
            gt = full_obs  # [cart_pos, cart_vel, pole_angle, pole_vel]
            gmm = belief.gmm

            if gmm is not None:
                writer.add_scalar("Belief/gt_cart_pos",   gt[0], global_step)
                writer.add_scalar("Belief/gt_cart_vel",   gt[1], global_step)
                writer.add_scalar("Belief/gt_pole_angle", gt[2], global_step)
                writer.add_scalar("Belief/gt_pole_vel",   gt[3], global_step)

                for m_i in range(NG):
                    mu = gmm.means[m_i].cpu()
                    writer.add_scalar(f"Belief/mode{m_i}_cart_pos",   mu[0].item(), global_step)
                    writer.add_scalar(f"Belief/mode{m_i}_cart_vel",   mu[1].item(), global_step)
                    writer.add_scalar(f"Belief/mode{m_i}_pole_angle", mu[2].item(), global_step)

                entropy = belief.get_entropy().item()
                writer.add_scalar("Belief/entropy", entropy, global_step)

                # pos error: best GMM mode mean vs ground truth
                best_mode = gmm.hefts.argmax().item()
                pos_err = abs(gmm.means[best_mode, 0].item() - gt[0])
                vel_err = abs(gmm.means[best_mode, 1].item() - gt[1])
                writer.add_scalar("Belief/pos_error_best_mode", pos_err, global_step)
                writer.add_scalar("Belief/vel_error_best_mode", vel_err, global_step)

            # visualise
            if render and global_step % 5 == 0 and gmm is not None:
                particles = belief.get_particles()
                entropy = belief.get_entropy().item()
                vis.update(frame, particles, gmm, full_obs, episode, ep_steps, entropy)

            obs_t = next_obs_t
            ep_reward += reward
            ep_steps += 1
            global_step += 1

        writer.add_scalar("Reward/episode_reward", ep_reward, episode)
        writer.add_scalar("Reward/episode_length", ep_steps, episode)

        if episode % 10 == 0:
            print(f"ep={episode:5d}  steps={ep_steps:4d}  reward={ep_reward:.0f}  "
                  f"global_step={global_step}")

    env.close()
    vis.close()
    writer.close()
    print(f"Done. {episode} episodes, {global_step} steps.")
    print(f"Run: tensorboard --logdir {log_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps",  type=int,  default=30_000)
    parser.add_argument("--no-render",  action="store_true")
    parser.add_argument("--log-dir",    default="runs/cartpole_pomdp_belief_dqn")
    parser.add_argument("--device",     default="cpu")
    parser.add_argument("--seed",       type=int,  default=42)
    args = parser.parse_args()

    run(
        timesteps=args.timesteps,
        render=not args.no_render,
        log_dir=args.log_dir,
        device=args.device,
        seed=args.seed,
    )
