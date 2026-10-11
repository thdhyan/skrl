"""BeliefPPO (agent-owned particle filter, Pattern C) on CartPole-POMDP.

Same POMDP and filter models as `cartpole_pomdp_belief_ppo.py` (cart velocity +
pole angular velocity hidden, obs noise 0.02), but instead of wrapping the env
the `BeliefPPO` agent owns the filter: the env hands the agent raw masked obs,
the agent runs predict/update/compress itself and trains on b̃.

Usage:
    python examples/contact_rich_uncertainty/cartpole_pomdp_belief_ppo_agent.py --timesteps 20000
    tensorboard --logdir runs/torch/CartPolePOMDP_BeliefPPOAgent
"""
from __future__ import annotations

import argparse

import torch
import gymnasium

import cartpole_pomdp_belief_ppo as M  # dynamics, measurement model, constants
from skrl.agents.torch.belief_ppo import BeliefPPO, BeliefPPO_CFG, ResNetCategoricalPolicy, ResNetValue
from skrl.belief.torch.particle_filter import ParticleFilter
from skrl.envs.wrappers.torch import wrap_env
from skrl.envs.wrappers.torch.pomdp_wrapper import PomdpWrapper
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed


def make_belief_factory(device: str):
    def motion_model(particles: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        # control = discrete action index from the policy (0 → push left, 1 → push right)
        force = M.FORCE_MAG if float(control.reshape(-1)[0]) > 0.5 else -M.FORCE_MAG
        return M.cartpole_step(particles, force, noise_std=0.05)

    def factory():
        return ParticleFilter(
            num_particles=M.NP,
            state_dim=M.STATE_DIM,
            num_gmm_components=M.NG,
            motion_model_fn=motion_model,
            measurement_model_fn=M.make_measurement_model(),
            device=device,
            resample_threshold=0.5,
        )

    return factory


def make_prior_fn(device: str):
    def prior(obs: torch.Tensor) -> torch.Tensor:
        # visible dims (cart_pos, pole_angle) anchor the prior; hidden velocities are broad
        n = M.NP
        p = torch.zeros(n, M.STATE_DIM, device=device)
        p[:, 0] = float(obs[0]) + torch.randn(n, device=device) * 0.3
        p[:, 1] = torch.randn(n, device=device) * 1.0
        p[:, 2] = float(obs[2]) + torch.randn(n, device=device) * 0.05
        p[:, 3] = torch.randn(n, device=device) * 0.5
        return p

    return prior


def run(timesteps: int, seed: int, device: str, log_dir: str) -> None:
    set_seed(seed)

    env = PomdpWrapper(gymnasium.make("CartPole-v1"), hidden_obs_indices=M.HIDDEN, obs_noise_std=0.02, seed=seed)
    env = wrap_env(env)
    device = env.device

    belief_space = gymnasium.spaces.Box(low=-float("inf"), high=float("inf"), shape=(M.BELIEF_DIM,))
    models = {
        "policy": ResNetCategoricalPolicy(belief_space, env.action_space, device, resnet_blocks=2, width=64),
        "value": ResNetValue(belief_space, env.action_space, device, resnet_blocks=2, width=64),
    }

    cfg = BeliefPPO_CFG(
        belief_factory=make_belief_factory(str(device)),
        belief_prior_fn=make_prior_fn(str(device)),
        rollouts=256,
        learning_epochs=8,
        mini_batches=4,
        discount_factor=0.99,
        gae_lambda=0.95,
        learning_rate=1e-3,
        entropy_loss_scale=0.01,
        value_loss_scale=0.5,
        observation_preprocessor=RunningStandardScaler,
        observation_preprocessor_kwargs={"size": M.BELIEF_DIM, "device": device},
        value_preprocessor=RunningStandardScaler,
        value_preprocessor_kwargs={"size": 1, "device": device},
    )
    cfg.experiment.directory = log_dir
    cfg.experiment.write_interval = "auto"
    cfg.experiment.checkpoint_interval = "auto"

    agent = BeliefPPO(
        models=models,
        memory=RandomMemory(memory_size=cfg.rollouts, num_envs=1, device=device),
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        cfg=cfg,
    )

    print(f"TensorBoard: tensorboard --logdir {log_dir}")
    SequentialTrainer(cfg={"timesteps": timesteps, "headless": True}, env=env, agents=agent).train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log-dir", default="runs/torch/CartPolePOMDP_BeliefPPOAgent")
    args = parser.parse_args()
    run(args.timesteps, args.seed, args.device, args.log_dir)
