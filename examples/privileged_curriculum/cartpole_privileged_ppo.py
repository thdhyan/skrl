"""PrivilegedPPO on CartPole-v1 POMDP with domain randomization.

The policy sees cart position and pole angle (velocities hidden, noisy obs) plus a history of them.
Privileged values: cart velocity, pole angular velocity, pole mass, pole length (mass and length are
randomized per episode). The mask curriculum moves training from fully observable (actor uses the
privileged encoder) to fully masked (actor uses the history estimator).

    python examples/privileged_curriculum/cartpole_privileged_ppo.py --mode curriculum
    python examples/privileged_curriculum/cartpole_privileged_ppo.py --mode masked    # history only, no curriculum
    python examples/privileged_curriculum/cartpole_privileged_ppo.py --mode visible   # privileged always visible

At the end, the policy is evaluated with the privileged values masked and visible.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import gymnasium

from skrl.agents.torch.privileged_ppo import (
    ObsLayout,
    PrivilegedCategoricalPolicy,
    PrivilegedPPO,
    PrivilegedPPO_CFG,
    PrivilegedValue,
    SoftmaxMaskCurriculum,
)
from skrl.envs.wrappers.torch import wrap_env
from skrl.envs.wrappers.torch.pomdp_wrapper import PomdpWrapper
from skrl.envs.wrappers.torch.privileged_history_wrapper import PrivilegedHistoryWrapper
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed


HISTORY = 8
MASSPOLE_RANGE = (0.05, 0.5)
LENGTH_RANGE = (0.3, 0.8)
LAYOUT = ObsLayout(obs_dim=4, history_length=HISTORY, privileged_dim=4)


class FixedMask:
    """Curriculum stand-in with a constant mask probability (baselines and evaluation)."""

    def __init__(self, p: float) -> None:
        self.p = p

    def mask_probability(self) -> float:
        return self.p

    def set_progress(self, progress: float) -> None:
        pass


def randomize(env: gymnasium.Env, rng: np.random.Generator) -> None:
    cp = env.unwrapped
    cp.masspole = rng.uniform(*MASSPOLE_RANGE)
    cp.length = rng.uniform(*LENGTH_RANGE)
    cp.total_mass = cp.masspole + cp.masscart
    cp.polemass_length = cp.masspole * cp.length


def privileged(env: gymnasium.Env) -> np.ndarray:
    cp = env.unwrapped
    return np.array([cp.state[1], cp.state[3], cp.masspole, cp.length], dtype=np.float32)


def make_env(curriculum, seed: int) -> gymnasium.Env:
    env = gymnasium.make("CartPole-v1")
    env = PomdpWrapper(env, hidden_obs_indices=[1, 3], obs_noise_std=0.02, seed=seed)
    return PrivilegedHistoryWrapper(
        env,
        privileged_fn=privileged,
        privileged_dim=LAYOUT.privileged_dim,
        history_length=HISTORY,
        curriculum=curriculum,
        randomize_fn=randomize,
        seed=seed,
    )


@torch.no_grad()
def evaluate(policy, mask: float, episodes: int, seed: int) -> float:
    env = make_env(FixedMask(mask), seed)
    returns = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, total = False, 0.0
        while not done:
            x = torch.as_tensor(obs, dtype=torch.float32, device=policy.device)[None]
            logits = policy.compute({"observations": x})[0]
            obs, r, terminated, truncated, _ = env.step(int(logits.argmax()))
            total += r
            done = terminated or truncated
        returns.append(total)
    return float(np.mean(returns))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["curriculum", "masked", "visible"], default="curriculum")
    parser.add_argument("--timesteps", type=int, default=200_000, help="total env steps")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--log-dir", default="runs/privileged_curriculum")
    args = parser.parse_args()
    set_seed(args.seed)

    curriculum = {
        "curriculum": SoftmaxMaskCurriculum(start=0.1, end=0.7, sharpness=10.0),
        "masked": FixedMask(1.0),
        "visible": FixedMask(0.0),
    }[args.mode]

    env = gymnasium.vector.SyncVectorEnv(
        [lambda i=i: make_env(curriculum, args.seed + i) for i in range(args.num_envs)],
        autoreset_mode=gymnasium.vector.AutoresetMode.SAME_STEP,
    )
    env = wrap_env(env)
    device = env.device

    models = {
        "policy": PrivilegedCategoricalPolicy(env.observation_space, env.action_space, device, layout=LAYOUT),
        "value": PrivilegedValue(env.observation_space, env.action_space, device, layout=LAYOUT),
    }
    rollouts = 128
    cfg = PrivilegedPPO_CFG(
        curriculum=curriculum,
        rollouts=rollouts,
        learning_epochs=8,
        mini_batches=4,
        discount_factor=0.99,
        learning_rate=1e-3,
        entropy_loss_scale=0.01,
        value_loss_scale=0.5,
    )
    cfg.experiment.directory = args.log_dir
    cfg.experiment.experiment_name = f"{args.mode}_seed{args.seed}"
    cfg.experiment.write_interval = rollouts
    cfg.experiment.checkpoint_interval = args.timesteps // args.num_envs // 4

    agent = PrivilegedPPO(
        models=models,
        memory=RandomMemory(memory_size=rollouts, num_envs=env.num_envs, device=device),
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        cfg=cfg,
    )
    trainer = SequentialTrainer(cfg={"timesteps": args.timesteps // args.num_envs, "headless": True}, env=env, agents=agent)
    trainer.train()

    policy = models["policy"]
    masked = evaluate(policy, 1.0, args.eval_episodes, args.seed + 1000)
    visible = evaluate(policy, 0.0, args.eval_episodes, args.seed + 1000)
    print(f"[eval] mode={args.mode} seed={args.seed} return masked={masked:.1f} visible={visible:.1f}")


if __name__ == "__main__":
    main()
