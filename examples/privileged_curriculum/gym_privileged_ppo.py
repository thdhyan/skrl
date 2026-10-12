"""PrivilegedPPO on built-in gymnasium tasks with hidden velocities and randomized dynamics.

The policy sees the task observation with the velocities zeroed (plus small noise) and a history of it.
Privileged values: the hidden velocities and the randomized dynamics (per episode):

* ``Pendulum-v1``: angular velocity; pendulum mass ``m`` and length ``l``.
* ``InvertedPendulum-v5``, ``HalfCheetah-v5``, ``Hopper-v5``: joint velocities; global scales of all body
  masses and of all sliding frictions (MuJoCo ``model.body_mass`` and ``model.geom_friction[:, 0]``).

Modes:

* ``curriculum``: time-based softmax mask schedule (``SoftmaxMaskCurriculum``).
* ``gated``: reward-gated schedule (``RewardGatedMaskCurriculum``): the mask probability grows by one step
  only once the training return is constant at its best level, and backs off when it drops.
* ``masked`` / ``visible``: privileged values always masked / always visible.

At the end the policy is evaluated with the privileged values masked (deployable) and visible, and the
held-out R² of a probe that predicts each privileged group from the estimator latent ẑ (and the encoder
latent z) is printed. A JSON summary is written to ``<log-dir>/<task>_<mode>_seed<seed>.json``.

    python examples/privileged_curriculum/gym_privileged_ppo.py --task Pendulum-v1 --mode gated
    python examples/privileged_curriculum/gym_privileged_ppo.py --task HalfCheetah-v5 --mode curriculum --timesteps 1000000
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os

import numpy as np
import torch
import gymnasium

from skrl.agents.torch.privileged_ppo import (
    ESTIMATORS,
    ObsLayout,
    PrivilegedGaussianPolicy,
    PrivilegedPPO,
    PrivilegedPPO_CFG,
    PrivilegedValue,
    RewardGatedMaskCurriculum,
    SoftmaxMaskCurriculum,
)
from skrl.envs.wrappers.torch import wrap_env
from skrl.envs.wrappers.torch.pomdp_wrapper import PomdpWrapper
from skrl.envs.wrappers.torch.privileged_history_wrapper import PrivilegedHistoryWrapper
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed


@dataclasses.dataclass
class Task:
    velocity_indices: list[int]  # observation indices hidden from the policy
    dynamics: list[str]  # names of the randomized dynamics parameters
    ranges: list[tuple[float, float]]
    timesteps: int  # default budget (env steps)
    history: int = 8


TASKS = {
    "Pendulum-v1": Task([2], ["mass", "length"], [(0.5, 1.5), (0.6, 1.4)], 300_000),
    "InvertedPendulum-v5": Task([2, 3], ["mass", "friction"], [(0.5, 2.0), (0.5, 1.5)], 150_000),
    "HalfCheetah-v5": Task(list(range(8, 17)), ["mass", "friction"], [(0.6, 1.4), (0.4, 1.6)], 1_000_000),
    "Hopper-v5": Task(list(range(5, 11)), ["mass", "friction"], [(0.7, 1.3), (0.5, 1.5)], 1_000_000),
}


class Switchable:
    """Wraps a curriculum so that evaluation can force the mask probability (``override``)."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.override: float | None = None

    def mask_probability(self) -> float:
        return self.inner.mask_probability() if self.override is None else self.override

    def __getattr__(self, name):
        return getattr(self.inner, name)


class Fixed:
    def __init__(self, p: float) -> None:
        self.p = p

    def mask_probability(self) -> float:
        return self.p

    def set_progress(self, progress: float) -> None:
        pass


def make_randomize(name: str, task: Task):
    def randomize(env: gymnasium.Env, rng: np.random.Generator) -> None:
        raw = env.unwrapped
        scales = [rng.uniform(*r) for r in task.ranges]
        raw._privileged_dynamics = np.array(scales, dtype=np.float32)
        if name == "Pendulum-v1":
            raw.m, raw.l = scales
            return
        model = raw.model
        if not hasattr(raw, "_nominal_dynamics"):
            raw._nominal_dynamics = (model.body_mass.copy(), model.geom_friction[:, 0].copy())
        mass, friction = raw._nominal_dynamics
        model.body_mass[:] = mass * scales[0]
        model.geom_friction[:, 0] = friction * scales[1]

    return randomize


def make_privileged(task: Task):
    def privileged(env: gymnasium.Env) -> np.ndarray:
        full = env.get_full_obs()  # normalized observation before the velocities are hidden
        return np.concatenate([full[task.velocity_indices], env.unwrapped._privileged_dynamics]).astype(np.float32)

    return privileged


def make_env(name: str, task: Task, curriculum, seed: int, actions: bool = False) -> gymnasium.Env:
    env = gymnasium.make(name)
    env = gymnasium.wrappers.NormalizeObservation(env)
    env = PomdpWrapper(env, hidden_obs_indices=task.velocity_indices, obs_noise_std=0.01, seed=seed)
    return PrivilegedHistoryWrapper(
        env,
        privileged_fn=make_privileged(task),
        privileged_dim=len(task.velocity_indices) + len(task.dynamics),
        history_length=task.history,
        curriculum=curriculum,
        randomize_fn=make_randomize(name, task),
        seed=seed,
        include_last_action=actions,
    )


@torch.no_grad()
def evaluate(env, vec_env, policy, curriculum: Switchable, mask: float, episodes: int) -> float:
    """Deterministic returns on the training envs (shared observation statistics, frozen during evaluation)."""
    curriculum.override = mask
    for sub in vec_env.envs:
        _normalizer(sub).update_running_mean = False
    env._reset_once = True  # the skrl vector-env wrapper resets only once: force fresh episodes with this mask
    obs, _ = env.reset()
    totals = torch.zeros(env.num_envs, device=env.device)
    returns = []
    low = torch.as_tensor(env.action_space.low, device=env.device)
    high = torch.as_tensor(env.action_space.high, device=env.device)
    while len(returns) < episodes:
        actions = policy.compute({"observations": obs})[0].clamp(low, high)
        obs, rewards, terminated, truncated, _ = env.step(actions)
        totals += rewards.view(-1)
        done = (terminated | truncated).view(-1)
        returns += totals[done].tolist()
        totals[done] = 0
    for sub in vec_env.envs:
        _normalizer(sub).update_running_mean = True
    curriculum.override = None
    return float(np.mean(returns[:episodes]))


def _normalizer(env: gymnasium.Env) -> gymnasium.wrappers.NormalizeObservation:
    while not isinstance(env, gymnasium.wrappers.NormalizeObservation):
        env = env.env
    return env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=list(TASKS), default="Pendulum-v1")
    parser.add_argument("--mode", choices=["curriculum", "gated", "masked", "visible"], default="gated")
    parser.add_argument("--timesteps", type=int, default=None, help="total env steps (default: per task)")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-episodes", type=int, default=16)
    parser.add_argument("--log-dir", default="runs/privileged_gym")
    parser.add_argument("--history", type=int, default=None, help="history length H (default: per task)")
    parser.add_argument("--actions", action="store_true", help="observation-action history (a_{t-1} appended to o_t)")
    parser.add_argument("--encoder-loss-scale", type=float, default=0.1, help="λ of ||z - sg(ẑ)||² (0: z free)")
    parser.add_argument("--estimator", choices=list(ESTIMATORS), default="mlp", help="history estimator architecture")
    parser.add_argument("--sysid-loss-scale", type=float, default=0.0, help="supervised system-ID loss (0: off)")
    args = parser.parse_args()
    set_seed(args.seed)
    task = TASKS[args.task]
    if args.history is not None:
        task = dataclasses.replace(task, history=args.history)
    timesteps = args.timesteps or task.timesteps

    inner = {
        "curriculum": SoftmaxMaskCurriculum(start=0.1, end=0.7, sharpness=10.0),
        "gated": RewardGatedMaskCurriculum(step=0.1, tolerance=0.1, patience=3, return_scale=10.0, deadline=0.8),
        "masked": Fixed(1.0),
        "visible": Fixed(0.0),
    }[args.mode]
    curriculum = Switchable(inner)

    vec_env = gymnasium.vector.SyncVectorEnv(
        [lambda i=i: make_env(args.task, task, curriculum, args.seed + i, args.actions) for i in range(args.num_envs)],
        autoreset_mode=gymnasium.vector.AutoresetMode.SAME_STEP,
    )
    env = wrap_env(vec_env)
    device = env.device

    n_vel = len(task.velocity_indices)
    obs_dim = vec_env.single_observation_space.shape[0] - n_vel - len(task.dynamics) - 1
    layout = ObsLayout(obs_dim=obs_dim // (1 + task.history), history_length=task.history, privileged_dim=n_vel + len(task.dynamics))
    models = {
        "policy": PrivilegedGaussianPolicy(
            env.observation_space, env.action_space, device, layout=layout, hidden=128,
            estimator=args.estimator, sysid_head=args.sysid_loss_scale > 0,
        ),
        "value": PrivilegedValue(env.observation_space, env.action_space, device, layout=layout, hidden=128),
    }
    groups = {"velocities": list(range(n_vel))}
    groups.update({name: [n_vel + i] for i, name in enumerate(task.dynamics)})
    rollouts = 256
    cfg = PrivilegedPPO_CFG(
        curriculum=curriculum,
        privileged_groups=groups,
        rollouts=rollouts,
        learning_epochs=10,
        mini_batches=8,
        discount_factor=0.99,
        gae_lambda=0.95,
        learning_rate=3e-4,
        grad_norm_clip=0.5,
        ratio_clip=0.2,
        entropy_loss_scale=0.0,
        value_loss_scale=0.5,
        value_preprocessor=RunningStandardScaler,
        value_preprocessor_kwargs={"size": 1, "device": device},
        encoder_loss_scale=args.encoder_loss_scale,
        sysid_loss_scale=args.sysid_loss_scale,
    )
    variant = (f"_H{task.history}" if args.history is not None else "") + ("_act" if args.actions else "")
    variant += f"_lam{args.encoder_loss_scale:g}" if args.encoder_loss_scale != 0.1 else ""
    variant += f"_{args.estimator}" if args.estimator != "mlp" else ""
    variant += f"_sysid{args.sysid_loss_scale:g}" if args.sysid_loss_scale > 0 else ""
    run_name = f"{args.task}_{args.mode}{variant}_seed{args.seed}"
    cfg.experiment.directory = args.log_dir
    cfg.experiment.experiment_name = run_name
    cfg.experiment.write_interval = rollouts
    cfg.experiment.checkpoint_interval = timesteps // args.num_envs // 4

    agent = PrivilegedPPO(
        models=models,
        memory=RandomMemory(memory_size=rollouts, num_envs=env.num_envs, device=device),
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        cfg=cfg,
    )
    trainer_cfg = {"timesteps": timesteps // args.num_envs, "headless": True, "disable_progressbar": True}
    SequentialTrainer(cfg=trainer_cfg, env=env, agents=agent).train()

    policy = models["policy"]
    masked = evaluate(env, vec_env, policy, curriculum, 1.0, args.eval_episodes)
    visible = evaluate(env, vec_env, policy, curriculum, 0.0, args.eval_episodes)
    probes = {k: round(v, 3) for k, v in agent.probe_metrics.items()}
    # gated: the probability reached by the gate itself (without the deadline floor)
    final_p = inner.gated if args.mode == "gated" else inner.mask_probability()
    summary = {
        "task": args.task,
        "mode": args.mode,
        "seed": args.seed,
        "timesteps": timesteps,
        "history": task.history,
        "history_actions": args.actions,
        "encoder_loss_scale": args.encoder_loss_scale,
        "estimator": args.estimator,
        "sysid_loss_scale": args.sysid_loss_scale,
        "return_masked": masked,
        "return_visible": visible,
        "final_mask_probability": final_p,
        "probe_r2": probes,
    }
    if args.mode == "gated":
        summary["gate_steps_up"], summary["gate_steps_down"] = inner.steps_up, inner.steps_down
    os.makedirs(args.log_dir, exist_ok=True)
    with open(os.path.join(args.log_dir, f"{run_name}.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"[eval] {args.task} mode={args.mode} seed={args.seed} masked={masked:.1f} visible={visible:.1f}")
    print(f"[probe] held-out R2 per privileged group: {probes}")
    env.close()


if __name__ == "__main__":
    main()
