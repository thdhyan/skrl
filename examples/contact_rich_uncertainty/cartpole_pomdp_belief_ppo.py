"""Quick smoke test: PPO trained on a GMM-compressed particle-filter belief.

There is no BeliefPPO agent in this branch yet (only `BeliefPPO_CFG` — see
`skrl/agents/torch/belief_ppo/belief_ppo_cfg.py` and `tasks.md` §7). This
script gets a belief-based on-policy benchmark running today by reusing the
already-tested belief plumbing (`ParticleFilter`, GMM `compress()`) as an
*observation wrapper* in front of skrl's standard, working `PPO` agent —
instead of feeding the policy raw (partially-masked) obs, it feeds the
compressed belief b̃.

Same POMDP setup as `cartpole_pomdp_belief_dqn.py` (cart velocity + pole
angular velocity hidden), so reward curves are directly comparable to the
BeliefDQN numbers already recorded in BENCHMARK_ACTION_SPACE_ANALYSIS.md.

Usage:
    # belief-based run
    python examples/contact_rich_uncertainty/cartpole_pomdp_belief_ppo.py --headless --timesteps 20000
    # same POMDP, same PPO hyperparameters, raw obs instead of belief -- comparison baseline
    python examples/contact_rich_uncertainty/cartpole_pomdp_belief_ppo.py --headless --timesteps 20000 --baseline
    # then compare learning curves:
    tensorboard --logdir runs/torch
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import torch
import torch.nn as nn
import gymnasium

from skrl.agents.torch.ppo import PPO, PPO_CFG
from skrl.belief.torch.particle_filter import ParticleFilter
from skrl.envs.wrappers.torch import wrap_env
from skrl.envs.wrappers.torch.pomdp_wrapper import PomdpWrapper
from skrl.memories.torch import RandomMemory
from skrl.models.torch import CategoricalMixin, DeterministicMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed


# -----------------------------------------------------------------------
# CartPole state: [cart_pos, cart_vel, pole_angle, pole_ang_vel]
# Hidden:         [cart_vel (1), pole_ang_vel (3)]  -- same POMDP as the DQN example
# -----------------------------------------------------------------------
STATE_DIM = 4
HIDDEN = [1, 3]
NG = 2          # GMM components
NP = 200        # particles
BELIEF_DIM = NG + 2 * NG * STATE_DIM  # 18

GRAVITY = 9.8
MASSCART = 1.0
MASSPOLE = 0.1
TOTAL_MASS = MASSCART + MASSPOLE
LENGTH = 0.5
POLEMASS_LENGTH = MASSPOLE * LENGTH
FORCE_MAG = 10.0
TAU = 0.02


def cartpole_step(state: torch.Tensor, action_force: float, noise_std: float = 0.01) -> torch.Tensor:
    """Euler-integrated CartPole dynamics for particle propagation (same model as the belief-DQN example)."""
    squeeze = state.dim() == 1
    if squeeze:
        state = state.unsqueeze(0)

    x, x_dot, theta, theta_dot = state[:, 0], state[:, 1], state[:, 2], state[:, 3]

    force = action_force
    costheta = torch.cos(theta)
    sintheta = torch.sin(theta)

    temp = (force + POLEMASS_LENGTH * theta_dot ** 2 * sintheta) / TOTAL_MASS
    theta_acc = (GRAVITY * sintheta - costheta * temp) / (
        LENGTH * (4.0 / 3.0 - MASSPOLE * costheta ** 2 / TOTAL_MASS)
    )
    x_acc = temp - POLEMASS_LENGTH * theta_acc * costheta / TOTAL_MASS

    next_state = torch.stack(
        [
            x + TAU * x_dot,
            x_dot + TAU * x_acc,
            theta + TAU * theta_dot,
            theta_dot + TAU * theta_acc,
        ],
        dim=-1,
    )
    next_state = next_state + torch.randn_like(next_state) * noise_std

    return next_state.squeeze(0) if squeeze else next_state


def make_motion_model(action_force_getter, noise_std: float = 0.05):
    def motion_model(particles: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        return cartpole_step(particles, action_force_getter(), noise_std=noise_std)
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


class BeliefObservationWrapper(gymnasium.Wrapper):
    """Feeds a GMM-compressed particle-filter belief b̃ to the policy instead of raw partial obs.

    Wraps a ``PomdpWrapper``-wrapped env. Each step: predict (motion model,
    driven by the action just taken) -> update (measurement model, using the
    masked obs) -> compress (GMM fit) -> the flattened belief is returned as
    the observation. Any standard skrl agent (PPO here) can then train on it
    unmodified.
    """

    def __init__(
        self,
        env: gymnasium.Env,
        *,
        state_dim: int = STATE_DIM,
        num_particles: int = NP,
        num_gmm_components: int = NG,
        device: str = "cpu",
    ) -> None:
        super().__init__(env)
        self._state_dim = state_dim
        self._device = device
        self._current_force = [0.0]

        motion_model = make_motion_model(lambda: self._current_force[0])
        measurement_model = make_measurement_model()

        self._belief = ParticleFilter(
            num_particles=num_particles,
            state_dim=state_dim,
            num_gmm_components=num_gmm_components,
            motion_model_fn=motion_model,
            measurement_model_fn=measurement_model,
            device=device,
            resample_threshold=0.5,
        )
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._belief.belief_dim,), dtype=np.float32
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        full_obs = self.env.get_full_obs()
        n = self._belief.num_particles

        prior = torch.zeros(n, self._state_dim, device=self._device)
        prior[:, 0] = float(full_obs[0]) + torch.randn(n) * 0.3
        prior[:, 1] = torch.randn(n) * 1.0    # unknown cart velocity
        prior[:, 2] = float(full_obs[2]) + torch.randn(n) * 0.05
        prior[:, 3] = torch.randn(n) * 0.5     # unknown pole angular velocity
        self._belief.reset(prior)
        self._belief.compress()

        return self._belief.get_compressed().cpu().numpy(), info

    def step(self, action):
        self._current_force[0] = FORCE_MAG if int(action) == 1 else -FORCE_MAG
        obs, reward, terminated, truncated, info = self.env.step(action)

        self._belief.predict(control=torch.tensor([self._current_force[0]], device=self._device))
        self._belief.update(measurement=torch.tensor(obs, dtype=torch.float32, device=self._device))
        self._belief.compress()
        info["belief_entropy"] = self._belief.get_entropy().item()

        return self._belief.get_compressed().cpu().numpy(), reward, terminated, truncated, info


# -----------------------------------------------------------------------
# skrl models: plain MLP actor-critic on the compressed belief b̃ (18-dim)
# -----------------------------------------------------------------------
class Policy(CategoricalMixin, Model):
    def __init__(self, observation_space, state_space, action_space, device, unnormalized_log_prob=True):
        Model.__init__(
            self, observation_space=observation_space, state_space=state_space, action_space=action_space, device=device
        )
        CategoricalMixin.__init__(self, unnormalized_log_prob=unnormalized_log_prob)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_actions),
        )

    def compute(self, inputs, role):
        return self.net(inputs["observations"]), {}


class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, state_space, action_space, device):
        Model.__init__(
            self, observation_space=observation_space, state_space=state_space, action_space=action_space, device=device
        )
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


def apply_cartpole_physics(base_env: gymnasium.Env, masscart: float, masspole: float) -> None:
    """Override cart/pole mass on a constructed CartPoleEnv (and its derived terms).

    Gymnasium's classic CartPole reads ``masscart``/``masspole``/``total_mass``/
    ``polemass_length`` directly in ``step()`` (see cartpole.py), so mutating
    these instance attributes after construction changes the true dynamics.
    Also updates this module's globals so the particle filter's internal
    motion model (``cartpole_step``) matches the true environment -- i.e. no
    model-mismatch confound when comparing algorithms across mass settings.
    """
    global MASSCART, MASSPOLE, TOTAL_MASS, POLEMASS_LENGTH
    MASSCART = masscart
    MASSPOLE = masspole
    TOTAL_MASS = MASSCART + MASSPOLE
    POLEMASS_LENGTH = MASSPOLE * LENGTH

    unwrapped = base_env.unwrapped
    unwrapped.masscart = MASSCART
    unwrapped.masspole = MASSPOLE
    unwrapped.total_mass = TOTAL_MASS
    unwrapped.polemass_length = POLEMASS_LENGTH


def run(
    timesteps: int,
    headless: bool,
    seed: int,
    device: str,
    baseline: bool,
    masscart: float = MASSCART,
    masspole: float = MASSPOLE,
    log_dir: str | None = None,
) -> None:
    set_seed(seed)

    base_env = gymnasium.make("CartPole-v1", render_mode=None if headless else "human")
    apply_cartpole_physics(base_env, masscart, masspole)
    env = PomdpWrapper(base_env, hidden_obs_indices=HIDDEN, obs_noise_std=0.02, seed=seed)
    if not baseline:
        # belief-based: policy/value see the GMM-compressed belief b̃ instead of raw obs
        env = BeliefObservationWrapper(env, device=device)
    # baseline: policy/value see the raw, partially-masked obs directly (same POMDP, no belief)
    env = wrap_env(env)

    device = env.device
    memory = RandomMemory(memory_size=256, num_envs=env.num_envs, device=device)

    models = {
        "policy": Policy(env.observation_space, env.state_space, env.action_space, device),
        "value": Value(env.observation_space, env.state_space, env.action_space, device),
    }

    cfg = PPO_CFG()
    cfg.rollouts = 256
    cfg.learning_epochs = 8
    cfg.mini_batches = 4
    cfg.discount_factor = 0.99
    cfg.gae_lambda = 0.95
    cfg.learning_rate = 1e-3
    cfg.grad_norm_clip = 0.5
    cfg.ratio_clip = 0.2
    cfg.value_clip = 0.2
    cfg.entropy_loss_scale = 0.01
    cfg.value_loss_scale = 0.5
    cfg.observation_preprocessor = RunningStandardScaler
    cfg.observation_preprocessor_kwargs = {"size": env.observation_space, "device": device}
    cfg.value_preprocessor = RunningStandardScaler
    cfg.value_preprocessor_kwargs = {"size": 1, "device": device}
    cfg.experiment.write_interval = "auto"
    cfg.experiment.checkpoint_interval = "auto"
    if log_dir is None:
        log_dir = "runs/torch/CartPolePOMDP_BaselinePPO" if baseline else "runs/torch/CartPolePOMDP_BeliefPPO"
    cfg.experiment.directory = log_dir

    agent = PPO(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        state_space=env.state_space,
        action_space=env.action_space,
        device=device,
    )

    print(f"TensorBoard: tensorboard --logdir {cfg.experiment.directory}")
    trainer = SequentialTrainer(cfg={"timesteps": timesteps, "headless": headless}, env=env, agents=agent)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Skip the belief wrapper: train PPO directly on the raw partially-masked obs (POMDP, no belief). "
        "Use this to get a same-hyperparameter comparison curve against the belief-based run.",
    )
    parser.add_argument("--masscart", type=float, default=MASSCART, help="Override cart mass (default 1.0 kg).")
    parser.add_argument("--masspole", type=float, default=MASSPOLE, help="Override pole mass (default 0.1 kg).")
    parser.add_argument("--log-dir", default=None, help="Override TensorBoard/checkpoint directory.")
    args = parser.parse_args()

    run(
        timesteps=args.timesteps,
        headless=args.headless,
        seed=args.seed,
        device=args.device,
        baseline=args.baseline,
        masscart=args.masscart,
        masspole=args.masspole,
        log_dir=args.log_dir,
    )
