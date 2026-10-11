from __future__ import annotations

from typing import Any

import itertools
import gymnasium
from packaging import version

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config, logger
from skrl.agents.torch import Agent
from skrl.agents.torch.belief_mixin import BeliefAgentMixin
from skrl.agents.torch.ppo.ppo import compute_gae
from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.utils import ScopedTimer

from .belief_ppo_cfg import BeliefPPO_CFG


class BeliefPPO(BeliefAgentMixin, Agent):
    """Belief-conditioned PPO (on-policy counterpart to BeliefDQN).

    The agent owns a Belief (ParticleFilter).  Policy and value networks never
    see the raw env observation: they act on the compressed belief b̃.

    Per env step:
    - ``act()``: (re)initialise the filter from ``belief_prior_fn(obs)`` at
      episode start, compress → b̃_t, sample a ~ π(·|b̃_t), compute V(b̃_t).
    - ``record_transition()``: predict(a) → update(next_obs) → compress →
      b̃_{t+1}; store (b̃_t, a, log π, V, r, done) in the rollout memory.
    - every ``rollouts`` steps: GAE on stored values, bootstrapped with
      V(b̃_{t+1}); clipped-surrogate PPO epochs over mini-batches of b̃.

    A single filter is tracked, so the agent supports ``num_envs == 1``.

    :param models: Dict with keys ``"policy"`` and ``"value"``, both taking
        ``inputs["observations"]`` of shape ``[batch, belief_dim]``.
    :param memory: Rollout memory (``memory_size == cfg.rollouts``).  Stores compressed beliefs.
    :param observation_space: Raw env observation space (only passed to the filter).
    :param state_space: Unused.
    :param action_space: Action space (Discrete(NC) per paper, Box also works).
    :param device: Torch device.
    :param cfg: BeliefPPO_CFG instance or dict.
    """

    def __init__(
        self,
        *,
        models: dict[str, Model],
        memory: Memory | None = None,
        observation_space: gymnasium.Space | None = None,
        state_space: gymnasium.Space | None = None,
        action_space: gymnasium.Space | None = None,
        device: str | torch.device | None = None,
        cfg: BeliefPPO_CFG | dict = {},
    ) -> None:
        self.cfg: BeliefPPO_CFG
        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
            cfg=BeliefPPO_CFG(**cfg) if isinstance(cfg, dict) else cfg,
        )

        if self.cfg.belief_factory is None:
            raise ValueError("BeliefPPO_CFG.belief_factory must be set to a callable() -> Belief.")

        # instantiate the belief tracker
        self._belief_setup(self.cfg.belief_factory)

        # models
        self.policy = self.models.get("policy", None)
        self.value = self.models.get("value", None)

        self.checkpoint_modules["policy"] = self.policy
        self.checkpoint_modules["value"] = self.value

        if config.torch.is_distributed:
            logger.info("Broadcasting models' parameters")
            if self.policy is not None:
                self.policy.broadcast_parameters()
                if self.value is not None and self.policy is not self.value:
                    self.value.broadcast_parameters()

        # AMP
        self._device_type = torch.device(self.device).type
        if version.parse(torch.__version__) >= version.parse("2.4"):
            self.scaler = torch.amp.GradScaler(device=self._device_type, enabled=self.cfg.mixed_precision)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.mixed_precision)

        # optimizer + scheduler
        self.scheduler = None
        if self.policy is not None and self.value is not None:
            if self.policy is self.value:
                params = self.policy.parameters()
            else:
                params = itertools.chain(self.policy.parameters(), self.value.parameters())
            self.optimizer = torch.optim.Adam(params, lr=self.cfg.learning_rate)
            self.checkpoint_modules["optimizer"] = self.optimizer
            if self.cfg.learning_rate_scheduler is not None:
                self.scheduler = self.cfg.learning_rate_scheduler(
                    self.optimizer, **self.cfg.learning_rate_scheduler_kwargs
                )

        # preprocessors (applied to b̃ and to values)
        if self.cfg.observation_preprocessor:
            self._obs_preprocessor = self.cfg.observation_preprocessor(**self.cfg.observation_preprocessor_kwargs)
            self.checkpoint_modules["obs_preprocessor"] = self._obs_preprocessor
        else:
            self._obs_preprocessor = self._empty_preprocessor

        if self.cfg.value_preprocessor:
            self._value_preprocessor = self.cfg.value_preprocessor(**self.cfg.value_preprocessor_kwargs)
            self.checkpoint_modules["value_preprocessor"] = self._value_preprocessor
        else:
            self._value_preprocessor = self._empty_preprocessor

    # ------------------------------------------------------------------
    # Agent interface
    # ------------------------------------------------------------------

    def init(self, *, trainer_cfg: dict[str, Any] | None = None) -> None:
        super().init(trainer_cfg=trainer_cfg)
        self.enable_models_training_mode(False)

        if self.memory is not None:
            belief_dim = self._belief.belief_dim
            self.memory.create_tensor(name="beliefs", size=belief_dim, dtype=torch.float32)
            self.memory.create_tensor(name="actions", size=self.action_space, dtype=torch.float32)
            self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="terminated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="truncated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="log_prob", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="values", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="returns", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="advantages", size=1, dtype=torch.float32)

        self._tensors_names = ["beliefs", "actions", "log_prob", "values", "returns", "advantages"]

        self._current_log_prob = None
        self._current_values = None
        self._needs_belief_reset = True
        self._rollout = 0

    def act(
        self,
        observations: torch.Tensor,
        states: torch.Tensor | None,
        *,
        timestep: int,
        timesteps: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Sample an action from π(·|b̃_t).

        :param observations: Raw env observation, shape ``[1, obs_dim]``.  Only used to
            seed the filter prior at episode start (via ``cfg.belief_prior_fn``).
        :return: Action ``[1, action_dim]`` and the policy outputs dict.
        """
        if observations is not None and observations.shape[0] != 1:
            raise ValueError(f"BeliefPPO tracks a single filter: expected num_envs == 1, got {observations.shape[0]}")

        if self._needs_belief_reset and self.cfg.belief_prior_fn is not None:
            self.reset_belief(self.cfg.belief_prior_fn(observations.squeeze(0)))
        self._needs_belief_reset = False

        # b̃_t is already fresh from the previous record_transition(); compress only after a reset
        b_tilde = self._b_tilde if self._b_tilde is not None else self._belief_observe()
        inputs = {"observations": self._obs_preprocessor(b_tilde)}

        if timestep < self.cfg.random_timesteps:
            return self.policy.random_act(inputs, role="policy")

        with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
            actions, outputs = self.policy.act(inputs, role="policy")
            self._current_log_prob = outputs["log_prob"]

            if self.training:
                values, _ = self.value.act(inputs, role="value")
                self._current_values = self._value_preprocessor(values, inverse=True)

        return actions, outputs

    def record_transition(
        self,
        *,
        observations: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        next_states: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: Any,
        timestep: int,
        timesteps: int,
    ) -> None:
        """Advance the filter to b̃_{t+1} and store (b̃_t, a, log π, V, r, done)."""
        super().record_transition(
            observations=observations,
            states=states,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            next_states=next_states,
            terminated=terminated,
            truncated=truncated,
            infos=infos,
            timestep=timestep,
            timesteps=timesteps,
        )

        # b̃_t was set by act(); advance the filter → b̃_{t+1} (kept for GAE bootstrap)
        b_t = self._b_tilde
        self._belief_advance(actions, next_observations)

        # the env auto-resets after a done step: re-seed the filter on the next act()
        if (terminated | truncated).any():
            self._needs_belief_reset = True

        if self.training and self.memory is not None:
            if self.cfg.rewards_shaper is not None:
                rewards = self.cfg.rewards_shaper(rewards, timestep, timesteps)

            self.memory.add_samples(
                beliefs=b_t,
                actions=actions,
                rewards=rewards,
                terminated=terminated,
                truncated=truncated,
                log_prob=self._current_log_prob,
                values=self._current_values,
            )

    def pre_interaction(self, *, timestep: int, timesteps: int) -> None:
        pass

    def post_interaction(self, *, timestep: int, timesteps: int) -> None:
        if self.training:
            self._rollout += 1
            if not self._rollout % self.cfg.rollouts and timestep >= self.cfg.learning_starts:
                with ScopedTimer() as timer:
                    self.enable_models_training_mode(True)
                    self.update(timestep=timestep, timesteps=timesteps)
                    self.enable_models_training_mode(False)
                    self.track_data("Stats / Algorithm update time (ms)", timer.elapsed_time_ms)

        super().post_interaction(timestep=timestep, timesteps=timesteps)

    # ------------------------------------------------------------------
    # Training update (GAE + clipped surrogate)
    # ------------------------------------------------------------------

    def update(self, *, timestep: int, timesteps: int) -> None:
        # bootstrap with V(b̃_{t+1}) of the last recorded transition
        with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
            self.value.enable_training_mode(False)
            last_values, _ = self.value.act({"observations": self._obs_preprocessor(self._b_tilde)}, role="value")
            self.value.enable_training_mode(True)
            last_values = self._value_preprocessor(last_values, inverse=True)

        values = self.memory.get_tensor_by_name("values")
        returns, advantages = compute_gae(
            rewards=self.memory.get_tensor_by_name("rewards"),
            terminated=self.memory.get_tensor_by_name("terminated"),
            truncated=self.memory.get_tensor_by_name("truncated"),
            values=values,
            last_values=last_values,
            discount_factor=self.cfg.discount_factor,
            lambda_coefficient=self.cfg.gae_lambda,
        )

        self.memory.set_tensor_by_name("values", self._value_preprocessor(values, train=True))
        self.memory.set_tensor_by_name("returns", self._value_preprocessor(returns, train=True))
        self.memory.set_tensor_by_name("advantages", advantages)

        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0

        for epoch in range(self.cfg.learning_epochs):
            kl_divergences = []

            for (
                sampled_beliefs,
                sampled_actions,
                sampled_log_prob,
                sampled_values,
                sampled_returns,
                sampled_advantages,
            ) in self.memory.sample(
                names=self._tensors_names, batch_size=len(self.memory), mini_batches=self.cfg.mini_batches
            ):

                with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                    inputs = {"observations": self._obs_preprocessor(sampled_beliefs, train=not epoch)}

                    _, outputs = self.policy.act({**inputs, "taken_actions": sampled_actions}, role="policy")
                    next_log_prob = outputs["log_prob"]

                    with torch.no_grad():
                        ratio = next_log_prob - sampled_log_prob
                        kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
                        kl_divergences.append(kl_divergence)

                    if self.cfg.kl_threshold and kl_divergence > self.cfg.kl_threshold:
                        break

                    if self.cfg.entropy_loss_scale:
                        entropy_loss = -self.cfg.entropy_loss_scale * self.policy.get_entropy(role="policy").mean()
                    else:
                        entropy_loss = 0

                    ratio = torch.exp(next_log_prob - sampled_log_prob)
                    surrogate = sampled_advantages * ratio
                    surrogate_clipped = sampled_advantages * torch.clip(
                        ratio, 1.0 - self.cfg.ratio_clip, 1.0 + self.cfg.ratio_clip
                    )
                    policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                    predicted_values, _ = self.value.act(inputs, role="value")
                    if self.cfg.value_clip > 0:
                        predicted_values = sampled_values + torch.clip(
                            predicted_values - sampled_values, min=-self.cfg.value_clip, max=self.cfg.value_clip
                        )
                    value_loss = self.cfg.value_loss_scale * F.mse_loss(sampled_returns, predicted_values)

                self.optimizer.zero_grad()
                self.scaler.scale(policy_loss + entropy_loss + value_loss).backward()

                if config.torch.is_distributed:
                    self.policy.reduce_parameters()
                    if self.policy is not self.value:
                        self.value.reduce_parameters()

                if self.cfg.grad_norm_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    if self.policy is self.value:
                        nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.grad_norm_clip)
                    else:
                        nn.utils.clip_grad_norm_(
                            itertools.chain(self.policy.parameters(), self.value.parameters()), self.cfg.grad_norm_clip
                        )

                self.scaler.step(self.optimizer)
                self.scaler.update()

                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss += value_loss.item()
                if self.cfg.entropy_loss_scale:
                    cumulative_entropy_loss += entropy_loss.item()

            if self.scheduler:
                if isinstance(self.scheduler, KLAdaptiveLR):
                    kl = torch.tensor(kl_divergences, device=self.device).mean()
                    if config.torch.is_distributed:
                        torch.distributed.all_reduce(kl, op=torch.distributed.ReduceOp.SUM)
                        kl /= config.torch.world_size
                    self.scheduler.step(kl.item())
                else:
                    self.scheduler.step()

        n = self.cfg.learning_epochs * self.cfg.mini_batches
        self.track_data("Loss / Policy loss", cumulative_policy_loss / n)
        self.track_data("Loss / Value loss", cumulative_value_loss / n)
        if self.cfg.entropy_loss_scale:
            self.track_data("Loss / Entropy loss", cumulative_entropy_loss / n)
        if self.scheduler:
            self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])
