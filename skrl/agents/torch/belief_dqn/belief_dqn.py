from __future__ import annotations

from typing import Any

import gymnasium
from packaging import version

import torch
import torch.nn.functional as F

from skrl import config, logger
from skrl.agents.torch import Agent
from skrl.agents.torch.belief_mixin import BeliefAgentMixin
from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl.utils import ScopedTimer

from .belief_dqn_cfg import BeliefDQN_CFG


class BeliefDQN(BeliefAgentMixin, Agent):
    """Belief-conditioned DQN for contact-rich manipulation under uncertainty.

    Implements the high-level policy from Wirnshofer et al. RSS 2020.

    Key differences from standard DQN:
    - Agent owns a Belief (ParticleFilter) instance.
    - Observation fed to Q-network is the compressed belief b̃, not raw env obs.
    - On each env step: predict(action) → env.step() → update(obs) → compress().
    - Replay buffer stores (b̃_t, a, r, b̃_{t+1}, done) — compressed beliefs.
    - Double-Q + dueling head + soft target update (all per paper §V-A).
    - Tracks belief entropy and compression time in TensorBoard.

    The agent's act() method takes the raw joint-encoder observation from the
    env but returns the DQN action (controller index).  The belief update
    is driven by those raw observations.

    :param models: Dict with keys ``"q_network"`` and ``"target_q_network"``,
        both instances of ResNetDuelingQNetwork (or any DeterministicMixin Model
        returning Q-values of shape ``[batch, NC]``).
    :param memory: Replay buffer.  Stores compressed beliefs.
    :param observation_space: Raw env observation space (joint encoders).
        Only used for the memory tensor if ``belief_dim`` not in cfg.
    :param state_space: Optional privileged state space (unused by default).
    :param action_space: Discrete(NC).
    :param device: Torch device.
    :param cfg: BeliefDQN_CFG instance or dict.
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
        cfg: BeliefDQN_CFG | dict = {},
    ) -> None:
        self.cfg: BeliefDQN_CFG
        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
            cfg=BeliefDQN_CFG(**cfg) if isinstance(cfg, dict) else cfg,
        )

        if self.cfg.belief_factory is None:
            raise ValueError("BeliefDQN_CFG.belief_factory must be set to a callable() -> Belief.")

        # instantiate the belief tracker
        self._belief_setup(self.cfg.belief_factory)

        # validate action space
        if not hasattr(self.action_space, "n"):
            raise ValueError("BeliefDQN requires a Discrete action space.")
        self._nc = int(self.action_space.n)

        # models
        self.q_network = self.models.get("q_network", None)
        self.target_q_network = self.models.get("target_q_network", None)

        self.checkpoint_modules["q_network"] = self.q_network
        self.checkpoint_modules["target_q_network"] = self.target_q_network

        if config.torch.is_distributed:
            logger.info("Broadcasting models' parameters")
            if self.q_network is not None:
                self.q_network.broadcast_parameters()

        # AMP
        self._device_type = torch.device(self.device).type
        if version.parse(torch.__version__) >= version.parse("2.4"):
            self.scaler = torch.amp.GradScaler(device=self._device_type, enabled=self.cfg.mixed_precision)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.mixed_precision)

        # optimizer
        if self.q_network is not None:
            self.optimizer = torch.optim.Adam(
                self.q_network.parameters(),
                lr=self.cfg.learning_rate,
                weight_decay=self.cfg.l2_reg,
            )
            self.checkpoint_modules["optimizer"] = self.optimizer
            if self.cfg.learning_rate_scheduler is not None:
                self.scheduler = self.cfg.learning_rate_scheduler(
                    self.optimizer, **self.cfg.learning_rate_scheduler_kwargs
                )

        # freeze and hard-copy target
        if self.target_q_network is not None:
            self.target_q_network.freeze_parameters(True)
            self.target_q_network.update_parameters(self.q_network, polyak=1)

        # belief preprocessor (optional, same pattern as DQN)
        if self.cfg.observation_preprocessor:
            self._obs_preprocessor = self.cfg.observation_preprocessor(
                **self.cfg.observation_preprocessor_kwargs
            )
            self.checkpoint_modules["obs_preprocessor"] = self._obs_preprocessor
        else:
            self._obs_preprocessor = self._empty_preprocessor

        self._update_counter = 0

    # ------------------------------------------------------------------
    # Agent interface
    # ------------------------------------------------------------------

    def init(self, *, trainer_cfg: dict[str, Any] | None = None) -> None:
        super().init(trainer_cfg=trainer_cfg)
        self.enable_models_training_mode(False)

        if self.memory is not None:
            belief_dim = self._belief.belief_dim
            self.memory.create_tensor(name="beliefs", size=belief_dim, dtype=torch.float32)
            self.memory.create_tensor(name="next_beliefs", size=belief_dim, dtype=torch.float32)
            self.memory.create_tensor(name="actions", size=self.action_space, dtype=torch.int64)
            self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="terminated", size=1, dtype=torch.bool)

        self._tensors_names = ["beliefs", "actions", "rewards", "next_beliefs", "terminated"]

    def act(
        self,
        observations: torch.Tensor,
        states: torch.Tensor | None,
        *,
        timestep: int,
        timesteps: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Select a controller index given the current belief.

        The compressed belief b̃ is already updated by record_transition()
        from the previous step.  On the very first step it is computed here.

        :param observations: Raw joint-encoder reading, shape ``[num_envs, obs_dim]``.
        :param states: Unused.
        :param timestep: Current global timestep.
        :param timesteps: Total training timesteps.
        :return: Action (controller index) ``[num_envs, 1]`` and outputs dict.
        """
        # compress current particles → b̃
        b_tilde = self._belief_observe()
        b_tilde = self._obs_preprocessor(b_tilde)

        # pure random during warm-up
        if timestep < self.cfg.random_timesteps:
            action = torch.randint(0, self._nc, (b_tilde.shape[0], 1), device=self.device)
            return action, self._with_belief_markers({})

        if self.cfg.exploration_scheduler is None:
            with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                q_values, outputs = self.q_network.act({"observations": b_tilde}, role="q_network")
            action = torch.argmax(q_values, dim=1, keepdim=True)
            return action, self._with_belief_markers(outputs)

        # epsilon-greedy
        epsilon = self.cfg.exploration_scheduler(timestep, timesteps)
        self.track_data("Exploration / epsilon", epsilon)

        with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
            q_values, outputs = self.q_network.act({"observations": b_tilde}, role="q_network")
        action = torch.argmax(q_values, dim=1, keepdim=True)

        rand_mask = torch.rand(action.shape[0], device=self.device) < epsilon
        if rand_mask.any():
            action[rand_mask] = torch.randint(0, self._nc, (rand_mask.sum().item(), 1), device=self.device)

        return action, self._with_belief_markers(outputs)

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
        """Store (b̃_t, a, r, b̃_{t+1}, done) in the replay buffer.

        Drives the belief forward with the chosen action, then updates
        it with the next observation to produce b̃_{t+1}.
        """
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

        if self.memory is None:
            return

        # b̃_t already computed in act()
        b_t = self._b_tilde if self._b_tilde is not None else self._belief.get_compressed().unsqueeze(0)

        # advance belief with the action's controller reference
        # (the env already stepped; we propagate the filter to match)
        b_t1 = self._belief_advance(actions, next_observations, infos)

        # shape rewards
        if self.cfg.rewards_shaper is not None:
            rewards = self.cfg.rewards_shaper(rewards, timestep, timesteps)

        self.memory.add_samples(
            beliefs=b_t,
            next_beliefs=b_t1,
            actions=actions,
            rewards=rewards,
            terminated=terminated,
        )

    def pre_interaction(self, *, timestep: int, timesteps: int) -> None:
        pass

    def post_interaction(self, *, timestep: int, timesteps: int) -> None:
        if timestep >= self.cfg.learning_starts and timestep % self.cfg.update_interval == 0:
            self.update(timestep=timestep, timesteps=timesteps)

    # ------------------------------------------------------------------
    # Training update (Double-Q + Dueling)
    # ------------------------------------------------------------------

    def update(self, *, timestep: int, timesteps: int) -> None:
        if self.memory is None or len(self.memory) < self.cfg.batch_size:
            return

        for _ in range(self.cfg.gradient_steps):
            self._update_counter += 1

            # sample mini-batch — returns list[list[tensor]], take first mini-batch
            sampled = self.memory.sample(
                names=self._tensors_names, batch_size=self.cfg.batch_size
            )[0]
            b_t, actions, rewards, b_t1, terminated = sampled

            b_t = self._obs_preprocessor(b_t)
            b_t1 = self._obs_preprocessor(b_t1)

            with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                # current Q-values
                q_values, _ = self.q_network.act({"observations": b_t}, role="q_network")
                q_a = q_values.gather(1, actions.long())

                with torch.no_grad():
                    if self.cfg.double_q:
                        # double-Q: online net selects action, target net evaluates
                        next_q_online, _ = self.q_network.act({"observations": b_t1}, role="q_network")
                        best_a = next_q_online.argmax(1, keepdim=True)
                        next_q_target, _ = self.target_q_network.act({"observations": b_t1}, role="target_q_network")
                        max_q = next_q_target.gather(1, best_a)
                    else:
                        next_q_target, _ = self.target_q_network.act({"observations": b_t1}, role="target_q_network")
                        max_q = next_q_target.max(1, keepdim=True).values

                    target = rewards + self.cfg.discount_factor * max_q * (~terminated)

                loss = F.mse_loss(q_a, target)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # soft target update
            if self._update_counter % self.cfg.target_update_interval == 0:
                self.target_q_network.update_parameters(self.q_network, polyak=self.cfg.polyak)

            self.track_data("Loss / q_loss", loss.item())
            self.track_data("Q-values / mean", q_values.mean().item())
            self.track_data("Q-values / max", q_values.max().item())

        if hasattr(self, "scheduler") and self.scheduler is not None:
            self.scheduler.step()
