from __future__ import annotations

from typing import Any

import dataclasses

import gymnasium
import torch
import torch.nn.functional as F

from skrl.agents.torch.ppo import PPO, PPO_CFG
from skrl.memories.torch import Memory
from skrl.models.torch import Model


@dataclasses.dataclass(kw_only=True)
class PrivilegedPPO_CFG(PPO_CFG):
    """Configuration for :class:`PrivilegedPPO` (PPO fields plus the privileged-curriculum ones)."""

    curriculum: Any = None
    """Shared mask curriculum (e.g. :class:`SoftmaxMaskCurriculum`). The agent sets its progress to
    ``timestep / timesteps``; the env wrapper reads the mask probability at every reset."""

    estimator_loss_scale: float = 1.0
    """Weight of ``||sg(z) - ẑ||²``: pulls the history estimator towards the privileged encoder."""

    encoder_loss_scale: float = 0.1
    """Weight of ``||z - sg(ẑ)||²``: keeps the encoder latent reachable by the estimator (ROA-style)."""


class PrivilegedPPO(PPO):
    """PPO with a privileged-observation curriculum and an encoder/estimator regularizer.

    The actor (:class:`PrivilegedActorTrunk`) acts on the encoder latent z of the privileged values
    in unmasked episodes, and on the history-estimator latent ẑ in masked ones. The fraction of
    masked episodes follows ``cfg.curriculum`` from fully observable to fully masked. After every
    PPO update, one more pass over the rollout minimizes::

        estimator_loss_scale * ||sg(z) - ẑ||²  +  encoder_loss_scale * ||z - sg(ẑ)||²

    so at the end of the curriculum the deployable actor needs only the observation history.
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
        cfg: PrivilegedPPO_CFG | dict = {},
    ) -> None:
        cfg = PrivilegedPPO_CFG(**cfg) if isinstance(cfg, dict) else cfg
        # the mask flag in the observation must reach the actor unscaled
        if cfg.observation_preprocessor is not None:
            raise ValueError("PrivilegedPPO does not support an observation preprocessor (it would rescale the mask flag)")
        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
            cfg=cfg,
        )
        self.cfg: PrivilegedPPO_CFG
        if not hasattr(getattr(self.policy, "trunk", None), "latents"):
            raise ValueError("The policy needs a `trunk` with a `latents()` method (see PrivilegedActorTrunk)")

    def post_interaction(self, *, timestep: int, timesteps: int) -> None:
        if self.cfg.curriculum is not None:
            self.cfg.curriculum.set_progress(timestep / max(timesteps, 1))
            self.track_data("Curriculum / Mask probability", self.cfg.curriculum.mask_probability())
        super().post_interaction(timestep=timestep, timesteps=timesteps)

    def update(self, *, timestep: int, timesteps: int) -> None:
        super().update(timestep=timestep, timesteps=timesteps)

        trunk = self.policy.trunk
        cumulative_estimator_loss = 0.0
        cumulative_encoder_loss = 0.0
        for (sampled_observations,) in self.memory.sample(
            names=["observations"], batch_size=len(self.memory), mini_batches=self.cfg.mini_batches
        ):
            z, z_hat = trunk.latents(sampled_observations)
            estimator_loss = F.mse_loss(z_hat, z.detach())
            encoder_loss = F.mse_loss(z, z_hat.detach())

            self.optimizer.zero_grad()
            self.scaler.scale(
                self.cfg.estimator_loss_scale * estimator_loss + self.cfg.encoder_loss_scale * encoder_loss
            ).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            cumulative_estimator_loss += estimator_loss.item()
            cumulative_encoder_loss += encoder_loss.item()

        self.track_data("Loss / Estimator loss", cumulative_estimator_loss / self.cfg.mini_batches)
        self.track_data("Loss / Encoder loss", cumulative_encoder_loss / self.cfg.mini_batches)
        masks = trunk.layout.mask(self.memory.get_tensor_by_name("observations").reshape(-1, trunk.layout.size))
        self.track_data("Curriculum / Masked fraction (rollout)", masks.mean().item())
