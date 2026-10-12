from __future__ import annotations

from typing import Any

import dataclasses

import gymnasium
import numpy as np
import torch
import torch.nn as nn
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

    sysid_loss_scale: float = 0.0
    """Weight of the supervised system-identification loss ``||ψ(ẑ) - ē_t||² + ||ψ(z) - ē_t||²``, where ``ē_t`` are
    the privileged values standardized per dimension over the rollout and ψ is the trunk's ``sysid`` decoder
    (``PrivilegedActorTrunk(sysid_head=True)``). The ẑ term makes the history estimator regress the privileged
    values directly, with every dimension weighted equally; the z term makes (encoder, ψ) an autoencoder, so z and ẑ
    are decoded by the same ψ. ``0``: off (the estimator only matches z)."""

    estimator_whiten: bool = False
    """Compute ``||sg(z) - ẑ||²`` in the whitened coordinates of z (PCA of the mini-batch, every direction scaled to
    unit variance), so the estimator must match the low-variance directions of z (e.g. mass) as much as the dominant
    one (velocity). With an ``osi`` trunk the estimator regresses the standardized privileged values instead."""

    contrastive_loss_scale: float = 0.0
    """Weight of a supervised-contrastive (InfoNCE) loss on the projection ``g(ẑ)`` (trunk ``contrastive_head=True``):
    samples whose ``contrastive_dims`` are equal (same episode dynamics) are positives, all others negatives. It makes
    ẑ separate episodes by their dynamics regardless of the state. ``0``: off."""

    contrastive_dims: list[int] | None = None
    """Privileged dimensions that are constant within an episode (the randomized dynamics); they define positives."""

    contrastive_temperature: float = 0.1

    freeze_policy_after: float | None = None
    """RMA two-phase training: after this fraction of the training timesteps, PPO stops and the policy, value and
    encoder are frozen; only the estimator (and the sysid / contrastive heads) keep training on the new rollouts
    (RMA phase 2: the curriculum should mask every episode then, so the student collects its own data).
    ``None``: one phase (ROA)."""

    privileged_groups: dict[str, list[int]] | None = None
    """Named groups of privileged dimensions (e.g. ``{"velocities": [0, 1], "dynamics": [2, 3]}``). Two probes
    decode the (standardized) privileged values from the detached latents, one from the encoder latent z and one
    from the history estimator latent ẑ, and their R² is logged per group (``"Probe / ẑ R2 <group>"``): how much
    of each privileged group the history recovers (online system identification). ``None``: one group per dim."""

    probe_learning_rate: float = 1e-3
    """Learning rate of the probes (they never change the policy)."""


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
        layout = self.policy.trunk.layout
        self.privileged_groups = self.cfg.privileged_groups or {f"dim{i}": [i] for i in range(layout.privileged_dim)}
        if self.cfg.sysid_loss_scale > 0 and getattr(self.policy.trunk, "sysid", None) is None:
            raise ValueError("sysid_loss_scale > 0 needs a trunk with a `sysid` decoder (sysid_head=True)")
        if self.cfg.contrastive_loss_scale > 0 and (
            getattr(self.policy.trunk, "contrast", None) is None or not self.cfg.contrastive_dims
        ):
            raise ValueError("contrastive_loss_scale > 0 needs contrastive_head=True and contrastive_dims")
        self._probes: nn.ModuleDict | None = None
        self.probe_metrics: dict[str, float] = {}
        """Last probe R² per group and latent (``"ẑ R2 <group>"``, ``"z R2 <group>"``), measured on each new rollout
        before the probes train on it."""

    def post_interaction(self, *, timestep: int, timesteps: int) -> None:
        if self.cfg.curriculum is not None:
            self.cfg.curriculum.set_progress(timestep / max(timesteps, 1))
            self.track_data("Curriculum / Mask probability", self.cfg.curriculum.mask_probability())
        super().post_interaction(timestep=timestep, timesteps=timesteps)

    def update(self, *, timestep: int, timesteps: int) -> None:
        curriculum = self.cfg.curriculum
        if hasattr(curriculum, "observe_return") and len(self._track_rewards):
            curriculum.observe_return(float(np.mean(self._track_rewards)))
            self.track_data("Curriculum / Smoothed return", curriculum.smoothed)

        # RMA phase 2: no PPO update, the policy, value and encoder stay frozen
        frozen = self.cfg.freeze_policy_after is not None and timestep >= self.cfg.freeze_policy_after * timesteps
        if not frozen:
            super().update(timestep=timestep, timesteps=timesteps)
        self.track_data("Curriculum / Policy frozen", float(frozen))

        trunk = self.policy.trunk
        observations = self.memory.get_tensor_by_name("observations").reshape(-1, trunk.layout.size)
        sysid = self.cfg.sysid_loss_scale > 0
        contrastive = self.cfg.contrastive_loss_scale > 0
        privileged = trunk.layout.privileged(observations)
        mean, std = privileged.mean(0), privileged.std(0) + 1e-6
        if sysid or getattr(trunk, "osi", False):
            self._track_sysid(observations, mean, std)
        cumulative_estimator_loss = 0.0
        cumulative_encoder_loss = 0.0
        cumulative_sysid_loss = 0.0
        cumulative_contrastive_loss = 0.0
        for (sampled_observations,) in self.memory.sample(
            names=["observations"], batch_size=len(self.memory), mini_batches=self.cfg.mini_batches
        ):
            z, z_hat = trunk.latents(sampled_observations)
            if frozen:
                z = z.detach()
            sampled_privileged = trunk.layout.privileged(sampled_observations)
            if getattr(trunk, "osi", False):  # UP-OSI: regress the privileged values, every dimension standardized
                estimator_loss = ((trunk.estimate(sampled_observations) - sampled_privileged) / std).pow(2).mean()
            elif self.cfg.estimator_whiten:
                w = self._whitening(z.detach())
                estimator_loss = F.mse_loss(z_hat @ w, z.detach() @ w)
            else:
                estimator_loss = F.mse_loss(z_hat, z.detach())
            encoder_loss = F.mse_loss(z, z_hat.detach())
            loss = self.cfg.estimator_loss_scale * estimator_loss
            if not frozen:
                loss = loss + self.cfg.encoder_loss_scale * encoder_loss
            if sysid:
                target = (sampled_privileged - mean) / std
                sysid_loss = F.mse_loss(trunk.sysid(z_hat), target) + F.mse_loss(trunk.sysid(z), target)
                loss = loss + self.cfg.sysid_loss_scale * sysid_loss
                cumulative_sysid_loss += sysid_loss.item()
            if contrastive:
                contrastive_loss = self._contrastive_loss(z_hat, sampled_privileged[:, self.cfg.contrastive_dims])
                loss = loss + self.cfg.contrastive_loss_scale * contrastive_loss
                cumulative_contrastive_loss += contrastive_loss.item()

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            cumulative_estimator_loss += estimator_loss.item()
            cumulative_encoder_loss += encoder_loss.item()

        self.track_data("Loss / Estimator loss", cumulative_estimator_loss / self.cfg.mini_batches)
        self.track_data("Loss / Encoder loss", cumulative_encoder_loss / self.cfg.mini_batches)
        if sysid:
            self.track_data("Loss / SysID loss", cumulative_sysid_loss / self.cfg.mini_batches)
        if contrastive:
            self.track_data("Loss / Contrastive loss", cumulative_contrastive_loss / self.cfg.mini_batches)
        self.track_data("Curriculum / Masked fraction (rollout)", trunk.layout.mask(observations).mean().item())
        self._update_probes(observations)

    @staticmethod
    def _whitening(z: torch.Tensor) -> torch.Tensor:
        """Matrix ``W`` with ``cov(z W) = I`` (PCA whitening of the batch; tiny directions regularized)."""
        centered = z - z.mean(0)
        cov = centered.T @ centered / (len(z) - 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        return eigenvectors / (eigenvalues.clamp(min=0) + 1e-3 * eigenvalues.mean() + 1e-8).sqrt()

    def _contrastive_loss(self, z_hat: torch.Tensor, dynamics: torch.Tensor) -> torch.Tensor:
        """Supervised-contrastive loss: samples with the same dynamics (same episode) are positives."""
        projection = F.normalize(self.policy.trunk.contrast(z_hat), dim=-1)
        logits = projection @ projection.T / self.cfg.contrastive_temperature
        eye = torch.eye(len(z_hat), dtype=torch.bool, device=z_hat.device)
        log_prob = logits.masked_fill(eye, float("-inf")).log_softmax(dim=-1)
        positives = (torch.cdist(dynamics, dynamics) < 1e-6) & ~eye
        count = positives.sum(-1)
        loss = -log_prob.masked_fill(~positives, 0).sum(-1) / count.clamp(min=1)
        return loss[count > 0].mean() if (count > 0).any() else z_hat.sum() * 0

    @torch.no_grad()
    def _track_sysid(self, observations: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> None:
        """R² per group of the system-ID head ψ(ẑ), and of the OSI estimate ê, on the new rollout, before they train on
        it (held out)."""
        trunk = self.policy.trunk
        target = (trunk.layout.privileged(observations) - mean) / std
        estimates = {}
        if trunk.sysid is not None:
            estimates["ψ(ẑ)"] = trunk.sysid(trunk.latents(observations)[1])
        if getattr(trunk, "osi", False):
            estimates["ê"] = (trunk.estimate(observations) - mean) / std
        for name, estimate in estimates.items():
            error = (estimate - target).pow(2).mean(0)
            for group, dims in self.privileged_groups.items():
                r2 = 1.0 - error[dims].mean().item()
                self.probe_metrics[f"{name} R2 {group}"] = r2
                self.track_data(f"SysID / {name} R2 {group}", r2)

    def _update_probes(self, observations: torch.Tensor) -> None:
        """Evaluate, then train, the z and ẑ probes on the rollout (privileged values standardized per rollout)."""
        trunk = self.policy.trunk
        with torch.no_grad():
            z, z_hat = trunk.latents(observations)
            target = trunk.layout.privileged(observations)
            target = (target - target.mean(0)) / (target.std(0) + 1e-6)
        if self._probes is None:
            hidden = 64
            self._probes = nn.ModuleDict(
                {
                    name: nn.Sequential(nn.Linear(z.shape[-1], hidden), nn.ELU(), nn.Linear(hidden, target.shape[-1]))
                    for name in ("z", "z_hat")
                }
            ).to(self.device)
            self._probe_optimizer = torch.optim.Adam(self._probes.parameters(), lr=self.cfg.probe_learning_rate)

        latents = {"z": z, "z_hat": z_hat}
        with torch.no_grad():  # held-out evaluation: the probes have not seen this rollout yet
            for name, latent in latents.items():
                error = (self._probes[name](latent) - target).pow(2).mean(0)  # per dim; target variance is 1
                label = "ẑ" if name == "z_hat" else "z"
                for group, dims in self.privileged_groups.items():
                    r2 = 1.0 - error[dims].mean().item()
                    self.probe_metrics[f"{label} R2 {group}"] = r2
                    self.track_data(f"Probe / {label} R2 {group}", r2)
        for indices in torch.randperm(len(target), device=self.device).chunk(self.cfg.mini_batches):
            loss = sum(F.mse_loss(self._probes[n](latent[indices]), target[indices]) for n, latent in latents.items())
            self._probe_optimizer.zero_grad()
            loss.backward()
            self._probe_optimizer.step()
