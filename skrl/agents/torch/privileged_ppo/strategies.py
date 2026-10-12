from __future__ import annotations

import copy

# Named system-identification strategies for PrivilegedPPO: (policy-trunk kwargs, agent cfg, default curriculum).
# Values set explicitly in the config override the preset. ``contrastive`` also needs ``contrastive_dims`` (the
# privileged dimensions that are constant within an episode), which depend on the task.
STRATEGIES: dict[str, tuple[dict, dict, dict | None]] = {
    # ROA: estimator ||sg(z) - ẑ||² and encoder regularizer λ||z - sg(ẑ)||², one phase
    "roa": ({}, {"encoder_loss_scale": 0.1}, None),
    # ROA without the encoder regularizer (z keeps what the history cannot predict)
    "roa_lam0": ({}, {"encoder_loss_scale": 0.0}, None),
    # supervised system-ID head ψ: ψ(ẑ) and ψ(z) regress the standardized privileged values
    "sysid": ({"sysid_head": True}, {"sysid_loss_scale": 1.0, "encoder_loss_scale": 0.0}, None),
    # supervised-contrastive InfoNCE on g(ẑ): same episode dynamics = positives
    "contrastive": ({"contrastive_head": True}, {"contrastive_loss_scale": 0.1, "encoder_loss_scale": 0.0}, None),
    # estimator loss in the whitened (PCA) coordinates of z
    "whiten": ({}, {"estimator_whiten": True, "encoder_loss_scale": 0.0}, None),
    # UP-OSI: the estimator regresses the privileged values ê, the policy reads ẑ = μ(sg(ê))
    "osi": ({"osi": True}, {"encoder_loss_scale": 0.0}, None),
    # RMA two-phase: teacher on z until 75 % of training, then PPO stops and only the estimator trains on
    # fully masked rollouts
    "rma": (
        {},
        {"freeze_policy_after": 0.75, "encoder_loss_scale": 0.0},
        {"class": "TwoPhaseMaskCurriculum", "switch": 0.75},
    ),
    # RMA two-phase with a UP-OSI estimator
    "uposi": (
        {"osi": True},
        {"freeze_policy_after": 0.75, "encoder_loss_scale": 0.0},
        {"class": "TwoPhaseMaskCurriculum", "switch": 0.75},
    ),
}


def apply_strategy(
    name: str | None, policy_kwargs: dict, agent_cfg: dict
) -> tuple[dict, dict]:
    """Merge the preset of strategy ``name`` under the given policy kwargs and agent cfg (explicit values win).

    :return: ``(policy_kwargs, agent_cfg)``. ``agent_cfg["curriculum"]`` gets the preset curriculum spec if unset.
    """
    if name is None:
        return policy_kwargs, agent_cfg
    if name.lower() not in STRATEGIES:
        raise ValueError(f"unknown strategy '{name}', expected one of {list(STRATEGIES)}")
    model_preset, agent_preset, curriculum = copy.deepcopy(STRATEGIES[name.lower()])
    agent_cfg = {**agent_preset, **agent_cfg}
    if agent_cfg.get("curriculum") is None and curriculum is not None:
        agent_cfg["curriculum"] = curriculum
    if (
        agent_cfg.get("freeze_policy_after") is not None
        and isinstance(agent_cfg.get("curriculum"), dict)
        and agent_cfg["curriculum"].get("class", "").lower() == "twophasemaskcurriculum"
    ):
        agent_cfg["curriculum"].setdefault("switch", agent_cfg["freeze_policy_after"])
    return {**model_preset, **policy_kwargs}, agent_cfg
