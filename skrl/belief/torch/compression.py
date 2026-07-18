from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class GMMState:
    """Fitted GMM parameters.

    :param hefts: Component weights, shape ``[NG]``, sums to 1.
    :param means: Component means, shape ``[NG, state_dim]``.
    :param log_diag_covs: Log of diagonal covariance entries,
        shape ``[NG, state_dim]``.  Using the log avoids scaling
        instabilities during neural network training (paper §V-A).
    :param num_components: NG.
    :param state_dim: Dimension of a single state vector.
    """

    hefts: torch.Tensor
    means: torch.Tensor
    log_diag_covs: torch.Tensor
    num_components: int
    state_dim: int


def _init_kmeans(particles: torch.Tensor, k: int) -> torch.Tensor:
    """Approximate k-medoids initialisation for GMM-EM (paper §V-A).

    Uses k-means++ seeding — picks the first centre randomly, then each
    subsequent centre proportional to distance-squared from the nearest
    existing centre.  This is faster than full k-medoids and sufficient
    for small k (NG = 2).

    :param particles: Shape ``[NP, state_dim]``.
    :param k: Number of clusters.
    :return: Initial centres, shape ``[k, state_dim]``.
    """
    NP = particles.shape[0]
    idx = torch.randint(NP, (1,), device=particles.device).item()
    centres = [particles[idx]]
    for _ in range(k - 1):
        dists = torch.stack([((particles - c) ** 2).sum(-1) for c in centres], dim=0).min(0).values
        total = dists.sum()
        if total < 1e-12:
            idx = torch.randint(NP, (1,), device=particles.device).item()
        else:
            probs = dists / total
            idx = torch.multinomial(probs.clamp(min=0), 1).item()
        centres.append(particles[idx])
    return torch.stack(centres, dim=0)


def gmm_fit(
    particles: torch.Tensor,
    weights: torch.Tensor,
    ng: int = 2,
    max_iter: int = 50,
    tol: float = 1e-4,
    eps: float = 1e-6,
) -> GMMState:
    """Fit a weighted GMM to a particle set using EM.

    The paper uses NG=2 with QGMM for SE(3) orientations (Kim et al. 2017).
    We operate in Euclidean space on the full state vector; callers that need
    SE(3)-aware fitting should subclass and override this function.

    Covariances are diagonal only — the off-diagonal entries are discarded.
    This keeps belief_dim tractable and matches the paper's input layout.

    :param particles: Shape ``[NP, state_dim]``.
    :param weights: Normalised importance weights, shape ``[NP]``.
    :param ng: Number of mixture components.
    :param max_iter: Maximum EM iterations.
    :param tol: Log-likelihood convergence threshold.
    :param eps: Numerical stability floor for covariances.
    :return: Fitted GMMState.
    """
    NP, state_dim = particles.shape
    device = particles.device
    dtype = particles.dtype

    # initialise responsibilities
    centres = _init_kmeans(particles, ng)  # [NG, state_dim]
    dists = torch.cdist(particles, centres)  # [NP, NG]
    r = torch.zeros(NP, ng, device=device, dtype=dtype)
    r.scatter_(1, dists.argmin(1, keepdim=True), 1.0)

    hefts = torch.full((ng,), 1.0 / ng, device=device, dtype=dtype)
    means = centres.clone()
    log_diag_covs = torch.zeros(ng, state_dim, device=device, dtype=dtype)

    prev_ll = float("-inf")
    for _ in range(max_iter):
        # --- M step ---
        # Effective count per component (weighted by particle weights)
        w_r = weights.unsqueeze(1) * r  # [NP, NG]
        nk = w_r.sum(0).clamp(min=eps)  # [NG]
        hefts = nk / nk.sum()
        means = (w_r.T @ particles) / nk.unsqueeze(1)  # [NG, state_dim]
        diff = particles.unsqueeze(1) - means.unsqueeze(0)  # [NP, NG, state_dim]
        var = (w_r.unsqueeze(2) * diff ** 2).sum(0) / nk.unsqueeze(1) + eps  # [NG, state_dim]
        log_diag_covs = var.log()

        # --- E step ---
        # log p(x_i | component k)
        log_p = -0.5 * (
            state_dim * torch.log(torch.tensor(2 * 3.14159265358979, device=device, dtype=dtype))
            + log_diag_covs.sum(-1)  # [NG]
            + (diff ** 2 / var.unsqueeze(0)).sum(-1)  # [NP, NG]
        )
        log_r = log_p + hefts.log().unsqueeze(0)  # [NP, NG]
        log_sum = torch.logsumexp(log_r, dim=1, keepdim=True)  # [NP, 1]
        r = (log_r - log_sum).exp()  # [NP, NG]

        ll = (weights * log_sum.squeeze(1)).sum().item()
        if abs(ll - prev_ll) < tol:
            break
        prev_ll = ll

    return GMMState(
        hefts=hefts,
        means=means,
        log_diag_covs=log_diag_covs,
        num_components=ng,
        state_dim=state_dim,
    )


def flatten_gmm(gmm: GMMState) -> torch.Tensor:
    """Flatten GMMState to the b̃ vector used as policy input (eq. 5 of paper).

    Layout: ``[h_1..h_NG | mu_1..mu_NG | log_diag_cov_1..log_diag_cov_NG]``
    Shape: ``[NG + 2 * NG * state_dim]``

    :param gmm: Fitted GMMState.
    :return: Flat tensor, shape ``[NG + 2 * NG * state_dim]``.
    """
    return torch.cat([
        gmm.hefts,                                        # [NG]
        gmm.means.reshape(-1),                            # [NG * state_dim]
        gmm.log_diag_covs.reshape(-1),                    # [NG * state_dim]
    ], dim=0)


def belief_dim(ng: int, state_dim: int) -> int:
    """Return the length of the flattened belief vector."""
    return ng + 2 * ng * state_dim
