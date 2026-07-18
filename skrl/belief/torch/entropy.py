from __future__ import annotations

import math

import torch

from skrl.belief.torch.compression import GMMState


def gmm_entropy_upper_bound(gmm: GMMState) -> torch.Tensor:
    """Upper bound on differential entropy of a diagonal-covariance GMM.

    Uses the Huber et al. 2008 bound:
        H(p) <= -sum_i h_i * log(h_i) + sum_i h_i * H(N_i)
    where H(N_i) = 0.5 * log((2*pi*e)^d * det(Sigma_i))
                 = 0.5 * (d * log(2*pi*e) + sum_j log_sigma_ij^2)
    and log_diag_covs stores log(sigma^2), so:
        H(N_i) = 0.5 * (d * log(2*pi*e) + sum_j log_diag_covs[i,j])

    Used by active-localisation termination criterion (eq. 9 of paper).

    :param gmm: Fitted GMMState.
    :return: Scalar tensor, the entropy upper bound in nats.
    """
    d = gmm.state_dim
    log2pie = math.log(2 * math.pi * math.e)

    # Gaussian component entropies: 0.5 * (d*log(2pi e) + sum_j log_var_j)
    # log_diag_covs[i,j] = log(sigma^2), so this is exact
    h_gauss = 0.5 * (d * log2pie + gmm.log_diag_covs.sum(-1))  # [NG]

    # mixture entropy bound
    h_mixture = -(gmm.hefts * (gmm.hefts + 1e-10).log()).sum()  # scalar
    entropy = h_mixture + (gmm.hefts * h_gauss).sum()
    return entropy
