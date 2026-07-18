"""Tests for GMM compression and entropy."""
from __future__ import annotations

import torch
import pytest

from skrl.belief.torch.compression import gmm_fit, flatten_gmm, belief_dim, GMMState
from skrl.belief.torch.entropy import gmm_entropy_upper_bound


STATE_DIM = 6
NG = 2


def _make_particles_two_clusters(n: int = 200) -> tuple[torch.Tensor, torch.Tensor]:
    """Two well-separated Gaussian clusters."""
    torch.manual_seed(0)
    c1 = torch.randn(n // 2, STATE_DIM) * 0.3 + torch.tensor([5.0] + [0.0] * (STATE_DIM - 1))
    c2 = torch.randn(n // 2, STATE_DIM) * 0.3 - torch.tensor([5.0] + [0.0] * (STATE_DIM - 1))
    particles = torch.cat([c1, c2], dim=0)
    weights = torch.ones(n) / n
    return particles, weights


def test_gmm_fit_shapes():
    particles, weights = _make_particles_two_clusters()
    gmm = gmm_fit(particles, weights, ng=NG)
    assert gmm.hefts.shape == (NG,)
    assert gmm.means.shape == (NG, STATE_DIM)
    assert gmm.log_diag_covs.shape == (NG, STATE_DIM)


def test_gmm_fit_hefts_sum_to_one():
    particles, weights = _make_particles_two_clusters()
    gmm = gmm_fit(particles, weights, ng=NG)
    assert abs(gmm.hefts.sum().item() - 1.0) < 1e-5


def test_gmm_fit_recovers_cluster_means():
    """Fitted means should be close to the two true cluster centres."""
    particles, weights = _make_particles_two_clusters(n=400)
    gmm = gmm_fit(particles, weights, ng=NG)
    # sort means by first coordinate
    means = gmm.means[gmm.means[:, 0].argsort()]
    assert means[0, 0].item() < -4.0
    assert means[1, 0].item() > 4.0


def test_flatten_gmm_shape():
    particles, weights = _make_particles_two_clusters()
    gmm = gmm_fit(particles, weights, ng=NG)
    b = flatten_gmm(gmm)
    expected = belief_dim(NG, STATE_DIM)
    assert b.shape == (expected,)


def test_flatten_gmm_no_nan():
    particles, weights = _make_particles_two_clusters()
    gmm = gmm_fit(particles, weights, ng=NG)
    b = flatten_gmm(gmm)
    assert not torch.isnan(b).any()
    assert not torch.isinf(b).any()


def test_belief_dim_formula():
    assert belief_dim(2, 10) == 2 + 2 * 2 * 10  # 42
    assert belief_dim(3, 6) == 3 + 2 * 3 * 6    # 39


def test_entropy_upper_bound_positive():
    particles, weights = _make_particles_two_clusters()
    gmm = gmm_fit(particles, weights, ng=NG)
    h = gmm_entropy_upper_bound(gmm)
    assert h.item() > 0.0


def test_entropy_decreases_with_narrower_clusters():
    """Narrower clusters → lower entropy."""
    torch.manual_seed(1)
    p_wide = torch.randn(200, STATE_DIM) * 3.0
    p_narrow = torch.randn(200, STATE_DIM) * 0.1
    w = torch.ones(200) / 200

    gmm_wide = gmm_fit(p_wide, w, ng=1)
    gmm_narrow = gmm_fit(p_narrow, w, ng=1)
    h_wide = gmm_entropy_upper_bound(gmm_wide).item()
    h_narrow = gmm_entropy_upper_bound(gmm_narrow).item()
    assert h_narrow < h_wide


def test_log_diag_covs_finite():
    """log-cov should be finite for any reasonable particle set."""
    torch.manual_seed(2)
    particles = torch.randn(100, STATE_DIM)
    weights = torch.ones(100) / 100
    gmm = gmm_fit(particles, weights, ng=NG)
    assert torch.isfinite(gmm.log_diag_covs).all()
