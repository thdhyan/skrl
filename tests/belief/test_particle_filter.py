"""Tests for ParticleFilter — filter convergence, entropy, resampling."""
from __future__ import annotations

import math

import pytest
import torch

from skrl.belief.backends.stub_backend import StubBackend
from skrl.belief.torch.particle_filter import ParticleFilter


N_JOINTS = 2
STATE_DIM = 10  # matches StubBackend default


def _make_filter(NP: int = 50, NG: int = 2, device: str = "cpu") -> tuple[ParticleFilter, StubBackend]:
    backend = StubBackend(n_joints=N_JOINTS, device=device, seed=42)

    def motion_model(particles, control):
        return backend.step(particles, control, n_substeps=1)

    def measurement_model(particles, measurement):
        return backend.get_measurement(particles, sigma_enc=1.0)

    pf = ParticleFilter(
        num_particles=NP,
        state_dim=STATE_DIM,
        num_gmm_components=NG,
        motion_model_fn=motion_model,
        measurement_model_fn=measurement_model,
        device=device,
        resample_threshold=0.5,
    )
    return pf, backend


def test_belief_dim():
    pf, _ = _make_filter(NP=50, NG=2)
    assert pf.belief_dim == 2 + 2 * 2 * STATE_DIM


def test_reset_shapes():
    pf, _ = _make_filter(NP=50)
    particles = torch.randn(50, STATE_DIM)
    pf.reset(particles)
    assert pf.get_particles().shape == (50, STATE_DIM)
    assert pf.get_weights().shape == (50,)


def test_uniform_weights_after_reset():
    pf, _ = _make_filter(NP=50)
    particles = torch.randn(50, STATE_DIM)
    pf.reset(particles)
    w = pf.get_weights().exp()
    assert torch.allclose(w, torch.ones(50) / 50, atol=1e-5)


def test_predict_changes_particles():
    pf, _ = _make_filter(NP=50)
    particles = torch.randn(50, STATE_DIM)
    pf.reset(particles.clone())
    before = pf.get_particles().clone()
    control = torch.zeros(N_JOINTS)
    pf.predict(control)
    # particles should have moved (dynamics + noise)
    assert not torch.allclose(pf.get_particles(), before)


def test_update_normalises_weights():
    pf, backend = _make_filter(NP=50)
    particles = torch.randn(50, STATE_DIM)
    pf.reset(particles)
    obs = torch.zeros(N_JOINTS)
    pf.update(obs)
    w = pf.get_weights().exp()
    assert abs(w.sum().item() - 1.0) < 1e-5


def test_compress_shape():
    pf, _ = _make_filter(NP=50, NG=2)
    particles = torch.randn(50, STATE_DIM)
    pf.reset(particles)
    pf.compress()
    b_tilde = pf.get_compressed()
    assert b_tilde.shape == (pf.belief_dim,)


def test_entropy_positive():
    pf, _ = _make_filter(NP=50, NG=2)
    pf.reset(torch.randn(50, STATE_DIM))
    pf.compress()
    h = pf.get_entropy()
    assert h.item() > 0.0


def test_entropy_decreases_after_update():
    """Entropy should (on average) decrease after incorporating a measurement
    that concentrates the distribution."""
    pf, backend = _make_filter(NP=200, NG=2)
    # start with wide prior
    particles = torch.randn(200, STATE_DIM) * 5.0
    pf.reset(particles)
    pf.compress()
    h_before = pf.get_entropy().item()

    # update with a measurement close to the origin (where gt_state is ~0)
    obs = torch.zeros(N_JOINTS)
    for _ in range(5):
        pf.predict(torch.zeros(N_JOINTS))
        pf.update(obs)

    pf.compress()
    h_after = pf.get_entropy().item()
    assert h_after < h_before, f"entropy did not decrease: {h_before:.3f} -> {h_after:.3f}"


def test_systematic_resample_triggered():
    """Force resampling by making all weight except one negligible."""
    pf, _ = _make_filter(NP=50)
    particles = torch.randn(50, STATE_DIM)
    pf.reset(particles)

    # manufacture degenerate log-weights (one particle dominates)
    log_w = torch.full((50,), -100.0)
    log_w[25] = 0.0
    pf._log_weights = log_w - torch.logsumexp(log_w, dim=0)

    obs = torch.zeros(N_JOINTS)
    pf.update(obs)

    # after resampling, weights should be uniform again
    w = pf.get_weights().exp()
    assert abs(w.sum().item() - 1.0) < 1e-5


@pytest.mark.parametrize("device", ["cpu"])
def test_full_predict_update_compress_cycle(device):
    pf, backend = _make_filter(NP=100, NG=2, device=device)
    backend.reset(seed=0)
    particles = torch.randn(100, STATE_DIM)
    pf.reset(particles)

    for _ in range(3):
        control = torch.zeros(N_JOINTS)
        obs = backend.step_ground_truth(control)
        pf.predict(control)
        pf.update(obs)

    pf.compress()
    b = pf.get_compressed()
    assert b.shape == (pf.belief_dim,)
    assert not torch.isnan(b).any()
    assert not torch.isinf(b).any()
