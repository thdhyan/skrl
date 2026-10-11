"""Tests for the belief markers, statistics, HTTP server and MuJoCo drawing."""
from __future__ import annotations

import pytest
import torch

from skrl.belief.torch import ParticleFilter
from skrl.belief.torch.markers import (
    BeliefMarkerServer,
    belief_markers,
    belief_stats,
    fetch_belief_markers,
    mujoco_add_belief_geoms,
)


def _filter(particles: torch.Tensor, ng: int = 2) -> ParticleFilter:
    pf = ParticleFilter(
        num_particles=particles.shape[0],
        state_dim=particles.shape[1],
        num_gmm_components=ng,
        motion_model_fn=lambda p, c: p,
        measurement_model_fn=lambda p, m: torch.zeros(p.shape[0]),
        device="cpu",
    )
    pf.reset(particles)
    pf.compress()
    return pf


def _two_clusters(spread: float = 0.1) -> torch.Tensor:
    torch.manual_seed(0)
    a = torch.randn(100, 3) * spread + torch.tensor([1.0, 0.0, 0.0])
    b = torch.randn(100, 3) * spread - torch.tensor([1.0, 0.0, 0.0])
    return torch.cat([a, b])


def test_stats_two_clusters_and_true_state():
    pf = _filter(_two_clusters())
    stats = belief_stats(pf, torch.tensor([1.0, 0.0, 0.0]))
    assert stats["num_clusters"] == 2
    assert stats["effective_clusters"] == pytest.approx(2.0, abs=0.1)
    assert stats["spread"] == pytest.approx(1.0, abs=0.1)
    assert stats["ess_fraction"] == pytest.approx(1.0)
    assert stats["cluster_std"] == pytest.approx(0.1, abs=0.03)
    assert stats["cluster_error_min"] < 0.05  # one cluster sits on the true state
    assert stats["cluster_error_weighted"] == pytest.approx(1.0, abs=0.1)
    assert stats["mean_error"] == pytest.approx(1.0, abs=0.1)
    assert len(stats["cluster_errors"]) == len(stats["cluster_hefts"]) == 2
    assert stats["normalized_error"] < 3.0


def test_spread_shrinks_with_tighter_belief():
    wide, tight = belief_stats(_filter(_two_clusters(0.5))), belief_stats(_filter(_two_clusters(0.5) * 0.1))
    assert tight["spread"] < wide["spread"] and tight["cluster_std"] < wide["cluster_std"]


def test_markers_positions_and_server():
    pf = _filter(_two_clusters())
    markers = belief_markers(pf, torch.zeros(3), state_to_position=lambda s: s[..., :3] * 2)
    assert markers["particles"].shape == (200, 3) and markers["weights"].sum() == pytest.approx(1.0)
    assert markers["cluster_means"].shape == markers["cluster_stds"].shape == (2, 3)
    assert (markers["cluster_hefts"][:-1] >= markers["cluster_hefts"][1:]).all()
    assert markers["particle_positions"].shape == (200, 3) and markers["true_position"].shape == (3,)

    server = BeliefMarkerServer(port=0)
    try:
        url = f"http://127.0.0.1:{server.address[1]}/belief"
        assert fetch_belief_markers(url)["seq"] == 0
        server.publish(markers)
        data = fetch_belief_markers(url)
        assert data["seq"] == 1 and len(data["particles"]) == 200 and "spread" in data["stats"]
    finally:
        server.close()


def test_mujoco_geoms():
    mujoco = pytest.importorskip("mujoco")
    pf = _filter(_two_clusters())
    markers = belief_markers(pf, torch.zeros(3), state_to_position=lambda s: s[..., :3])
    model = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
    scene = mujoco.MjvScene(model, maxgeom=100)
    added = mujoco_add_belief_geoms(scene, markers, max_particles=20)
    assert added == 20 + 2 + 1 == scene.ngeom
