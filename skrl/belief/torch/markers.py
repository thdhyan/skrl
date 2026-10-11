"""Belief markers and statistics for visualization in a simulator and for logging.

A simulator (Isaac Sim / Isaac Lab, MuJoCo) can draw the belief of a :class:`~skrl.belief.torch.Belief` as
particles (points scaled by weight) and GMM clusters (ellipsoids, one per component), next to the true state,
and watch the belief shrink as measurements arrive.

* :func:`belief_markers` returns the particles, weights and GMM clusters as tensors, optionally mapped to world
  positions with a user ``state_to_position`` function, plus :func:`belief_stats`.
* :func:`belief_stats` summarizes the belief as scalars (spread, effective sample size, number of clusters,
  cluster stds) and, given the true state, the errors of the belief mean and of every cluster mean.
* :func:`markers_to_json` makes the markers JSON-serializable for another process.
* :class:`BeliefMarkerServer` serves the latest markers over HTTP (``GET /belief``), so a simulator in another
  process can poll them in real time with :func:`fetch_belief_markers`.
* :func:`mujoco_add_belief_geoms` draws markers into a MuJoCo ``mjvScene`` (e.g. ``viewer.user_scn``).
"""

from __future__ import annotations

from typing import Any, Callable

import json
import threading

import torch

from skrl.belief.torch.base import Belief


def belief_stats(
    belief: Belief, true_state: torch.Tensor | None = None, *, min_heft: float = 0.05
) -> dict[str, float | list[float]]:
    """Scalar summary of the belief (call after ``compress()``).

    Keys:

    * ``spread``: weighted RMS distance of the particles to their weighted mean.
    * ``ess`` / ``ess_fraction``: effective sample size ``1 / sum(w^2)`` (and divided by the number of particles).
    * ``num_clusters``: GMM components with heft >= ``min_heft``; ``effective_clusters``: ``1 / sum(h^2)``.
    * ``cluster_std``: heft-weighted mean of the component standard deviations (mean over state dims);
      ``cluster_std_max``: largest component std.
    * ``entropy``: GMM entropy upper bound.
    * ``cluster_hefts`` / ``cluster_stds``: per component, sorted by decreasing heft (lists).

    With ``true_state`` ``[state_dim]`` also:

    * ``mean_error``: distance of the weighted particle mean to the true state.
    * ``cluster_error_min`` / ``cluster_error_weighted`` / ``cluster_error_std``: distance of the cluster means to
      the true state: closest cluster, heft-weighted mean, std across clusters with heft >= ``min_heft``.
    * ``cluster_errors``: per component, same order as ``cluster_hefts`` (list).
    * ``normalized_error``: ``mean(|mu - s| / sigma)`` of the closest cluster (about 1 when the belief is calibrated,
      much larger when it is over-confident).
    """
    particles = belief.get_particles().float()
    weights = belief.get_weights().float().exp()
    weights = weights / weights.sum()
    mean = (weights[:, None] * particles).sum(0)
    stats: dict[str, float | list[float]] = {
        "spread": (weights * (particles - mean).pow(2).sum(-1)).sum().sqrt().item(),
        "ess": (1.0 / weights.pow(2).sum()).item(),
    }
    stats["ess_fraction"] = stats["ess"] / len(weights)
    gmm = belief.gmm
    if gmm is None:
        belief.compress()
        gmm = belief.gmm
    order = gmm.hefts.argsort(descending=True)
    hefts = gmm.hefts[order].float()
    stds = (0.5 * gmm.log_diag_covs[order].float()).exp()  # [NG, d]
    component_std = stds.mean(-1)
    stats.update(
        num_clusters=int((hefts >= min_heft).sum()),
        effective_clusters=(1.0 / hefts.pow(2).sum()).item(),
        cluster_std=(hefts * component_std).sum().item() / hefts.sum().item(),
        cluster_std_max=component_std.max().item(),
        entropy=belief.get_entropy().item(),
        cluster_hefts=hefts.tolist(),
        cluster_stds=component_std.tolist(),
    )
    if true_state is not None:
        true_state = true_state.to(particles).view(-1)
        errors = (gmm.means[order].float() - true_state).norm(dim=-1)
        significant = errors[hefts >= min_heft]
        closest = errors.argmin()
        stats.update(
            mean_error=(mean - true_state).norm().item(),
            cluster_error_min=errors.min().item(),
            cluster_error_weighted=(hefts * errors).sum().item() / hefts.sum().item(),
            cluster_error_std=significant.std().item() if len(significant) > 1 else 0.0,
            cluster_errors=errors.tolist(),
            normalized_error=((gmm.means[order][closest] - true_state).abs() / stds[closest]).mean().item(),
        )
    return stats


def belief_markers(
    belief: Belief,
    true_state: torch.Tensor | None = None,
    *,
    state_to_position: Callable[[torch.Tensor], torch.Tensor] | None = None,
    min_heft: float = 0.05,
) -> dict[str, Any]:
    """Particles, weights and GMM clusters of the belief, for drawing in a simulator.

    :param belief: Belief (particle filter); ``compress()`` is called if no GMM has been fitted yet.
    :param true_state: Optional true state ``[state_dim]`` (adds ``true_state`` and the error statistics).
    :param state_to_position: Optional ``f(states[..., state_dim]) -> positions[..., 3]`` that maps belief states
        (e.g. joint angles, object pose, cart position) to world positions. Adds ``particle_positions [NP, 3]``,
        ``cluster_positions [NG, 3]``, ``cluster_position_stds [NG, 3]`` (std of 256 samples of each component
        mapped to world positions) and ``true_position [3]``.
    :param min_heft: Heft threshold of a "significant" cluster in the statistics.

    :return: Detached tensors: ``particles [NP, d]``, ``weights [NP]`` (probabilities), ``cluster_hefts [NG]``,
        ``cluster_means [NG, d]``, ``cluster_stds [NG, d]`` (components sorted by decreasing heft), optionally the
        positions above, and ``stats`` (:func:`belief_stats`).
    """
    stats = belief_stats(belief, true_state, min_heft=min_heft)
    gmm = belief.gmm
    order = gmm.hefts.argsort(descending=True)
    weights = belief.get_weights().detach().exp()
    markers: dict[str, Any] = {
        "particles": belief.get_particles().detach(),
        "weights": weights / weights.sum(),
        "cluster_hefts": gmm.hefts[order].detach(),
        "cluster_means": gmm.means[order].detach(),
        "cluster_stds": (0.5 * gmm.log_diag_covs[order]).exp().detach(),
        "stats": stats,
    }
    if true_state is not None:
        markers["true_state"] = true_state.detach().view(-1)
    if state_to_position is not None:
        markers["particle_positions"] = state_to_position(markers["particles"])
        markers["cluster_positions"] = state_to_position(markers["cluster_means"])
        samples = markers["cluster_means"][:, None] + markers["cluster_stds"][:, None] * torch.randn(
            markers["cluster_means"].shape[0], 256, markers["cluster_means"].shape[1], device=gmm.means.device
        )
        markers["cluster_position_stds"] = state_to_position(samples).std(dim=1)
        if true_state is not None:
            markers["true_position"] = state_to_position(markers["true_state"][None])[0]
    return markers


def markers_to_json(markers: dict[str, Any]) -> dict[str, Any]:
    """Tensors to nested lists (JSON-serializable)."""
    return {k: v.tolist() if isinstance(v, torch.Tensor) else v for k, v in markers.items()}


class BeliefMarkerServer:
    """Serve the latest belief markers as JSON over HTTP, in a background thread.

    The training or evaluation process calls :meth:`publish` (e.g. every step); the simulator polls
    ``GET http://<host>:<port>/belief`` (see :func:`fetch_belief_markers`). Only the latest markers are kept.
    The response also has a ``seq`` counter so the client can skip unchanged data.

    Example::

        server = BeliefMarkerServer(port=8765)
        ...
        server.publish(agent.get_belief_markers(true_state=info_state))
        ...
        server.close()
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8765) -> None:
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        self._lock = threading.Lock()
        self._payload = b'{"seq": 0}'
        self.seq = 0
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                if self.path.split("?")[0] != "/belief":
                    self.send_error(404)
                    return
                with server._lock:
                    payload = server._payload
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args: Any) -> None:
                pass

        self._httpd = ThreadingHTTPServer((host, port), Handler)
        self.address = self._httpd.server_address
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def publish(self, markers: dict[str, Any]) -> None:
        self.seq += 1
        payload = json.dumps({"seq": self.seq, **markers_to_json(markers)}).encode()
        with self._lock:
            self._payload = payload

    def close(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()


def fetch_belief_markers(url: str = "http://127.0.0.1:8765/belief", timeout: float = 1.0) -> dict[str, Any]:
    """Poll a :class:`BeliefMarkerServer` (simulator side). Values are nested lists (convert with ``np.asarray``)."""
    from urllib.request import urlopen

    with urlopen(url, timeout=timeout) as response:
        return json.loads(response.read())


def mujoco_add_belief_geoms(
    scene: Any,
    markers: dict[str, Any],
    *,
    max_particles: int = 200,
    particle_size: float = 0.01,
    cluster_scale: float = 2.0,
) -> int:
    """Draw belief markers into a MuJoCo ``mjvScene`` (e.g. ``viewer.user_scn`` or a ``Renderer``'s scene).

    Needs ``particle_positions``, ``cluster_positions`` (and ``true_position``) from ``state_to_position``.
    Particles: small translucent blue spheres (the heaviest ``max_particles``, radius scaled by weight).
    Clusters: translucent orange ellipsoids with semi-axes ``cluster_scale * cluster_position_stds`` (alpha
    follows the heft). True state: opaque green sphere.

    :return: Number of geoms added.
    """
    import mujoco
    import numpy as np

    def add(kind: int, size: Any, pos: Any, rgba: Any) -> None:
        if scene.ngeom >= scene.maxgeom:
            return
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            kind,
            np.asarray(size, dtype=np.float64),
            np.asarray(pos, dtype=np.float64),
            np.eye(3).reshape(-1),
            np.asarray(rgba, dtype=np.float32),
        )
        scene.ngeom += 1

    start = scene.ngeom
    weights = np.asarray(markers["weights"], dtype=np.float64)
    positions = np.asarray(markers["particle_positions"], dtype=np.float64)
    top = np.argsort(-weights)[:max_particles]
    relative = weights[top] / max(weights[top].max(), 1e-12)
    for index, w in zip(top, relative):
        add(mujoco.mjtGeom.mjGEOM_SPHERE, [particle_size * (0.5 + w), 0, 0], positions[index], [0.2, 0.4, 1.0, 0.4])
    hefts = np.asarray(markers["cluster_hefts"], dtype=np.float64)
    stds = np.asarray(markers["cluster_position_stds"], dtype=np.float64)
    for heft, std, position in zip(hefts, stds, np.asarray(markers["cluster_positions"], dtype=np.float64)):
        size = np.maximum(cluster_scale * std, particle_size)
        add(mujoco.mjtGeom.mjGEOM_ELLIPSOID, size, position, [1.0, 0.5, 0.0, 0.15 + 0.5 * heft])
    if "true_position" in markers:
        add(mujoco.mjtGeom.mjGEOM_SPHERE, [2 * particle_size, 0, 0], markers["true_position"], [0.1, 0.9, 0.1, 1.0])
    return scene.ngeom - start
