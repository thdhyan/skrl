"""Belief markers in a MuJoCo scene: the agent side publishes, the simulator side pulls and draws.

Agent side: CartPole POMDP (velocities hidden) with the particle-filter belief of ``cartpole_pomdp_belief_ppo.py``.
Every step it publishes ``info["belief"]`` (particles, GMM clusters, statistics vs the true state) with a
:class:`~skrl.belief.torch.markers.BeliefMarkerServer`.

Simulator side: a MuJoCo cart-pole set to the true state pulls the markers over HTTP
(:func:`~skrl.belief.torch.markers.fetch_belief_markers`) and draws them with
:func:`~skrl.belief.torch.markers.mujoco_add_belief_geoms`: blue particles (pole-tip positions), orange GMM
ellipsoids (2 sigma), green true pole tip. The video has an overview and a close-up panel and a status header.

    python examples/contact_rich_uncertainty/belief_markers_mujoco_demo.py --out runs/belief_markers/demo.mp4

In Isaac Sim / Isaac Lab or another MuJoCo process, poll ``http://127.0.0.1:8765/belief`` the same way.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import gymnasium
import imageio
import mujoco

from cartpole_pomdp_belief_ppo import HIDDEN, LENGTH, BeliefObservationWrapper
from skrl.belief.torch.markers import BeliefMarkerServer, fetch_belief_markers, mujoco_add_belief_geoms
from skrl.envs.wrappers.torch.pomdp_wrapper import PomdpWrapper


XML = f"""
<mujoco>
  <visual><global offwidth="640" offheight="480"/><headlight ambient="0.5 0.5 0.5" diffuse="0.5 0.5 0.5"/></visual>
  <asset><texture name="sky" type="skybox" builtin="gradient" rgb1="0.8 0.85 0.9" rgb2="1 1 1" width="64" height="64"/></asset>
  <worldbody>
    <light pos="0 -2 3"/>
    <geom type="plane" size="3 1 0.1" rgba="0.9 0.9 0.9 1"/>
    <geom type="box" pos="0 0 0.02" size="2.4 0.02 0.01" rgba="0.3 0.3 0.3 1"/>
    <body name="cart" pos="0 0 0.05">
      <joint name="x" type="slide" axis="1 0 0"/>
      <geom type="box" size="0.15 0.08 0.05" rgba="0.2 0.2 0.2 1"/>
      <body name="pole">
        <joint name="theta" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 {2 * LENGTH}" size="0.02" rgba="0.6 0.4 0.2 1"/>
      </body>
    </body>
    <camera name="overview" pos="0 -4 1.2" xyaxes="1 0 0 0 0.3 1"/>
  </worldbody>
</mujoco>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="runs/belief_markers/demo.mp4")
    args = parser.parse_args()

    # agent side
    env = BeliefObservationWrapper(
        PomdpWrapper(gymnasium.make("CartPole-v1"), hidden_obs_indices=HIDDEN, obs_noise_std=0.02, seed=args.seed)
    )
    server = BeliefMarkerServer(port=args.port)
    url = f"http://127.0.0.1:{server.address[1]}/belief"
    rng = np.random.default_rng(args.seed)

    # simulator side
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=360, width=480)
    close = mujoco.MjvCamera()
    close.distance, close.elevation, close.azimuth = 1.8, -10.0, 90.0

    env.reset(seed=args.seed)
    frames = []
    for step in range(args.steps):
        state = env.unwrapped.state
        action = int(state[2] + 0.5 * state[3] > 0) if rng.random() > 0.2 else int(rng.integers(2))
        _, _, terminated, truncated, info = env.step(action)
        server.publish(info["belief"])

        markers = fetch_belief_markers(url)
        stats = markers["stats"]
        x, _, theta, _ = markers["true_state"]
        data.qpos[:] = [x, theta]
        mujoco.mj_forward(model, data)
        close.lookat[:] = [x, 0.0, 0.6]
        panels = []
        for camera in ("overview", close):
            renderer.update_scene(data, camera=camera)
            mujoco_add_belief_geoms(renderer.scene, markers, particle_size=0.02)
            panels.append(renderer.render())
        frame = np.concatenate(panels, axis=1)
        header = (
            f"step {step} | spread {stats['spread']:.3f} | clusters {stats['num_clusters']} "
            f"(eff {stats['effective_clusters']:.2f}) | cluster std {stats['cluster_std']:.3f} | "
            f"mean err {stats['mean_error']:.3f} | closest cluster err {stats['cluster_error_min']:.3f}"
        )
        frames.append(_with_header(frame, header, ["overview", "follow (pole tip)"]))
        if terminated or truncated:
            env.reset()
    server.close()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    imageio.mimsave(args.out, frames, fps=25)
    imageio.imwrite(os.path.splitext(args.out)[0] + "_frame.png", frames[len(frames) // 2])
    print(f"video: {args.out} ({len(frames)} frames)")


def _with_header(frame: np.ndarray, header: str, labels: list[str]) -> np.ndarray:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (frame.shape[1], frame.shape[0] + 24), "white")
    image.paste(Image.fromarray(frame), (0, 24))
    draw = ImageDraw.Draw(image)
    draw.text((6, 6), header, fill="black")
    width = frame.shape[1] // len(labels)
    for i, label in enumerate(labels):
        draw.rectangle([i * width + 4, 28, i * width + 8 + 7 * len(label), 44], fill="white")
        draw.text((i * width + 6, 30), label, fill="black")
    return np.asarray(image)


if __name__ == "__main__":
    main()
