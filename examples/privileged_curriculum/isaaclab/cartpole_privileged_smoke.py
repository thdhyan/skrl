"""Isaac Lab smoke run of PrivilegedPPO from the YAML config through the skrl Runner, with a debug video.

    # on zz-bw (see zzbw_smoke.sh), never on the laptop:
    ssh zz-bw SKRL_GPU=1 ~/Projects/skrl/examples/privileged_curriculum/isaaclab/zzbw_smoke.sh run

Task: Isaac Lab's direct Cartpole with the joint velocities hidden from the policy (it observes the cart and pole
positions only). The velocities are the privileged values: the encoder reads them, the history estimator must
infer them from the position history. The agent config is ``skrl_privileged_ppo_cfg.yaml`` resolved from the gym
registry (``skrl_privileged_ppo_cfg_entry_point``) as Isaac Lab's ``train.py --agent`` does, and the training
follows Isaac Lab's skrl train script (``Runner(SkrlVecEnvWrapper(env), agent_cfg).run()``).

Debug video (``<out>/videos``): overview, top-down and cart follow camera of env 0, a status header (step, mask
probability, whether env 0's privileged values are masked, true velocities, ||z - ẑ||), transparent red slabs at
the cart limits (``max_cart_pos``) and a sphere above the cart, green when the policy sees the privileged values
and orange when it acts on the history estimate. Needs the pe repo's ``isaaclab_pe/common/debug_video.py`` (path in
``$PE_DEBUG_VIDEO``). Run it on a server (zz-bw, dl), not on the laptop: see ``zzbw_smoke.sh``.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

HERE = Path(__file__).resolve().parent
PE_DEBUG_VIDEO = os.environ.get(
    "PE_DEBUG_VIDEO", str(Path.home() / "Projects/ebasa/pe/isaaclab_pe/common/debug_video.py"))
spec = importlib.util.spec_from_file_location("debug_video", PE_DEBUG_VIDEO)
debug_video = importlib.util.module_from_spec(spec)
spec.loader.exec_module(debug_video)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--timesteps", type=int, default=3000)
parser.add_argument("--strategy", default=None, help="override agent.strategy of the YAML")
parser.add_argument("--estimator", default=None, help="override models.policy.estimator (mlp/conv1d/lstm/transformer)")
parser.add_argument("--out", default=str(HERE.parents[2] / "runs" / "isaaclab_cartpole_privileged"))
debug_video.add_args(parser, interval=0, length=250, stride=2)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True  # Isaac Lab 3.0 runs without a window unless --visualizer is given
debug_video.prepare_app(args)
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_physx.physics import PhysxCfg  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg  # noqa: E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.core.cartpole.cartpole_direct_env import CartpoleEnv  # noqa: E402
from isaaclab_tasks.core.cartpole.cartpole_direct_env_cfg import CartpoleEnvCfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

from skrl.utils.runner.torch import Runner  # noqa: E402


class CartpoleHiddenVelocityEnv(CartpoleEnv):
    """Direct Cartpole whose policy observes the joint positions only; ``privileged()`` = the joint velocities."""

    debug_wrapper = None  # PrivilegedHistoryVecWrapper, set by the script for the debug markers and status
    debug_agent = None

    def _get_observations(self) -> dict:
        return {"policy": super()._get_observations()["policy"][:, :2]}

    def privileged(self) -> torch.Tensor:
        return torch.stack([self.joint_vel[:, self._cart_dof_idx[0]], self.joint_vel[:, self._pole_dof_idx[0]]], -1)

    # --- debug video -------------------------------------------------------------------------------------------

    def _cart_world(self, k: int) -> torch.Tensor:
        if not hasattr(self, "_cart_body"):
            self._cart_body = self.cartpole.find_bodies("cart")[0][0]
        return self.cartpole.data.body_link_pos_w.torch[k, self._cart_body]

    def debug_views(self):
        k = self.cfg.debug_envs[0]
        o, cart = self.scene.env_origins[k], self._cart_world(k)
        rail = getattr(self, "_rail", torch.tensor([1.0, 0.0], device=self.device))
        side = torch.tensor([rail[1], -rail[0], 0.0], device=self.device)  # beside the rail, looking across it
        at = (o + torch.tensor([0.0, 0.0, 2.0], device=self.device)).tolist()
        return [
            ("overview", (o + torch.tensor([-4.0, -7.0, 4.5], device=self.device)).tolist(), at),
            ("top down", (o + torch.tensor([0.0, -0.01, 11.0], device=self.device)).tolist(), at),
            ("follow cart", (cart + 3.0 * side + torch.tensor([0.0, 0.0, 0.6], device=self.device)).tolist(),
             (cart + torch.tensor([0.0, 0.0, 0.4], device=self.device)).tolist()),
        ]

    def _masked(self, k: int) -> bool | None:
        w = self.debug_wrapper
        return None if w is None or w._mask is None else bool(w._mask[k, 0] > 0.5)

    @torch.no_grad()
    def debug_status(self) -> str:
        k = self.cfg.debug_envs[0]
        w, vel = self.debug_wrapper, self.privileged()[k].tolist()
        text = f"true vel cart {vel[0]:+.2f} pole {vel[1]:+.2f}"
        if w is not None and w._curriculum is not None:
            text = f"p_mask {w._curriculum.mask_probability():.2f}  env{k} " + \
                ("MASKED (acts on history estimate z_hat)" if self._masked(k) else "visible (acts on z)") + "  " + text
        if w is not None and self.debug_agent is not None:
            z, z_hat = self.debug_agent.policy.trunk.latents(w._packed[k:k + 1])
            text += f"  |z - z_hat| {float((z - z_hat).norm()):.2f}"
        return text

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        """Red slabs: cart position limits. Sphere above the cart: green = privileged visible, orange = masked."""
        if debug_vis and not hasattr(self, "_dbg"):
            def mat(c, a):
                return sim_utils.PreviewSurfaceCfg(diffuse_color=c, opacity=a)

            self._dbg = VisualizationMarkers(VisualizationMarkersCfg(prim_path="/Visuals/privileged_debug", markers={
                "limit": sim_utils.CuboidCfg(size=(1.0, 1.0, 3.0), visual_material=mat((1.0, 0.1, 0.1), 0.35)),
                "visible": sim_utils.SphereCfg(radius=0.15, visual_material=mat((0.1, 0.9, 0.2), 0.6)),
                "masked": sim_utils.SphereCfg(radius=0.15, visual_material=mat((1.0, 0.55, 0.0), 0.6)),
            }))
        if hasattr(self, "_dbg"):
            self._dbg.set_visibility(debug_vis)

    def _debug_vis_callback(self, event) -> None:
        if not hasattr(self, "_dbg"):
            return
        k = self.cfg.debug_envs[0]
        o, cart = self.scene.env_origins[k], self._cart_world(k)
        q = float(self.joint_pos[k, self._cart_dof_idx[0]])
        if abs(q) > 0.2:  # rail direction in the world, from the cart displacement along the slider
            self._rail = (cart - o)[:2] / q
            self._rail = self._rail / self._rail.norm()
        rail = getattr(self, "_rail", torch.tensor([1.0, 0.0], device=self.device))
        axis = torch.cat([rail, torch.zeros(1, device=self.device)])
        centre = torch.cat([o[:2], cart[2:]])
        limits = torch.stack([centre + s * self.cfg.max_cart_pos * axis for s in (-1.0, 1.0)])
        thin = torch.cat([0.05 + 0.95 * rail.flip(0).abs(), torch.ones(1, device=self.device)])  # thin along the rail
        sphere = (cart + torch.tensor([0.0, 0.0, 1.4], device=self.device))[None]
        self._dbg.visualize(translations=torch.cat([limits, sphere]),
                            scales=torch.stack([thin, thin, torch.ones(3, device=self.device)]),
                            marker_indices=torch.tensor([0, 0, 2 if self._masked(k) else 1], device=self.device))


def privileged_velocities(env) -> torch.Tensor:
    """``privileged_fn`` of the YAML: the skrl-wrapped env's joint velocities."""
    return env.unwrapped.privileged()


def main() -> None:
    out = Path(args.out)
    task = "Isaac-Cartpole-Direct-HiddenVelocity-Privileged-v0"
    gym.register(id=task, entry_point=f"{__name__}:CartpoleHiddenVelocityEnv", disable_env_checker=True, kwargs={
        "env_cfg_entry_point": f"{__name__}:CartpoleEnvCfg",
        "skrl_privileged_ppo_cfg_entry_point": str(HERE / "skrl_privileged_ppo_cfg.yaml"),
    })
    agent_cfg = load_cfg_from_registry(task, "skrl_privileged_ppo_cfg_entry_point")  # what train.py --agent does
    agent_cfg["privileged_history"]["privileged_fn"] = privileged_velocities
    agent_cfg["agent"]["experiment"].update(directory=str(out), experiment_name="train")
    agent_cfg["trainer"].update(timesteps=args.timesteps, close_environment_at_exit=False)
    if args.strategy:
        agent_cfg["agent"]["strategy"] = args.strategy
    if args.estimator:
        agent_cfg["models"]["policy"]["estimator"] = args.estimator

    env_cfg = CartpoleEnvCfg()
    env_cfg.observation_space = 2
    env_cfg.sim.physics = PhysxCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.debug_envs = (args.video_env,)
    # clips at the start and at the end of training
    args.video_interval = max(args.timesteps - args.video_length, 0)
    debug_video.configure(env_cfg, out, args, eye=(-4.0, -7.0, 4.5), lookat=(0.0, 0.0, 2.0))
    env = CartpoleHiddenVelocityEnv(env_cfg)
    debug_video.attach(env, out, args, prefix=f"privileged_ppo_{agent_cfg['agent'].get('strategy')}")

    runner = Runner(SkrlVecEnvWrapper(env), agent_cfg)
    env.debug_wrapper, env.debug_agent = runner.trainer.env, runner.agent
    trunk = runner.agent.policy.trunk
    print(f"[smoke] agent {type(runner.agent).__name__}, strategy {agent_cfg['agent'].get('strategy')}, "
          f"estimator {type(trunk.estimator).__name__}, sysid head {trunk.sysid is not None}, "
          f"layout {runner.trainer.env.layout}", flush=True)
    runner.run()
    runner.agent.write_checkpoint(timestep=args.timesteps, timesteps=args.timesteps)
    for rec in env.video_recorders:
        rec.close()
    print(f"[smoke] done, final mask probability {runner.agent.cfg.curriculum.mask_probability():.2f}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
