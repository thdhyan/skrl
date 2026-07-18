"""End-to-end smoke test: Belief-based DQN on peg-in-hole with stub backend.

Usage:
    python examples/contact_rich_uncertainty/peg_in_hole.py [--backend stub|mujoco|mjlab] [--timesteps N]

This wires together:
    StubBackend -> ParticleFilter -> BeliefDQN + MotionControllerSuite
"""
from __future__ import annotations

import argparse
import math
import torch
import gymnasium

from skrl.agents.torch.belief_dqn import BeliefDQN, BeliefDQN_CFG, ResNetDuelingQNetwork
from skrl.agents.torch.motion_controllers import build_controller_suite, MotionControllerSuite
from skrl.belief.backends.stub_backend import StubBackend
from skrl.belief.torch.particle_filter import ParticleFilter
from skrl.memories.torch import RandomMemory


# ------------------------------------------------------------------ config
N_JOINTS = 2
STATE_DIM = 10      # StubBackend default
NG = 2              # GMM modes
N_TARGETS = 2       # target poses per mode
NC = N_TARGETS * NG # = 4 controllers
BELIEF_DIM = NG + 2 * NG * STATE_DIM
NP = 100            # particles (paper uses 300; reduced for smoke test)
N_SUBSTEPS = 10     # inner steps (paper: 100; reduced for smoke test)


# ------------------------------------------------------------------ stub FK / Jacobian (2-link arm)
def _fk(q: torch.Tensor):
    l1, l2 = 1.0, 1.0
    x = l1 * torch.cos(q[0]) + l2 * torch.cos(q[0] + q[1])
    y = l1 * torch.sin(q[0]) + l2 * torch.sin(q[0] + q[1])
    pos = torch.tensor([x, y, 0.0])
    quat = torch.tensor([0.0, 0.0, 0.0, 1.0])  # identity (planar)
    return pos, quat


def _jacobian(q: torch.Tensor):
    l1, l2 = 1.0, 1.0
    J = torch.zeros(6, N_JOINTS)
    s1 = torch.sin(q[0])
    s12 = torch.sin(q[0] + q[1])
    c1 = torch.cos(q[0])
    c12 = torch.cos(q[0] + q[1])
    J[0, 0] = -l1 * s1 - l2 * s12
    J[0, 1] = -l2 * s12
    J[1, 0] = l1 * c1 + l2 * c12
    J[1, 1] = l2 * c12
    # rows 2-5 (z, rot) = 0 for planar arm
    return J


def make_target_templates():
    """4 approach targets around the object (front, back, left, right)."""
    targets = []
    for angle in [0.0, math.pi / 2, math.pi, 3 * math.pi / 2]:
        r = 0.3  # approach radius
        x, y = r * math.cos(angle), r * math.sin(angle)
        pose = torch.tensor([x, y, 0.0, 0.0, 0.0, 0.0, 1.0])
        targets.append(pose)
    return targets[:N_TARGETS]  # keep N_TARGETS


def make_belief_factory(backend: StubBackend, device: str):
    def factory():
        def motion_model(particles, control):
            return backend.step(particles, control, n_substeps=1)

        def measurement_model(particles, measurement):
            return backend.get_measurement(particles, sigma_enc=1.0)

        return ParticleFilter(
            num_particles=NP,
            state_dim=STATE_DIM,
            num_gmm_components=NG,
            motion_model_fn=motion_model,
            measurement_model_fn=measurement_model,
            device=device,
            resample_threshold=0.5,
        )
    return factory


def make_agent(belief_factory, device: str):
    obs_space = gymnasium.spaces.Box(low=-float("inf"), high=float("inf"),
                                     shape=(N_JOINTS,), dtype="float32")
    action_space = gymnasium.spaces.Discrete(NC)
    belief_obs_space = gymnasium.spaces.Box(low=-float("inf"), high=float("inf"),
                                            shape=(BELIEF_DIM,), dtype="float32")

    q_net = ResNetDuelingQNetwork(belief_obs_space, action_space, device,
                                  resnet_blocks=4, width=64, dueling=True)
    tgt_net = ResNetDuelingQNetwork(belief_obs_space, action_space, device,
                                    resnet_blocks=4, width=64, dueling=True)

    memory = RandomMemory(memory_size=10_000, num_envs=1, device=device)

    cfg = BeliefDQN_CFG(
        belief_factory=belief_factory,
        num_controllers=NC,
        gmm_components=NG,
        batch_size=64,
        discount_factor=0.98,
        learning_rate=1e-4,
        l2_reg=0.01,
        random_timesteps=200,
        learning_starts=200,
        update_interval=1,
        target_update_interval=10,
        exploration_scheduler=lambda t, T: max(0.05, 1.0 - t / T),
        n_substeps=N_SUBSTEPS,
    )
    return BeliefDQN(
        models={"q_network": q_net, "target_q_network": tgt_net},
        memory=memory,
        observation_space=obs_space,
        action_space=action_space,
        device=device,
        cfg=cfg,
    )


def run(timesteps: int = 1000, device: str = "cpu"):
    print(f"[peg_in_hole] backend=stub  device={device}  timesteps={timesteps}")

    backend = StubBackend(n_joints=N_JOINTS, device=device, seed=42)
    belief_factory = make_belief_factory(backend, device)

    agent = make_agent(belief_factory, device)
    agent.init()

    # build NC motion controllers
    controllers = build_controller_suite(
        gmm_means=torch.zeros(NG, STATE_DIM),
        target_templates=make_target_templates(),
        n_joints=N_JOINTS,
        fk_fn=_fk,
        jacobian_fn=_jacobian,
        Kp=5.0,
        device=device,
    )
    suite = MotionControllerSuite(controllers, n_substeps=N_SUBSTEPS)

    # goal: object x-position > 0.5
    GOAL_DIST = 0.5

    global_step = 0
    episode = 0

    while global_step < timesteps:
        # --- reset ---
        episode += 1
        obs = backend.reset(seed=episode)  # [n_joints]
        particles = torch.randn(NP, STATE_DIM, device=device) * 0.5
        agent.reset_belief(particles)
        obs_t = obs.unsqueeze(0)  # [1, n_joints]
        done = False
        ep_reward = 0.0
        ep_steps = 0

        while not done and global_step < timesteps:
            action, _ = agent.act(obs_t, None, timestep=global_step, timesteps=timesteps)
            a = action.item()

            # run selected controller for N_SUBSTEPS and step ground truth
            belief = agent._belief
            belief.compress()
            robot_state = backend.get_robot_state()
            ref = suite.act(a, robot_state, belief)

            # step ground truth environment
            next_obs = backend.step_ground_truth(ref, n_substeps=N_SUBSTEPS)
            next_obs_t = next_obs.unsqueeze(0)

            # constant negative reward (paper: r = -1)
            reward = -1.0
            obj_pose = backend.get_object_pose()
            dist = obj_pose[:2].norm().item()
            terminated = dist > GOAL_DIST

            reward_t = torch.tensor([[reward]], device=device)
            term_t = torch.tensor([[terminated]], device=device)
            trunc_t = torch.tensor([[False]], device=device)

            agent.record_transition(
                observations=obs_t,
                states=None,
                actions=action,
                rewards=reward_t,
                next_observations=next_obs_t,
                next_states=None,
                terminated=term_t,
                truncated=trunc_t,
                infos={},
                timestep=global_step,
                timesteps=timesteps,
            )
            agent.post_interaction(timestep=global_step, timesteps=timesteps)

            obs_t = next_obs_t
            ep_reward += reward
            ep_steps += 1
            global_step += 1
            done = terminated

        print(f"  ep={episode:4d}  steps={ep_steps:4d}  reward={ep_reward:.1f}  "
              f"total_steps={global_step}")

    print(f"[peg_in_hole] done  total_episodes={episode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="stub", choices=["stub", "mujoco", "mjlab"])
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.backend != "stub":
        raise NotImplementedError(
            f"Backend '{args.backend}' requires additional setup. "
            "See skrl/belief/backends/{args.backend}_backend.py."
        )

    run(timesteps=args.timesteps, device=args.device)
