[![pypi](https://img.shields.io/pypi/v/skrl)](https://pypi.org/project/skrl)
[<img src="https://img.shields.io/badge/%F0%9F%A4%97%20models-hugging%20face-F8D521">](https://huggingface.co/skrl)
![discussions](https://img.shields.io/github/discussions/Toni-SM/skrl)
<br>
[![license](https://img.shields.io/github/license/Toni-SM/skrl)](https://github.com/Toni-SM/skrl)
<span>&nbsp;&nbsp;&nbsp;&nbsp;</span>
[![docs](https://readthedocs.org/projects/skrl/badge/?version=latest)](https://skrl.readthedocs.io/en/latest/?badge=latest)
[![pre-commit](https://github.com/Toni-SM/skrl/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/Toni-SM/skrl/actions/workflows/pre-commit.yml)
[![pytest-torch](https://github.com/Toni-SM/skrl/actions/workflows/tests-torch.yml/badge.svg)](https://github.com/Toni-SM/skrl/actions/workflows/tests-torch.yml)
[![pytest-jax](https://github.com/Toni-SM/skrl/actions/workflows/tests-jax.yml/badge.svg)](https://github.com/Toni-SM/skrl/actions/workflows/tests-jax.yml)
[![pytest-warp](https://github.com/Toni-SM/skrl/actions/workflows/tests-warp.yml/badge.svg)](https://github.com/Toni-SM/skrl/actions/workflows/tests-warp.yml)

<br>

## This fork: belief, privileged-curriculum and transformer RL for robotics

Research fork of skrl 2.1.0. Upstream skrl is unchanged below this section. Work happens on
`feature/privileged-curriculum`, which also contains everything from `feature/belief-pomdp-manipulation`
and the cloud branch `claude/2d-multiagent-ppo-grid-n4ld8f`. Plans are in [`plans.md`](plans.md), the task
checklist is in [`tasks.md`](tasks.md).

| Capability | Code | Example | Result so far |
|---|---|---|---|
| Belief-based agents (particle filter → GMM belief b̃, Wirnshofer et al. RSS 2020) | `skrl/belief/`, `skrl/agents/torch/belief_dqn/`, `skrl/agents/torch/belief_ppo/` | `examples/contact_rich_uncertainty/` | CartPole POMDP: belief reaches 500, raw masked obs ~50–110 ([docs/belief_ppo](docs/belief_ppo/README.md)) |
| Privileged-observation masking curriculum (encoder z, history estimator ẑ, softmax mask schedule) | `skrl/agents/torch/privileged_ppo/`, `skrl/envs/wrappers/torch/privileged_history_wrapper.py` | `examples/privileged_curriculum/` | Masked (deployable) actor 493/500 on CartPole; reward-gated schedule (`RewardGatedMaskCurriculum`) and per-group probe R² on Pendulum / InvertedPendulum / HalfCheetah / Hopper: the history estimator recovers velocities, not dynamics ([docs/privileged_curriculum](docs/privileged_curriculum/README.md)) |
| Transformer policies instead of MLPs | `skrl/models/torch/transformer.py` | `examples/transformer_mujoco/` | MLP ≥ transformer on all 3 MuJoCo tasks at these budgets ([README](examples/transformer_mujoco/README.md)) |
| 2D pursuit-evasion self-play (MAPPO/IPPO, potential-field obs groups) | `skrl/envs/pursuit_evasion/`, `skrl/multi_agents/torch/selfplay/` | `examples/pursuit_evasion/` | [experiments/pomdp_and_obs_groups](experiments/pomdp_and_obs_groups/README.md) |
| POMDP wrapper | `skrl/envs/wrappers/torch/pomdp_wrapper.py` | — | multi-dim obs, negative indices, `info["full_obs"]` on reset and step |

Setup on the laptop (the `belief` conda env named in `CLAUDE.md` does not exist there, a uv `.venv` is used):

```bash
uv venv .venv && uv pip install -e . pytest gymnasium[mujoco] tensorboard matplotlib
.venv/bin/python -m pytest -q tests/belief tests/agents/torch/test_belief_dqn.py tests/agents/torch/test_belief_ppo.py \
    tests/agents/torch/test_privileged_ppo.py tests/models tests/envs/test_pomdp_wrapper.py \
    tests/envs/test_pursuit_evasion.py tests/multi_agents/torch/test_selfplay.py tests/utils/test_tensorboard_writer.py
# 99 passed (2026-10-11)
```

Slides on BeliefPPO: [Google Slides](https://docs.google.com/presentation/d/1hxhtqeWJjghpnop5HAuMqdpD_2vA05J6nQuaqO-eQgI/edit)
(source `docs/belief_ppo/build_slides.py`).

<br>
<p align="center">
  <a href="https://skrl.readthedocs.io">
  <img width="300rem" src="https://raw.githubusercontent.com/Toni-SM/skrl/main/docs/source/_static/data/logo-light-mode.png">
  </a>
</p>
<h2 align="center" style="border-bottom: 0 !important;">SKRL - Reinforcement Learning library</h2>
<br>

**Documentation:** <strong>https://skrl.readthedocs.io</strong>

**Description**: ``skrl`` is an open-source modular library for Reinforcement Learning written in Python
(implemented in [PyTorch](https://pytorch.org/), [JAX](https://jax.readthedocs.io) and [NVIDIA Warp](https://nvidia.github.io/warp/))
and designed with a focus on modularity, readability, simplicity, and transparency of algorithm implementation.
In addition to supporting
OpenAI [Gym](https://www.gymlibrary.dev),
Farama [Gymnasium](https://gymnasium.farama.org) and [PettingZoo](https://pettingzoo.farama.org),
[ManiSkill](https://maniskill.readthedocs.io/en/latest/index.html),
among other environment interfaces, it allows loading and configuring
NVIDIA [Isaac Lab](https://isaac-sim.github.io/IsaacLab/index.html) and
[MuJoCo Playground](https://playground.mujoco.org/)
environments, enabling agents' simultaneous training by scopes (subsets of environments among all available environments),
which may or may not share resources, in the same run.

<br>

### Refer to the documentation for details and examples: https://skrl.readthedocs.io

<br>

> **Note:** This project is under **active continuous development**. Please make sure you always have the latest version. Visit the [develop](https://github.com/Toni-SM/skrl/tree/develop) branch or its [documentation](https://skrl.readthedocs.io/en/develop) to access the latest updates to be released.

<br>

### Citing this library

To cite this library in publications, please use the following reference:

```bibtex
@article{serrano2023skrl,
  author  = {Antonio Serrano-Muñoz and Dimitrios Chrysostomou and Simon Bøgh and Nestor Arana-Arexolaleiba},
  title   = {skrl: Modular and Flexible Library for Reinforcement Learning},
  journal = {Journal of Machine Learning Research},
  year    = {2023},
  volume  = {24},
  number  = {254},
  pages   = {1--9},
  url     = {http://jmlr.org/papers/v24/23-0112.html}
}
```
