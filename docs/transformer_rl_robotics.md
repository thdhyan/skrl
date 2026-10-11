# Transformer models in robot reinforcement learning: locomotion, manipulation and controllers

Research notes, 2026-10-11. Purpose: find where transformers help robot RL and decide what to try in this fork
(PrivilegedPPO history estimator, belief agents, G1 controllers). Papers marked *(snippet)* were seen only through
search summaries; check the paper before relying on details.

## 1. Where the transformer sits

| Role | What the transformer reads | Why it helps | Examples |
|---|---|---|---|
| History policy (in-context adaptation) | Past observations and actions | Infers terrain, dynamics and disturbances from context without weight updates. This is implicit online system identification | Radosavovic et al. 2024 (humanoid), DexFormer 2026 |
| History estimator / adaptation module | Proprioceptive history | Regresses the privileged latent (RMA/ROA extrinsics) | RMA uses a 1D CNN; a transformer is the natural swap. The 2026 bipedal mobile manipulator uses windowed transformer + GRU |
| Embodiment-structured policy | One token per joint/sensor, attention masked by the body graph | Inductive bias from morphology; better scaling than MLP | Body Transformer (BoT) |
| Multi-modal fusion | Proprioception + vision (+ touch) tokens | Cross-attention between modalities | LocoTransformer (vision-guided quadruped), Humanoid Transformer with Touch Dreaming *(snippet)* |
| Critic / Q-function | Per-action-dimension tokens | Autoregressive Q over discretized action dims; scales offline RL | Q-Transformer |
| Foundation policy + RL fine-tuning | Images, language, proprioception | Large pretrained VLA, RL used for fine-tuning or steering | RT-2, OpenVLA, π0 (from prior knowledge); ZPRL steers a pretrained policy through a latent *(snippet)* |

## 2. Locomotion

- **Real-world humanoid locomotion with RL** (Radosavovic et al., Science Robotics 2024; arXiv 2303.03381).
  - The controller is a causal transformer over the observation–action history that predicts the next action.
  - It is trained with model-free RL on an ensemble of randomized simulations and deployed zero-shot on a full-size humanoid.
  - It uses no state estimator, reference trajectories or gait library.
  - The history lets the policy adapt in context. This is the strongest evidence that a transformer can *be* the history encoder.
- **TERT: Sim-to-real quadruped locomotion via Terrain Transformer** (Lai et al., ICRA 2023; arXiv 2212.07740).
  - Two stages: offline pretraining from a privileged teacher, then online correction.
  - Beats baselines in return, energy and smoothness.
  - On hardware it crossed 9 terrains, including a sand pit and stairs down.
  - Stairs up were excluded because the policy has no elevation input.
- **Body Transformer** (Sferrazza et al., CoRL 2024; arXiv 2408.06316).
  - The robot is modelled as a graph of sensors and actuators, and masked attention follows that graph.
  - Beats a vanilla transformer and an MLP, in both imitation and RL, on task completion, scaling and compute efficiency. Shown on the Unitree A1.
- **LocoTransformer** (Yang et al., ICLR 2022): a cross-modal transformer fuses proprioception and depth for vision-guided quadruped locomotion.
- **History encoders that are not transformers, the baseline to beat:**
  - **RMA** (Kumar et al., RSS 2021; arXiv 2107.04034). A 1D CNN over 0.5 s of state–action history estimates the extrinsics latent produced by an encoder of the privileged environment parameters. It predicts how behaviour should change rather than the raw parameters, which is the key difference from classic system ID.
  - **A-RMA** for bipeds (arXiv 2205.15299) uses about 2 s of history.

## 3. Manipulation

- **Q-Transformer** (Chebotar et al., CoRL 2023; arXiv 2309.10150).
  - Offline RL. Each action dimension is discretized and gets its own Q-value token, decoded autoregressively.
  - Learns from both demonstrations and autonomous data, and beats prior offline RL and imitation on a real multi-task manipulation suite.
- **DexFormer** (arXiv 2602.08278, Feb 2026).
  - A history-conditioned transformer infers hand morphology and dynamics from temporal context.
  - Zero-shot transfer across Leap, Allegro and Rapid hands.
  - Whether it is trained with RL is not confirmed.
- **In-Context Iterative Policy Improvement for Dynamic Manipulation** (arXiv 2508.15021, 2025). In-context learning predicts corrections to a parametric policy from previous interactions, and works well with little data, both in simulation and on a real robot.
- **VLA foundation models** (RT-2, OpenVLA, π0, from prior knowledge): large transformers trained mostly by imitation. RL enters as fine-tuning or as latent steering (e.g. ZPRL).

## 4. Loco-manipulation and whole-body controllers

- **Deep Whole-Body Control** (Fu, Cheng, Pathak, CoRL 2022; arXiv 2210.10044).
  - One RL policy for the legs and arm of a legged manipulator.
  - **Regularized Online Adaptation (ROA)** trains the privileged encoder and the history adaptation module together, with a regularizer between them. This is the single-phase alternative to the RMA teacher–student scheme.
  - Advantage Mixing handles the coupled action space.
  - *This fork's PrivilegedPPO implements the ROA regularizer, adding a mask curriculum on top.*
- **Holistic whole-body loco-manipulation with a bipedal mobile manipulator** (arXiv 2609.18930, Sept 2026).
  - One RL controller maps 6-DoF end-effector targets to base and arm actions.
  - A temporal context estimator combines windowed transformer encoding, GRU memory and auxiliary dynamics prediction.
  - The same controller is driven by VR teleoperation, a diffusion policy and scripted trajectories.
- **ULTRA (Unitree G1)** *(snippet)*: a universal tracking policy is distilled into a controller that generalizes to goal-conditioned whole-body loco-manipulation from egocentric perception.
- **Humanoid Transformer with Touch Dreaming** (arXiv 2604.13015) *(snippet)*: an RL whole-body controller handles the lower body, and a transformer over vision, proprioception and touch handles manipulation.

## 5. What this means for this fork

Our own result so far (`examples/transformer_mujoco/README.md`): a per-dimension-token transformer trunk lost to an
MLP on InvertedPendulum, InvertedDoublePendulum and HalfCheetah at laptop budgets. The literature agrees: transformers pay
off when the tokens carry *structure* (time steps, body parts, modalities, action dimensions), not single scalars.

Concrete next steps, in order of expected value:

1. **Temporal transformer as the PrivilegedPPO estimator.**
   - Tokens are the H history steps `[o_{t-k}, a_{t-k}]`, and the output is ẑ.
   - Compare against the current MLP-over-stacked-history estimator using the per-group probe R² (velocities vs mass vs friction) logged by PrivilegedPPO.
   - This is the RMA/ROA adaptation module with Radosavovic-style context. It is also the most direct test of "online system identification".
2. **Body-graph tokens for the G1 controller.** One token per joint (q, q̇, last action), with BoT-style masked attention, for the HOVER/G1 whole-body tracking work.
3. **GMM-component tokens for the belief agents.** One token per component of b̃ (heft, mean, log-std). The policy then becomes permutation-invariant over clusters, which the flat b̃ is not.
4. **Q-Transformer-style controller selection for BeliefDQN.** Per-dimension autoregressive Q, if the discrete controller set grows.
5. **Loco-manipulation command interface.** A single controller with an end-effector-target interface (as in 2609.18930), so that teleop, diffusion and scripted planners can share one RL low-level policy.

## Sources

- [Real-World Humanoid Locomotion with Reinforcement Learning (arXiv 2303.03381)](https://arxiv.org/abs/2303.03381)
- [Sim-to-Real Transfer for Quadrupedal Locomotion via Terrain Transformer (arXiv 2212.07740)](https://arxiv.org/pdf/2212.07740)
- [Body Transformer: Leveraging Robot Embodiment for Policy Learning (arXiv 2408.06316)](https://arxiv.org/abs/2408.06316v1), [PMLR v270](https://proceedings.mlr.press/v270/sferrazza25a.html)
- [Learning Vision-Guided Quadrupedal Locomotion End-to-End with Cross-Modal Transformers (ICLR 2022)](https://iclr.cc/virtual/2022/poster/6919)
- [RMA: Rapid Motor Adaptation for Legged Robots (arXiv 2107.04034)](https://arxiv.org/pdf/2107.04034)
- [Adapting Rapid Motor Adaptation for Bipedal Robots (arXiv 2205.15299)](https://arxiv.org/pdf/2205.15299)
- [Deep Whole-Body Control (arXiv 2210.10044)](https://arxiv.org/pdf/2210.10044), [PMLR v205](https://proceedings.mlr.press/v205/fu23a.html)
- [Q-Transformer (PMLR v229)](https://proceedings.mlr.press/v229/chebotar23a.html)
- [DexFormer (arXiv 2602.08278)](https://arxiv.org/html/2602.08278)
- [In-Context Iterative Policy Improvement for Dynamic Manipulation (arXiv 2508.15021)](https://arxiv.org/pdf/2508.15021)
- [Learning Holistic Whole-Body Loco-Manipulation with a Bipedal Mobile Manipulator (arXiv 2609.18930)](https://arxiv.org/pdf/2609.18930)
- [Humanoid Transformer with Touch Dreaming (arXiv 2604.13015)](https://www.opentrain.ai/papers/learning-versatile-humanoid-manipulation-with-touch-dreaming--arxiv-2604.13015/)
- [Interpreting Control Latents for System Identification via Conditional Flow Matching (arXiv 2608.23887)](https://arxiv.org/pdf/2608.23887)
