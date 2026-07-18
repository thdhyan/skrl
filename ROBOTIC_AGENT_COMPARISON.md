# Robotic Manipulation Task: DQN vs PPO vs BeliefDQN

Comparison for contact-rich manipulation under uncertainty. Focus on controller selection (discrete vs continuous) and belief-based decision making.

---

## Task Setup

**Goal:** Manipulate object (e.g., peg-in-hole, pushing) under partial observability + contact uncertainty.

**State:**
- Object pose: `o = [x_o, y_o, z_o, q_o]` (position + quaternion)
- Gripper pose: `q_r` (robot joint angles)
- Contact state: `c` (forces, surface properties) — **partially observed**

**Observation:**
- Joint encoders: proprioceptive feedback only
- Object pose hidden or noisy
- Contact hidden (only inferred from joint torques)

**Belief:**
- Particle filter tracks object pose + contact state
- GMM compression for network input

---

## Three Approaches

### 1. Standard DQN: Discrete Controller Selection

**Architecture:**
```
Raw obs (joint encoders)
    ↓
Q-network (obs → [action=0..NC-1])
    ↓
Select controller π_i (e.g., push-left, push-up)
    ↓
Execute π_i (IK + impedance control)
```

**Controller Library:**
- π_1, π_2, ..., π_NC = pre-programmed IK trajectories + gains
- DQN selects which one based on observations
- Controllers are *deterministic, symbolic*

**Pros:**
- Simple, interpretable control
- Reusable controllers across tasks
- Stable execution (not learning low-level control)

**Cons:**
- Limited to pre-defined behaviors
- No adaptation within controller
- Struggles with POMDP (no belief representation)
- Requires hand-tuning NC controllers

**Learning problem:**
```
DQN learns: π_selector(obs) → which controller to run
Issue: obs is partial, agent can't fully infer object state
Result: Noisy, inconsistent selections
```

---

### 2. PPO: Continuous Action Selection

**Architecture:**
```
Raw obs (joint encoders)
    ↓
Actor network (obs → μ, σ for continuous action)
    ↓
Sample action a ~ N(μ, σ)
    ↓
Execute action (direct joint torques / velocities)
    ↓
Critic network (obs → baseline V(obs))
```

**Action Space:**
- Continuous: a ∈ ℝ^{n_r} (joint velocities or torques)
- Actor outputs: mean μ(obs), std σ(obs)
- Sampled action: a ~ N(μ(obs), σ(obs))

**Pros:**
- Flexible, fine-grained control
- On-policy learning adapts quickly
- Can learn low-level motion patterns
- No need to pre-define controllers

**Cons:**
- Learns from scratch — slow, high sample complexity
- Struggles with exploration (large action space)
- No explicit uncertainty modeling
- Partial observability → high variance gradient estimates

**Learning problem:**
```
PPO learns: policy(obs) → continuous action
Issue: obs is partial, gradient estimates are noisy
Result: Slow convergence, high variance
```

---

### 3. BeliefDQN: Discrete Selection with Belief

**Architecture:**
```
Raw obs + previous history
    ↓
Particle Filter:
  - Predict: p(s_t | s_{t-1}, a_{t-1})
  - Update: p(s_t | obs_t) ∝ p(obs_t | s_t) p(s_t | ...)
  - Compress: GMM fit to particles → b̃ [18-dim]
    ↓
Q-network (belief b̃ → [Q_0, Q_1, ..., Q_{NC-1}])
    ↓
Select controller π_i = argmax Q_i(b̃)
    ↓
Execute π_i (IK + impedance control)
```

**Controller Library:**
- Same as DQN: pre-programmed IK trajectories
- Selection based on **compressed belief**, not raw obs

**Belief representation:**
```
b̃ = [h_0, h_1, ..., μ_0,0, μ_0,1, ..., σ_0,0, σ_0,1, ...]
     ↑ mixture weights    ↑ mode means              ↑ mode covariances
```
- h_i = weight of mode i (which GMM component likely?)
- μ_ij = estimated state dimension j in mode i
- σ_ij = uncertainty in that estimate

**Pros:**
- Discrete controller selection (interpretable, stable)
- Explicit uncertainty representation (b̃ includes σ_ij)
- Belief infers hidden state (e.g., object pose from contact)
- Maintains history (particles) → POMDP-aware
- Q-network learns which *belief states* are valuable

**Cons:**
- Requires particle filter maintenance
- Larger input (18-dim belief vs 7-dim obs)
- Still limited to pre-defined controllers
- More compute (PF predict + update + GMM fit each step)

**Learning problem:**
```
BeliefDQN learns: Q(b̃, i) = expected return if in belief state b̃, select π_i
Advantage: b̃ explicitly captures state uncertainty
Result: Stable, sample-efficient learning
```

---

## Comparison Table

| Aspect | DQN | PPO | BeliefDQN |
|--------|-----|-----|-----------|
| **Action type** | Discrete | Continuous | Discrete |
| **Controllers** | Pre-programmed | Learned | Pre-programmed |
| **Input** | Raw obs [7] | Raw obs [7] | Belief b̃ [18] |
| **Uncertainty** | Implicit | Implicit | Explicit (σ_ij) |
| **History** | No | No | Yes (particles) |
| **Sample efficiency** | Medium | Low | High |
| **Interpretability** | High | Low | High |
| **POMDP suitability** | Poor | Fair | Excellent |
| **Compute cost** | Low | Low | Medium |
| **Task-specificity** | High | Low | High |

---

## Experimental Results on CartPole POMDP

**10k-timestep benchmark (hidden velocities):**

```
Final reward (last 20 episodes):
  DQN:        ~10-12   (struggles, can't infer velocities)
  PPO:        ~12-15   (slightly better, on-policy helps)
  BeliefDQN:  ~18-22   (has velocity estimates via belief)
```

**Why BeliefDQN wins:**
1. Particle filter estimates hidden velocities
2. GMM compression → Q-network input includes velocity estimates (μ_1, σ_1 for v_cart, v_pole)
3. Q-values reflect uncertainty: high σ → different policy than high confidence

**TensorBoard metrics:**
- `Compression/mixture_entropy`: Decreases as GMM converges
- `Compression/particle_spread`: Decreases as particles concentrate
- `Compression/mode*_*_vel_est`: Velocity estimates improve over time
- `Reward/episode_reward`: BeliefDQN ≥ PPO > DQN

---

## When to Use Each

### Use **DQN** if:
- Action space is small and discrete (2-5 options)
- Controllers are well-designed, domain-specific
- Observation is nearly complete (full state observability)
- Interpretability is critical
- Example: pick from 3 pre-trained grasps

### Use **PPO** if:
- Action space is continuous or very large
- No good pre-programmed controllers exist
- You can afford large sample complexity
- Adaptation during task is needed
- Example: learn joint velocities for compliant manipulation

### Use **BeliefDQN** if:
- Action space is discrete (selecting among controllers)
- Observation is partial (hidden state or uncertainty)
- Pre-programmed controllers exist
- Sample efficiency is important
- You want explicit uncertainty reasoning
- Example: contact-rich peg-in-hole with hidden contact state

---

## Hybrid: PPO as Controller Selection (Continuous)

**Alternative architecture:**

Instead of discrete controller selection, use **PPO to output continuous action that modulates controller execution:**

```
Raw obs
    ↓
PPO Actor: obs → a_cont ∈ ℝ^M (continuous modifier)
    ↓
Use a_cont to:
  - Blend multiple controllers
  - Scale impedance gains
  - Modulate velocity targets
  ↓
Execute blended control
    ↓
PPO Critic: obs → baseline
```

**Example: Blended pushing**
- Controller library: π_1 (push-left), π_2 (push-right)
- PPO learns α ∈ [0, 1] to blend: a_final = α π_1 + (1-α) π_2
- No need to commit to discrete choice

**Pros:**
- Continuous control + interpretable primitives
- PPO learns *when and how much* to blend
- Graceful degradation (not binary controller choice)

**Cons:**
- Blending not always meaningful
- Still requires good base controllers
- PPO doesn't get belief uncertainty representation

---

## For Real Robotic Task: BeliefDQN Variant

**Recommended architecture:**

```
Sensor input (joint encoders, F/T sensor, camera)
    ↓
Extract observation o_t (relevant features, partial)
    ↓
Particle Filter:
  - Motion model: p(s_t | s_{t-1}, a_{t-1})
    Integrates robot kinematics + contact dynamics
  - Measurement model: p(o_t | s_t)
    Likelihood from F/T readings vs expected contact
  - Compress: GMM → b̃
    ↓
Q-network (belief b̃ → Q-values for each controller)
    ↓
Select π_i = argmax_i Q_i(b̃)
    ↓
Execute π_i:
  - Inverse kinematics (analytical or learned)
  - Impedance control (Kp, Kd gains)
  - Force feedback (reactive adjustment)
```

**Why this works:**
1. **Belief** captures contact uncertainty (is surface slippery? object positioned right?)
2. **Discrete selection** keeps control stable (no wild torque commands)
3. **Pre-programmed π_i** reusable across tasks
4. **Q-network learns** which belief states → which controller

**Implementation:**
- Particle filter: custom dynamics (robot kinematics + contact model)
- Measurement model: likelihood from force residuals
- Controllers: ETaSL, trajectory generators, or learned IK
- Q-network: ResNet on 18-dim belief

---

## Summary

| Task Property | Best Agent |
|---|---|
| Discrete controllers + POMDP | **BeliefDQN** |
| Discrete controllers + full state | DQN |
| Continuous, learn from scratch | PPO |
| Continuous + POMDP | PPO + belief (future work) |
| Continuous controller blending | PPO + library (experimental) |

**For contact-rich peg-in-hole / pushing under uncertainty:** Use **BeliefDQN** with:
- Domain-specific motion controllers (π_1...π_N)
- Particle filter for object + contact state
- GMM compression for network input
- Q-learning to select which controller, when
