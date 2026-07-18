# CartPole Action Space Analysis: DQN vs BeliefDQN vs PPO

## Problem Statement
Same environment (CartPole POMDP with hidden velocities). Same reward signal. Different agents see **different input spaces** — how does that affect action selection?

## Input Spaces

### Standard DQN
- **Input:** Raw observation `obs = [cart_pos, 0, pole_angle, 0]` (masked)
- **Input dim:** 4
- **Network:** `Linear(4) → ReLU → Linear → ReLU → Linear(2)` outputs Q-values
- **Challenge:** Cart velocity hidden (index 1), pole velocity hidden (index 3)
  - Must infer from history implicitly (no memory, no belief)
  - Single time-step input → struggles with POMDP

### BeliefDQN
- **Input:** Compressed belief `b̃ = [h₀, h₁, μ₀₀, σ₀₀, μ₀₁, σ₀₁, ..., h₁, μ₁₀, σ₁₀, ...]`
  - `h_i` = GMM weight (mixture component i)
  - `μ_ij` = mean of state dim j in component i
  - `σ_ij` = log std of state dim j in component i
  - NG=2 components, STATE_DIM=4 → **belief_dim = 2 + 2×2×4 = 18**
- **Input dim:** 18
- **Network:** `ResNetDuelingQNetwork(belief_space=[18], hidden=64)`
  - ResNet blocks smooth gradient flow through belief representation
  - Dueling head: V(b̃) + A(b̃, a) → Q(b̃, a)
- **Belief generation:**
  ```
  particles ← motion_model(particles, action)
  particles ← update(particles, obs) via Bayes
  b̃ ← compress(particles) via GMM fit
  ```
- **Advantage:** Belief summarizes full particle history + state uncertainty
  - Captures velocity estimates via particle spread
  - Can infer velocities from repeated observations

### PPO
- **Input:** Same as DQN — raw obs `[cart_pos, 0, pole_angle, 0]`
- **Input dim:** 4
- **Network:** Actor-Critic
  ```
  shared: Linear(4) → ReLU → Linear → ReLU
  actor: Linear(hidden) → Linear(2)  [logits for Discrete(2)]
  critic: Linear(hidden) → Linear(1)  [baseline]
  ```
- **Same problem as DQN:** No explicit state estimation
- **Advantage over DQN:** Policy gradient + baseline → better credit assignment
  - On-policy learning adapts faster to POMDP than off-policy

---

## Action Space: All Same

### Action Space Definition
All three agents: **Discrete(2)**
- Action 0: Push cart left
- Action 1: Push cart right

| Agent | Discrete(2)? | Output |
|-------|---|---|
| DQN | Yes | Q(obs, a) shape [batch=1, actions=2] |
| BeliefDQN | Yes | Q(b̃, a) shape [batch=1, actions=2] |
| PPO | Yes | logits shape [batch=1, actions=2], value shape [batch=1] |

**Why identical?** CartPole action space doesn't change. Difference is *what information* agent uses to decide which action.

---

## Key Difference: Input Representation

```
         ┌─ Raw obs [4] → DQN Q-network → Q-values [2]
         │
CartPole │
Obs ─────┤─ Raw obs [4] → PPO Actor → logits [2]
         │
         └─ Raw obs [4] → Belief PF → Compressed b̃ [18] → BeliefDQN Q-network → Q-values [2]
                          ↑ Motion model + Bayes update + GMM compression
```

---

## Does Belief Compression Work in Network Architecture?

### Test: Extract belief representation and verify compression.

```python
# In training loop after agent.post_interaction()
belief = agent._belief
gmm = belief.gmm

# Compressed belief components
if gmm is not None:
    hefts = gmm.hefts  # shape [2] — mixture weights
    means = gmm.means  # shape [2, 4] — mode means (cart_pos, cart_vel, pole_angle, pole_vel)
    log_covs = gmm.log_diag_covs  # shape [2, 4] — log std per dimension
    
    # Flatten to network input
    b_tilde = torch.cat([hefts, means.flatten(), log_covs.flatten()])  # [18]
    
    # Feed to Q-network
    q_vals = agent.q_network(b_tilde.unsqueeze(0))  # [1, 2]
    
    # Check: Does network output different Q-values for different beliefs?
    # If yes → compression preserves decision-relevant info
```

### What to measure:

1. **Belief entropy over training**
   ```
   H(b̃) = -Σ h_i * log(h_i) + Σ log|Σ_i|
   ```
   Should decrease as agent refines state estimate.

2. **Gradient flow through belief**
   - Hook Q-network input → measure gradient magnitude
   - If all zeros → compression fails (dead gradient)
   - If active → compression works, network learns from belief

3. **Compare Q-value changes w.r.t. belief changes**
   ```
   dQ/db̃ = ? 
   ```
   If ≠ 0 → network uses belief compression.

4. **Compare performance: DQN vs BeliefDQN on same CartPole**
   - DQN must infer velocities from obs alone
   - BeliefDQN has explicit velocity estimates
   - BeliefDQN should converge faster / higher reward

---

## Empirical Test: Does Compression Help?

### Hypothesis
- **DQN:** Learns implicit velocity detector in hidden layer. Slow, high variance.
- **BeliefDQN:** Belief provides pre-computed state estimate. Fast, stable learning.
- **PPO:** On-policy adaptation partially compensates for POMDP. Intermediate.

### Expected outcomes
```
DQN reward (last 100 eps):     ~100-150  (struggles, no velocity)
BeliefDQN reward (last 100 eps): ~200+   (has velocity estimates)
PPO reward (last 100 eps):     ~120-160  (better than DQN, worse than BeliefDQN)
```

### Run comparison (next step):
```bash
python benchmark_belief_dqn.py --timesteps 50000 --device cpu
# Monitor TensorBoard:
# tensorboard --logdir runs/
# Compare Reward/episode_reward across runs
```

---

## Network Architecture: Belief Compression

### BeliefDQN Q-network: ResNetDuelingQNetwork

```
Input: b̃ [18-dim compressed belief]
  ↓
ResNet block 1: Conv-BatchNorm-ReLU
  ↓ (residual connection)
ResNet block 2: Conv-BatchNorm-ReLU
  ↓
ResNet block 3: Conv-BatchNorm-ReLU
  ↓
ResNet block 4: Conv-BatchNorm-ReLU
  ↓
Dueling head:
  ├─→ V-stream: Linear → V(b̃)
  └─→ A-stream: Linear → A(b̃, a₀), A(b̃, a₁)
  ↓
Q(b̃, a) = V(b̃) + A(b̃, a) - mean_a[A(b̃, a)]
```

**Why ResNet for belief?**
- Belief [18] is low-dim but semantically rich
- ResNet blocks: non-linear transformations with skip connections
- Enables network to learn *which belief dimensions matter* for value
- Example: If cart velocity σ is high → uncertainty → different policy

**How compression is used:**
1. Belief encoder (agent) compresses particles → b̃ (18-dim)
2. Network learns: which b̃ dimensions → which actions
3. Gradient flows back: dQ/db̃ → used to update policy

**Compression works if:**
- b̃ captures state uncertainty (via σ_ij)
- b̃ captures state estimates (via μ_ij)
- Network learns different Q-values for different b̃
- Training reward improves faster than DQN

---

## Summary: Action Space

| Aspect | DQN | BeliefDQN | PPO |
|--------|-----|-----------|-----|
| **Action space** | Discrete(2) | Discrete(2) | Discrete(2) |
| **Input space** | obs [4] | belief [18] | obs [4] |
| **Compression** | None | GMM fit to particles | None |
| **Velocity info** | Implicit (learned) | Explicit (μ, σ for v_x, v_θ) | Implicit (learned) |
| **Expected reward** | ~100-150 | ~200+ | ~120-160 |

**Action space identical. Input representation is the key difference.**

To verify compression works: run 50k-step comparison, measure reward curves + belief entropy over time.
