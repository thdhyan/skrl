# BeliefPPO: how belief-based PPO updates the pipeline

`skrl/agents/torch/belief_ppo/` adds **BeliefPPO**, the on-policy counterpart to `BeliefDQN`
(Wirnshofer et al., RSS 2020). The agent owns a particle filter. Policy and value act only on the
compressed belief b̃ (a flattened NG-component GMM: `[weights | means | log-diag-covs]`,
`dim = NG + 2·NG·state_dim`), never on the raw observation. This is integration pattern **C**
in `HANDOFF_ISAACLAB_BELIEF.md`.

Diagrams (Graphviz sources plus PNG/SVG renders) are in `diagrams/`:

| file | shows |
|---|---|
| `01_step_pipeline` | one env step: prior → predict/update/resample → GMM compress → π, V → memory |
| `02_ppo_update` | `update()`: GAE bootstrapped with V(b̃_T) → clipped-surrogate epochs → Adam |
| `03_patterns` | where the filter lives: A wrapper, B env-side batched, C agent-owned |
| `04_smoke_curves` | CartPole-POMDP learning curves (see below) |

Slide deck: `BeliefPPO_pipeline.pptx` (built by `build_slides.py`), also in Google Slides as
[Belief-based PPO in skrl](https://docs.google.com/presentation/d/1hxhtqeWJjghpnop5HAuMqdpD_2vA05J6nQuaqO-eQgI/edit).

Re-render with `for f in diagrams/*.dot; do dot -Tpng -Gdpi=150 $f -o ${f%.dot}.png; done`.

## How it differs from stock PPO

| | stock `PPO` | `BeliefPPO` |
|---|---|---|
| network input | env observation | compressed belief b̃ |
| memory `observations` | raw obs | `beliefs` (b̃_t) |
| per step | act | act on b̃_t, then `record_transition` runs predict(a) → update(o_{t+1}) → compress → b̃_{t+1} |
| GAE bootstrap | V(next obs) | V(b̃_{t+1}), the filter state after the last transition |
| episode reset | none | `cfg.belief_prior_fn(o_0)` re-seeds the filter on the first `act()` after a done |
| envs | N | 1 (single filter; for N parallel envs use pattern B) |
| default nets | MLP | `ResNetCategoricalPolicy` / `ResNetValue` (paper §V-A trunk, shared with BeliefDQN) |

GAE, the clipped surrogate, value clipping, KL early stop, AMP and the LR scheduler use the same code
as stock PPO (`compute_gae` is imported from `skrl.agents.torch.ppo.ppo`).

## Tests

`tests/agents/torch/test_belief_ppo.py` has 8 tests: config validation, shapes, memory stores b̃,
`update()` changes the parameters, a loop with episode ends, the prior reset fires once per episode,
and `num_envs > 1` is rejected. Together with `tests/belief` and `test_belief_dqn.py`: **44 passed**.

## Smoke test (CartPole-v1 POMDP, cart vel + pole ang. vel hidden, 20k steps, seed 42, CPU)

```bash
cd examples/contact_rich_uncertainty
python cartpole_pomdp_belief_ppo_agent.py --timesteps 20000 --log-dir ../../runs/belief_ppo_smoke/C_BeliefPPO
python cartpole_pomdp_belief_ppo.py --headless --timesteps 20000 --log-dir ../../runs/belief_ppo_smoke/A_WrapperPPO
python cartpole_pomdp_belief_ppo.py --headless --timesteps 20000 --baseline --log-dir ../../runs/belief_ppo_smoke/Base_RawObsPPO
python ../../docs/belief_ppo/plot_smoke.py ../../runs/belief_ppo_smoke ../../docs/belief_ppo/diagrams/04_smoke_curves.png
```

| run | wall time | max return | mean of last 10 logs |
|---|---|---|---|
| C: BeliefPPO (agent-owned filter, ResNet 2×64) | 224 s | 500.0 | 238.8 |
| A: belief wrapper + stock PPO (MLP 64-64) | 135 s | 500.0 | 465.9 |
| baseline: stock PPO on raw masked obs | 54 s | 109.6 | 53.0 |

Both belief-based runs reach the 500 cap, and the raw-obs baseline does not get past about 110, so the belief
pipeline is doing its job. C is noisier than A over the last 2.5k steps. A and C differ in network (ResNet
against MLP) and in when the filter prior is seeded, and this is one seed, so the gap between them is not
established. The next step is multiple seeds, plus C with the same MLP as A.

Implementation note: `act()` reuses the b̃ that `record_transition()` just computed rather than compressing
again. A second GMM EM per step had made pattern C about 5× slower.
