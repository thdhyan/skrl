# Pursuit-evasion self-play (n pursuers vs. 1 evader)

A batched 2D grid pursuit-evasion game written natively in torch, potential-field observations, and a
generation-based self-play runner that trains both teams with skrl's MAPPO or IPPO.

| Component | Location |
|---|---|
| Environment, config, potential fields, observation groups | `skrl/envs/pursuit_evasion/` |
| skrl multi-agent wrapper (`wrap_env(env, wrapper="pursuit-evasion")`) | `skrl/envs/wrappers/torch/pursuit_evasion_envs.py` |
| Self-play runner, opponent pools / Elo, samplers, MLP models | `skrl/multi_agents/torch/selfplay/` |
| Example script | `examples/pursuit_evasion/torch_pursuit_evasion_selfplay.py` |
| Tests | `tests/envs/test_pursuit_evasion.py`, `tests/multi_agents/torch/test_selfplay.py` |

## Quick start

```bash
# a few seconds on CPU
python examples/pursuit_evasion/torch_pursuit_evasion_selfplay.py --generations 2 \
    --timesteps-per-generation 64 --num-envs 8 --eval-episodes 8 --device cpu

# a real run: 3 pursuers on a 12x12 grid with 10% obstacles, hedge sampling, alternating training
python examples/pursuit_evasion/torch_pursuit_evasion_selfplay.py --n-pursuers 3 --grid 12 12 \
    --obstacle-density 0.1 --obs-mode local_potential --sampler hedge --generations 20 \
    --timesteps-per-generation 3000 --checkpoint-dir runs/pe_hedge

# run the tests
pytest tests/envs/test_pursuit_evasion.py tests/multi_agents/torch/test_selfplay.py
```

Programmatic use:

```python
from skrl.envs.pursuit_evasion import PursuitEvasionEnv
from skrl.multi_agents.torch.selfplay import SelfPlayCfg, SelfPlayRunner

env = PursuitEvasionEnv(num_envs=64, height=10, width=10, num_pursuers=3, obs_mode="global_potential")
runner = SelfPlayRunner(env, SelfPlayCfg(generations=10, timesteps_per_generation=2000, sampler="adaptive_pbt"))
history = runner.run()  # one record per generation: train/eval win rates, Elo, opponent distribution, ...
```

## Environment

* `H x W` grid (`height`, `width`), optional static obstacles (`obstacles=[(r, c), ...]` and/or
  `obstacle_density` sampled once with `obstacle_seed`; the layout is shared by all sub-environments).
* Agents: `pursuer_0 ... pursuer_{n-1}` and `evader`. Teams are exposed as `env.teams`.
* Actions: `Discrete(5)` = stay / up / down / left / right. Moves into walls or obstacles leave the agent in place.
  All agents move simultaneously; pursuers may share a cell.
* Capture (termination): a pursuer within `capture_radius` of the evader (`capture_metric` = `"manhattan"` or
  `"chebyshev"`; 0 = same cell, 1 = adjacent), or a pursuer and the evader swapping cells.
  Truncation after `max_steps`.
* Zero-sum rewards: all pursuers share the team reward
  `R = capture_reward * captured - time_penalty + shaping_scale * (gamma * Phi(s') - Phi(s))`
  with the potential-based shaping `Phi(s) = -mean_i U_att(p_i)` (0 after capture); the evader receives `-R`.
* Vectorized over `num_envs` with automatic reset of finished sub-environments; `infos` carries `captured`, `done`,
  `episode_length` and episode statistics.
* Global state for MAPPO's centralized critic (`state()` / `state_space`): all positions, time fraction (and obstacle
  map if any).

### Potential fields

Distances are Euclidean in cell units, `D` is the grid diagonal, lower potential is better.

| Term | Formula | Used by |
|---|---|---|
| Attractive (conic) | `U_att(x; g) = k_att * d(x, g) / D` (0 at the goal) | pursuers (goal = evader) |
| Repulsive (Khatib, normalized) | `k * g(d)`, `g(d) = ((1/max(d, d_min) - 1/d0) / (1/d_min - 1/d0))^2` if `d < d0` else 0 | inter-pursuer (`k_pp`, `d0_pp`), pursuer-to-evader (`k_ep`, `d0_ep`), obstacles (`k_obs`, `d0_obs`) |
| Centre attraction | `k_center * d(x, centre) / D` | evader (discourages corner-hiding) |

* Pursuer `i`: `U_i = U_att(evader) + sum_{j != i} k_pp g_pp(d(x, p_j)) + U_obs`
* Evader: `U_e = sum_j k_ep g_ep(d(x, p_j)) + k_center d(x, c) / D + U_obs`

`env.potential_fields()` returns `total`, `attractive`, `repulsive` and `obstacle` tensors of shape
`(num_envs, num_agents, H, W)`. Observations normalize potentials to `[0, 1]` by `potential_clip`.

## Observation groups (`obs_mode`)

| Mode | Content (per agent, flat vector in `[-1, 1]`) | Size |
|---|---|---|
| `full_state` | absolute positions of all agents, positions relative to self, role one-hot, agent-id one-hot, time fraction, (+ obstacle map) | `5A + 3 (+ H*W)` |
| `global_potential` | whole map, channels: total potential, attractive, repulsive, obstacles, self position | `5 * H * W` |
| `local_potential` | egocentric `k x k` window (`k = 2 * local_radius + 1`; out-of-grid = max potential + obstacle) of total / attractive / repulsive / obstacle channels, own position, role | `4 k^2 + 4` |

`env.observation_image_shape` gives the `(C, H, W)` layout of the potential modes (for CNN policies).
New groups are added through the registry:

```python
from skrl.envs.pursuit_evasion import ObservationBuilder, register_observation

@register_observation("my_group")
class MyGroup(ObservationBuilder):
    requires_fields = False
    def size(self): return 2
    def build(self): return self.normalized_positions()  # (num_envs, num_agents, size)
```

## Self-play runner

Each team has a **population** of `population_size` learners (MAPPO or IPPO, MLP categorical policy + MLP value;
MAPPO's critic uses the global state) and an opponent pool of frozen snapshots (generation 0 = initial random
policies). The population defaults to 4 members for `adaptive_pbt` and 1 for every other sampler. Members are
fully independent: own weights, optimizer, learning rate, entropy coefficient, Elo rating and opponent sampler. They
train concurrently, each on its own contiguous slice of the team's sub-environments (`num_envs` is split evenly, so
use at least `population_size` sub-environments per learning team). For every generation:

1. **Train** every member for `timesteps_per_generation` steps. Each sub-environment plays an opponent snapshot drawn
   by the member's sampler from the other team's pool; a new opponent is drawn per sub-environment whenever its
   episode ends. Outcomes update the pool's statistics and the member's (adaptive) sampler.
2. **Evaluate** every member (`eval_episodes` per opponent) against the same opponents -- the most recent snapshots
   and the best-rated snapshot of the opposing pool (`eval_max_opponents`) -- and update zero-sum Elo ratings.
3. **Snapshot** the members into the team's pool (`population_snapshots`: `"all"`, or only the `"best"`), in
   increasing order of evaluation score, so the generation's strongest member is the latest pool entry (what the
   `last` sampler picks).
4. **PBT** (optional, default on for `adaptive_pbt`):
   * population: **truncation selection**. Members are ranked by mean evaluation score (comparable: same opponents,
     same episodes). Each member in the bottom `pbt_truncation` fraction (default 25%, at least one) that scores more
     than `pbt_margin` below a random member of the top fraction **exploits** it -- copying its policy/value weights,
     optimizer state, learning rate, entropy coefficient, rating and sampler statistics -- then **explores** by
     multiplying its learning rate and entropy coefficient by a random factor in `pbt_perturb_factors`. Members 1..P-1
     start from log-uniformly perturbed hyper-parameters (`pbt_init_range`, default x0.5..x2) so the population is
     diverse from the start.
   * single learner (`population_size=1`): the best-rated snapshot of the learner's own pool is evaluated against the
     same opponents; if the new snapshot scores more than `pbt_margin` worse, the learner reloads it (exploit) and
     perturbs its hyper-parameters (explore).

The generation record of a team reports training statistics over the whole population and the evaluation of its
best member; with a population it also lists every member (`members`: score, rating, hyper-parameters, opponent
distribution, PBT event) and the `best_member`. `runner.learners[team]` is the current best member's agent.

Training schemes (`training_scheme`): `alternating` (one team learns per generation on all sub-environments,
starting with `first_learner`) or `simultaneous` (both teams learn every generation, each on half of the
sub-environments). `learning_teams` can freeze a team entirely. With `checkpoint_dir` each snapshot is stored as
`{team}_gen{g}.pt` (`{team}_gen{g}_member{i}.pt` with a population) together with `history.json`.

## Opponent samplers (`sampler`)

| Name | Distribution over the opponent pool |
|---|---|
| `last` | (a) only the latest generation (naive self-play) |
| `best` | (b) only the highest Elo-rated generation (latest on ties) |
| `uniform` | (c) uniform over all generations (fictitious self-play) |
| `hedge` | (d) Hedge / exponential weights: `p_i = (1 - gamma) softmax(eta * G_i) + gamma / K`, where `G_i` accumulates the importance-weighted (Exp3) loss the learner suffered against snapshot `i`. Opponents that beat the learner are upweighted, mastered ones fade out; new snapshots start optimistically at `max G`. `top_k` restricts the support to the `k` best-rated generations. |
| `adaptive_pbt` | (e) adaptive PFSP: weight `f(x_i)` of the learner's estimated win rate `x_i` against each snapshot, `"hard"` `(1 - x)^p` or `"variance"` `x (1 - x)`, mixed with `epsilon` uniform exploration. The win-rate estimates are kept per sampler, i.e. per population member. After each generation it adapts: learner win rate above `upper_win_rate` -> hard weighting with a sharper exponent; below `lower_win_rate` -> variance weighting (curriculum of even match-ups). Selecting it enables a population of 4 concurrent learners per team with population-based training (see above). |

Sampler arguments are passed with `sampler_kwargs` (CLI: `--sampler-kwargs '{"eta": 0.05, "top_k": 3}'`). New
strategies are registered with `@register_sampler("name")` on an `OpponentSampler` subclass implementing
`probabilities(pool)` (and optionally `update(...)` / `on_generation_end(...)`).

## Limitations

* Distances in the potential fields are Euclidean, not geodesic: with many obstacles the attractive field can have
  local minima (agents learn to get around them; the fields are observations/shaping, not a controller).
* The obstacle layout is static and shared by all sub-environments.
* Elo ratings in an asymmetric game mix two populations; use them to rank snapshots *within* a team.
* Population members share the vectorized environment: P members each get `num_envs / P` sub-environments, so a
  larger population means fewer samples per member and generation (raise `num_envs` accordingly).
* Every member plays the evaluation opponents, so evaluation cost grows linearly with the population size, and each
  of those matches moves the opponents' Elo ratings.
* Frozen opponents run one forward pass per distinct sampled snapshot per step; very large pools with many
  distinct opponents in flight get slower (bound it with `pool_max_size` or a focused sampler).
