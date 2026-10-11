import pytest

import json
import os

import torch

from skrl.envs.pursuit_evasion import PursuitEvasionEnv
from skrl.multi_agents.torch.selfplay import (
    SAMPLERS,
    OpponentPool,
    PolicySnapshot,
    SelfPlayCfg,
    SelfPlayRunner,
    elo_update,
    make_sampler,
)

SAMPLER_NAMES = ["last", "best", "uniform", "hedge", "adaptive_pbt"]


def make_pool(ratings=(1000.0, 1100.0, 950.0, 1050.0)) -> OpponentPool:
    pool = OpponentPool("evader", ["evader"])
    for generation, rating in enumerate(ratings):
        pool.add(PolicySnapshot(team="evader", generation=generation, policies={"evader": {}}, rating=rating))
    return pool


def test_registry_contains_all_strategies():
    assert set(SAMPLER_NAMES) <= set(SAMPLERS)
    with pytest.raises(KeyError):
        make_sampler("does_not_exist")


@pytest.mark.parametrize("name", SAMPLER_NAMES)
def test_sampler_distribution_and_indices(name):
    pool = make_pool()
    sampler = make_sampler(name)
    probabilities = sampler.probabilities(pool)
    assert probabilities.shape == (len(pool),)
    assert torch.isclose(probabilities.sum(), torch.tensor(1.0, dtype=torch.float64))
    assert (probabilities >= 0).all()
    indices = sampler.sample(pool, 256, torch.Generator().manual_seed(0))
    assert indices.dtype == torch.long and indices.shape == (256,)
    assert indices.min() >= 0 and indices.max() < len(pool)
    assert (probabilities[indices] > 0).all()
    # learning from outcomes keeps a valid distribution
    scores = torch.rand(256)
    pool.record(indices, scores)
    sampler.update(pool, indices, scores)
    sampler.on_generation_end(pool, 0.9)
    probabilities = sampler.probabilities(pool)
    assert torch.isclose(probabilities.sum(), torch.tensor(1.0, dtype=torch.float64))
    # a pool with a single snapshot always yields it
    single = make_pool((1000.0,))
    assert (make_sampler(name).sample(single, 10) == 0).all()


def test_last_best_uniform():
    pool = make_pool()
    assert make_sampler("last").probabilities(pool).argmax() == 3
    assert make_sampler("best").probabilities(pool).argmax() == 1
    assert torch.allclose(make_sampler("uniform").probabilities(pool), torch.full((4,), 0.25, dtype=torch.float64))


def test_hedge_upweights_opponents_beating_the_learner():
    pool = make_pool()
    sampler = make_sampler("hedge", eta=0.5, gamma=0.0)
    assert torch.allclose(sampler.probabilities(pool), torch.full((4,), 0.25, dtype=torch.float64))
    for _ in range(20):
        indices = sampler.sample(pool, 64, torch.Generator().manual_seed(1))
        scores = (indices != 2).float()  # the learner only loses against snapshot 2
        sampler.update(pool, indices, scores)
    probabilities = sampler.probabilities(pool)
    assert probabilities.argmax() == 2 and probabilities[2] > 0.5
    # top-k restricts the support to the best-rated snapshots
    top = make_sampler("hedge", top_k=2).probabilities(pool)
    assert set(top.nonzero().view(-1).tolist()) == {1, 3}


def test_adaptive_pbt_prioritizes_hard_opponents_and_adapts():
    pool = make_pool()
    sampler = make_sampler("adaptive_pbt", epsilon=0.0)
    sampler.update(pool, torch.tensor([0, 1, 2, 3] * 10), torch.tensor([1.0, 0.0, 1.0, 0.5] * 10))
    # the statistics belong to the sampler (one per population member), not to the shared pool
    assert torch.allclose(make_sampler("adaptive_pbt").probabilities(pool), torch.full((4,), 0.25, dtype=torch.float64))
    probabilities = sampler.probabilities(pool)
    assert probabilities.argmax() == 1  # the opponent the learner never beats
    exponent = sampler.exponent
    sampler.on_generation_end(pool, 0.9)
    assert sampler.weighting == "hard" and sampler.exponent > exponent
    sampler.on_generation_end(pool, 0.1)
    assert sampler.weighting == "variance"
    assert sampler.probabilities(pool).argmax() == 3  # evenly matched opponent


def test_elo_update_is_zero_sum():
    a, b = elo_update(1000.0, 1000.0, 1.0, k=32)
    assert a == 1016.0 and b == 984.0


def test_pool_eviction_keeps_best():
    pool = OpponentPool("evader", ["evader"], max_size=3)
    for generation, rating in enumerate([1200.0, 900.0, 950.0, 1000.0]):
        pool.add(PolicySnapshot(team="evader", generation=generation, policies={"evader": {}}, rating=rating))
    assert [entry.generation for entry in pool] == [0, 2, 3]


@pytest.mark.parametrize(
    "scheme,sampler,algorithm,obs_mode",
    [
        ("alternating", "hedge", "mappo", "local_potential"),
        ("simultaneous", "adaptive_pbt", "ippo", "global_potential"),
        ("alternating", "uniform", "mappo", "full_state"),
    ],
)
def test_selfplay_smoke(tmp_path, scheme, sampler, algorithm, obs_mode):
    env = PursuitEvasionEnv(
        num_envs=4, height=6, width=6, num_pursuers=2, max_steps=12, obs_mode=obs_mode, device="cpu", seed=0
    )
    cfg = SelfPlayCfg(
        generations=2,
        timesteps_per_generation=16,
        algorithm=algorithm,
        sampler=sampler,
        training_scheme=scheme,
        eval_episodes=4,
        hidden_sizes=(16,),
        agent_cfg={"rollouts": 8, "learning_epochs": 1, "mini_batches": 1},
        checkpoint_dir=str(tmp_path),
        seed=0,
        verbose=False,
    )
    runner = SelfPlayRunner(env, cfg)
    history = runner.run()

    assert len(history) == 2
    if scheme == "alternating":
        assert [list(record["teams"]) for record in history] == [["pursuers"], ["evader"]]
        assert len(runner.pools["pursuers"]) == len(runner.pools["evader"]) == 2
    else:
        # adaptive_pbt: population of 2 members per team (capped by the 2 sub-environments of each team)
        assert all(set(record["teams"]) == {"pursuers", "evader"} for record in history)
        assert runner.population_size == 2
        assert len(runner.pools["pursuers"]) == len(runner.pools["evader"]) == 1 + 2 * 2
    for record in history:
        for info in record["teams"].values():
            assert info["eval_win_rate"] and all(0.0 <= v <= 1.0 for v in info["eval_win_rate"].values())
            assert abs(sum(info["opponent_distribution"]) - 1.0) < 1e-2
            for member in info.get("members", []):
                assert abs(sum(member["opponent_distribution"]) - 1.0) < 1e-2
    # the learner's weights actually changed and snapshots are frozen copies
    learner = next(iter(runner.learners.values()))
    team = next(iter(runner.learners))
    uid = runner.teams[team][0]
    initial = runner.pools[team][0].policies[uid]
    current = learner.policies[uid].state_dict()
    assert any(not torch.equal(initial[k], current[k].cpu()) for k in initial)
    # artifacts
    assert os.path.exists(tmp_path / "history.json")
    assert len(json.load(open(tmp_path / "history.json"))) == 2
    assert any(name.endswith(".pt") for name in os.listdir(tmp_path))


def make_population_runner(**overrides) -> SelfPlayRunner:
    env = PursuitEvasionEnv(num_envs=8, height=6, width=6, num_pursuers=2, max_steps=12, device="cpu", seed=0)
    cfg = dict(
        generations=2,
        timesteps_per_generation=16,
        sampler="adaptive_pbt",
        population_size=4,
        eval_episodes=4,
        hidden_sizes=(16,),
        agent_cfg={"rollouts": 8, "learning_epochs": 1, "mini_batches": 1},
        seed=0,
        verbose=False,
    )
    cfg.update(overrides)
    return SelfPlayRunner(env, SelfPlayCfg(**cfg))


def test_population_members_are_independent_learners():
    runner = make_population_runner()
    for team, members in runner.members.items():
        assert len(members) == 4
        # disjoint, contiguous sub-environment slices covering the team's span
        assert [(m.start, m.stop) for m in members] == [(0, 2), (2, 4), (4, 6), (6, 8)]
        learners = [m.learner for m in members]
        assert len({id(learner) for learner in learners}) == 4
        assert len({id(m.sampler) for m in members}) == 4
        uid = runner.teams[team][0]
        assert len({id(learner.policies[uid]) for learner in learners}) == 4
        # diversified initial hyper-parameters (member 0 keeps the configured ones)
        rates = [runner._hyperparameters(m)["learning_rate"] for m in members]
        assert rates[0] == runner.agent_cfg["learning_rate"] and len(set(rates)) == 4

    history = runner.run()
    record = history[0]["teams"]["pursuers"]
    assert len(record["members"]) == 4 and record["best_member"] in range(4)
    assert sum(member["train_games"] for member in record["members"]) == record["train_games"]
    best = max(record["members"], key=lambda m: m["eval_score"])
    assert record["eval_win_rate"] == best["eval_win_rate"]
    # truncation selection ran on the bottom member (25% of 4)
    assert sum("pbt" in member for member in record["members"]) == 1
    # every member is snapshotted, the generation's strongest last
    pool = runner.pools["pursuers"]
    assert len(pool) == 1 + 4
    assert pool[-1].metadata["member"] == record["best_member"]
    assert runner.learners["pursuers"] is runner.members["pursuers"][record["best_member"]].learner


def test_population_exploit_copies_weights_hyperparameters_and_sampler():
    runner = make_population_runner(pbt_margin=0.0)
    source, target = runner.members["pursuers"][0], runner.members["pursuers"][3]
    source.rating = 1234.0
    source.sampler.update(runner.pools["evader"], torch.tensor([0]), torch.tensor([1.0]))
    scores = {0: 1.0, 1: 0.6, 2: 0.4, 3: 0.0}
    events = runner._pbt_population("pursuers", scores, ranking=[3, 2, 1, 0])
    assert len(events) == 1 and events[0]["member"] == 3 and events[0]["exploit_from_member"] == 0
    uid = runner.teams["pursuers"][0]
    for key, value in source.learner.policies[uid].state_dict().items():
        assert torch.equal(value, target.learner.policies[uid].state_dict()[key])
    assert target.rating == 1234.0
    assert target.sampler is not source.sampler and target.sampler.win_rates == source.sampler.win_rates
    # explore: perturbed by one of the factors relative to the source
    factor = events[0]["explore"]["learning_rate_factor"]
    expected = runner._hyperparameters(source)["learning_rate"] * factor
    assert abs(runner._hyperparameters(target)["learning_rate"] - expected) < 1e-12


def test_population_options():
    runner = make_population_runner(population_snapshots="best", generations=1)
    runner.run()
    assert len(runner.pools["pursuers"]) == 2
    assert make_population_runner(sampler="uniform", population_size=None).population_size == 1
    with pytest.raises(ValueError):
        make_population_runner(population_size=9)


def test_wandb_logging_one_step_per_generation(monkeypatch):
    import sys
    import types

    calls = {"init": [], "log": [], "finish": 0}
    fake = types.ModuleType("wandb")
    fake.run = None

    def init(**kwargs):
        calls["init"].append(kwargs)
        fake.run = object()

    def finish():
        calls["finish"] += 1
        fake.run = None

    fake.init = init
    fake.log = lambda metrics, step: calls["log"].append((step, metrics))
    fake.finish = finish
    monkeypatch.setitem(sys.modules, "wandb", fake)

    runner = make_population_runner(wandb=True, wandb_kwargs={"project": "test", "config": {"tag": "x"}})
    runner.run()
    runner.close()
    assert len(calls["init"]) == 1 and calls["init"][0]["project"] == "test"
    config = calls["init"][0]["config"]
    assert config["sampler"] == "adaptive_pbt" and config["population_size"] == 4 and config["tag"] == "x"
    assert config["env"]["obs_mode"] == "full_state"
    assert [step for step, _ in calls["log"]] == [1, 2]
    step1 = calls["log"][0][1]
    assert {"pursuers/train_win_rate", "pursuers/eval_score", "pursuers/opponent_entropy"} <= set(step1)
    assert "pursuers/member3/learning_rate" in step1 and "evader/eval_score" in calls["log"][1][1]
    assert calls["finish"] == 1


def test_sampler_schedule_and_videos(tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("imageio_ffmpeg")
    env = PursuitEvasionEnv(num_envs=4, height=6, width=6, num_pursuers=2, max_steps=8, device="cpu", seed=0)
    cfg = SelfPlayCfg(
        generations=3,
        timesteps_per_generation=8,
        sampler_schedule=["last", "uniform"],
        eval_episodes=2,
        hidden_sizes=(16,),
        agent_cfg={"rollouts": 8, "learning_epochs": 1, "mini_batches": 1},
        video_dir=str(tmp_path / "videos"),
        video_interval=2,
        video_max_steps=5,
        seed=0,
        verbose=False,
    )
    history = SelfPlayRunner(env, cfg).run()
    assert [record["sampler"] for record in history] == ["last", "uniform", "last"]
    assert "videos" not in history[0] and "videos" in history[1]
    for path in history[1]["videos"].values():
        assert os.path.getsize(path) > 0
