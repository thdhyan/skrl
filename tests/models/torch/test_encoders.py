"""Dict-observation encoders (MLP, CNN, PointNet, DINOv2, CLIP, FiLM, custom), DictStandardScaler and Runner wiring."""
from __future__ import annotations

import types

import gymnasium
import numpy as np
import pytest
import torch
import torch.nn as nn

from skrl.envs.wrappers.torch.base import Wrapper
from skrl.models.torch import encoders as enc
from skrl.models.torch.encoders import (
    DictEncoder,
    EncoderDeterministic,
    EncoderGaussianPolicy,
    FiLM,
    make_encoder,
    register_encoder,
)
from skrl.resources.preprocessors.torch import DictStandardScaler
from skrl.utils.runner.torch import Runner
from skrl.utils.spaces.torch import flatten_tensorized_space

SPACE = gymnasium.spaces.Dict(
    {
        "proprio": gymnasium.spaces.Box(-np.inf, np.inf, (4,)),
        "image": gymnasium.spaces.Box(0, 255, (36, 36, 3), np.uint8),
        "points": gymnasium.spaces.Box(-1.0, 1.0, (16, 3)),
        "instruction": gymnasium.spaces.Box(0, 99, (8,), np.int64),
    }
)


class _FakeDINO(nn.Module):
    embed_dim = 12

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3, 12)

    def forward(self, x):
        assert x.shape[-2:] == (28, 28)  # resized to the backbone resolution
        return self.proj(x.mean(dim=(2, 3)))


class _FakeCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = types.SimpleNamespace(image_size=(28, 28), output_dim=10)
        self.image_proj = nn.Linear(3, 10)
        self.token_embedding = nn.Embedding(100, 10)
        self.text_projection = nn.Parameter(torch.eye(10))

    def encode_image(self, x):
        return self.image_proj(x.mean(dim=(2, 3)))

    def encode_text(self, tokens):
        return self.token_embedding(tokens).mean(1) @ self.text_projection


@pytest.fixture
def fake_foundation_models(monkeypatch):
    monkeypatch.setattr(enc, "_load_dinov2", lambda name: _FakeDINO())
    monkeypatch.setattr(enc, "_load_open_clip", lambda name, pretrained: _FakeCLIP())


def _sample(batch=5):
    return {k: torch.as_tensor(np.stack([s.sample() for _ in range(batch)])) for k, s in SPACE.items()}


def test_trainable_encoders_shapes():
    x = _sample()
    for spec, key, dim in [("mlp", "proprio", 64), ("cnn", "image", 256), ("pointnet", "points", 128),
                           ("identity", "instruction", 8)]:
        e = make_encoder(spec, SPACE[key])
        assert e.output_dim == dim
        assert e(x[key]).shape == (5, dim)


def test_pointnet_is_permutation_invariant():
    e = make_encoder({"type": "pointnet", "hidden": [32, 64], "features": 16}, SPACE["points"])
    pts = torch.randn(3, 16, 3)
    assert torch.allclose(e(pts), e(pts[:, torch.randperm(16)]), atol=1e-6)


def test_film_starts_as_identity_and_conditions():
    film = FiLM(context_dim=5, feature_dim=7)
    h, c = torch.randn(4, 7), torch.randn(4, 5)
    assert torch.allclose(film(h, c), h)
    nn.init.normal_(film.proj.weight)
    assert not torch.allclose(film(h, c), h)


def test_pretrained_encoders_are_frozen_and_normalize(fake_foundation_models):
    x = _sample()
    dino = make_encoder({"type": "dinov2", "size": 28}, SPACE["image"])
    clip_img = make_encoder("clip_image", SPACE["image"])
    clip_txt = make_encoder("clip_text", SPACE["instruction"])
    assert dino(x["image"]).shape == (5, 12)
    assert clip_img(x["image"]).shape == (5, 10)
    assert clip_txt(x["instruction"]).shape == (5, 10)
    for e in (dino, clip_img, clip_txt):
        assert not any(p.requires_grad for p in e.parameters())
        e.train()
        assert not e.backbone.training  # stays in eval mode when frozen
    unfrozen = make_encoder({"type": "dinov2", "size": 28, "freeze": False}, SPACE["image"])
    unfrozen(x["image"]).sum().backward()
    assert unfrozen.backbone.proj.weight.grad is not None


def test_dict_encoder_unflattens_in_skrl_order_and_films(fake_foundation_models):
    x = _sample()
    flat = flatten_tensorized_space(x).float()
    encoders = {"image": "clip_image", "points": "pointnet", "instruction": "clip_text", "proprio": "mlp"}
    concat = DictEncoder(SPACE, encoders)
    assert concat.output_dim == 10 + 10 + 128 + 64
    assert torch.allclose(concat(flat), concat({k: v for k, v in x.items()}))
    film = DictEncoder(SPACE, encoders, fusion={"type": "film", "context": "instruction"})
    assert film.output_dim == 10 + 128 + 64  # the context feature is not repeated
    assert set(film.film) == {"image", "points", "proprio"}
    assert film(flat).shape == (5, film.output_dim)
    with pytest.raises(ValueError, match="not in the observation space"):
        DictEncoder(SPACE, {"depth": "cnn"})
    with pytest.raises(ValueError, match="FiLM context"):
        DictEncoder(SPACE, encoders, fusion={"type": "film", "context": "depth"})


@register_encoder("test_mean")
class _MeanEncoder(nn.Module):
    def __init__(self, space, scale: float = 1.0):
        super().__init__()
        self.scale, self.output_dim = scale, 1

    def forward(self, x):
        return self.scale * x.reshape(x.shape[0], -1).float().mean(-1, keepdim=True)


def test_custom_encoders_by_registry_and_import_string():
    x = _sample()
    e = make_encoder({"type": "test_mean", "scale": 2.0}, SPACE["points"])
    assert torch.allclose(e(x["points"]), 2.0 * x["points"].reshape(5, -1).mean(-1, keepdim=True))
    e = make_encoder(f"{__name__}:_MeanEncoder", SPACE["proprio"])
    assert e(x["proprio"]).shape == (5, 1)
    with pytest.raises(ValueError, match="Unknown encoder"):
        make_encoder("nope", SPACE["proprio"])


def test_dict_standard_scaler_only_touches_low_dimensional_keys():
    scaler = DictStandardScaler(SPACE, device="cpu")
    assert scaler.keys == ["proprio"]  # not the uint8 image, the 2-D points or the int tokens
    x = _sample(64)
    x["proprio"] = 3.0 + 10.0 * torch.randn(64, 4)
    flat = flatten_tensorized_space(x).float()
    for _ in range(5):
        out = scaler(flat, train=True)
    # sorted keys: image (3888), instruction (8), points (48), proprio (4)
    assert torch.equal(out[:, :-4], flat[:, :-4])
    assert out[:, -4:].mean().abs() < 0.2 and (out[:, -4:].std() - 1.0).abs() < 0.2
    assert torch.allclose(scaler(out, inverse=True)[:, -4:], flat[:, -4:], atol=1e-3)
    assert "scaler.running_mean" in scaler.state_dict()
    assert torch.equal(DictStandardScaler(SPACE, keys=["points"], device="cpu").index, torch.arange(3896, 3944))


class FakeDictEnv(Wrapper):
    """Isaac Lab-like env with a Dict observation, returned flattened as the skrl wrappers do."""

    def __init__(self, num_envs: int = 4):
        super().__init__(types.SimpleNamespace(device="cpu"))
        self._n, self.t = num_envs, 0

    @property
    def num_envs(self):
        return self._n

    @property
    def observation_space(self):
        return SPACE

    @property
    def state_space(self):
        return None

    @property
    def action_space(self):
        return gymnasium.spaces.Box(-1.0, 1.0, (2,))

    def _obs(self):
        return flatten_tensorized_space(_sample(self._n)).float()

    def reset(self):
        return self._obs(), {}

    def step(self, actions):
        self.t += 1
        truncated = torch.full((self._n, 1), self.t % 5 == 0)
        return self._obs(), -actions.pow(2).sum(-1, keepdim=True), torch.zeros_like(truncated), truncated, {}

    def state(self):
        return None

    def render(self, *args, **kwargs):
        return None

    def close(self):
        pass


def test_runner_builds_and_trains_ppo_with_encoders_from_yaml(fake_foundation_models, tmp_path):
    encoders = {"image": {"type": "dinov2", "size": 28}, "points": {"type": "pointnet", "hidden": [32]},
                "instruction": "clip_text", "proprio": {"type": "mlp", "hidden": [16]}}
    fusion = {"type": "film", "context": "instruction"}
    cfg = {
        "seed": 0,
        "models": {
            "separate": True,
            "policy": {"class": "EncoderGaussianPolicy", "encoders": encoders, "fusion": fusion, "hidden": [32]},
            "value": {"class": "EncoderDeterministic", "encoders": encoders, "hidden": [32]},
        },
        "memory": {"class": "RandomMemory", "memory_size": -1},
        "agent": {
            "class": "PPO",
            "rollouts": 8,
            "learning_epochs": 1,
            "mini_batches": 2,
            "observation_preprocessor": "DictStandardScaler",
            "observation_preprocessor_kwargs": {"keys": ["proprio"]},
            "experiment": {"directory": str(tmp_path), "write_interval": 0, "checkpoint_interval": 8},
        },
        "trainer": {"class": "SequentialTrainer", "timesteps": 16, "headless": True, "disable_progressbar": True},
    }
    runner = Runner(FakeDictEnv(), cfg)
    policy = runner.agent.policy
    assert isinstance(policy, EncoderGaussianPolicy) and isinstance(runner.agent.value, EncoderDeterministic)
    assert isinstance(runner.agent._observation_preprocessor, DictStandardScaler)
    film = policy.net.encoder.film["points"].proj.weight.clone()
    dino = policy.net.encoder.encoders["image"].backbone.proj.weight.clone()
    runner.run("train")
    assert not torch.equal(film, policy.net.encoder.film["points"].proj.weight)  # FiLM trained
    assert torch.equal(dino, policy.net.encoder.encoders["image"].backbone.proj.weight)  # frozen DINOv2
    checkpoints = sorted(tmp_path.glob("*/checkpoints/agent_*.pt"))
    assert checkpoints
    Runner(FakeDictEnv(), cfg).agent.load(str(checkpoints[-1]))  # encoders and the per-key scaler round-trip
