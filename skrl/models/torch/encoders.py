"""Per-key observation encoders for Dict observation spaces (images, point clouds, language tokens, proprioception).

skrl stores a Dict observation flattened (keys in sorted order). :class:`DictEncoder` unflattens it with the
observation space, encodes each key with its own encoder and fuses the features (concatenation, or FiLM
conditioning on one key, e.g. a CLIP text embedding modulating the image features)::

    encoders:
      image: {type: dinov2, name: dinov2_vits14}      # frozen foundation model, lazy torch.hub import
      points: {type: pointnet, hidden: [64, 128]}
      instruction: {type: clip_text}                   # token ids from ``clip_tokenize``, lazy open_clip import
      proprio: {type: mlp, hidden: [64]}
    fusion: {type: film, context: instruction}         # or "concat"

Any other encoder: ``register_encoder("name")`` on an ``nn.Module`` class whose ``__init__`` takes the key's
gymnasium space as first argument and which exposes ``output_dim``, or ``type: "module:Class"`` in the YAML.
"""
from __future__ import annotations

import importlib
from typing import Any, Callable

import gymnasium
import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl.models.torch.base import Model
from skrl.models.torch.categorical import CategoricalMixin
from skrl.models.torch.deterministic import DeterministicMixin
from skrl.models.torch.gaussian import GaussianMixin
from skrl.utils.spaces.torch import compute_space_size, unflatten_tensorized_space

ENCODERS: dict[str, type[nn.Module]] = {}


def register_encoder(name: str) -> Callable[[type[nn.Module]], type[nn.Module]]:
    """Class decorator: make an encoder available as ``type: <name>`` in :func:`make_encoder` and the YAML."""

    def decorator(cls: type[nn.Module]) -> type[nn.Module]:
        ENCODERS[name.lower()] = cls
        return cls

    return decorator


def make_encoder(spec: str | dict, space: gymnasium.Space) -> nn.Module:
    """Build an encoder from ``"type"`` or ``{"type": ..., **kwargs}``; the type is a registered name or
    ``"module:Class"``."""
    spec = {"type": spec} if isinstance(spec, str) else dict(spec)
    name = spec.pop("type")
    if ":" in name:
        module, attr = name.split(":", 1)
        cls = getattr(importlib.import_module(module), attr)
    elif name.lower() in ENCODERS:
        cls = ENCODERS[name.lower()]
    else:
        raise ValueError(f"Unknown encoder '{name}'. Registered: {sorted(ENCODERS)}")
    return cls(space, **spec)


def _mlp(sizes: list[int], activation: type[nn.Module] = nn.ELU) -> nn.Sequential:
    layers = []
    for i in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[i], sizes[i + 1]), activation()]
    return nn.Sequential(*layers)


def _space_size(space: gymnasium.Space) -> int:
    return compute_space_size(space, occupied_size=True)


def _channels_first(x: torch.Tensor) -> torch.Tensor:
    """``[B, H, W, C]`` (gymnasium images) or ``[B, C, H, W]`` → ``[B, C, H, W]`` float."""
    if x.shape[-1] in (1, 3, 4) and x.shape[1] not in (1, 3, 4):
        x = x.permute(0, 3, 1, 2)
    return x.float()


def _to_unit_range(x: torch.Tensor, space: gymnasium.Space) -> torch.Tensor:
    """Pixels in ``[0, 255]`` (uint8 Box) → ``[0, 1]``."""
    return x / 255.0 if getattr(space, "dtype", None) == "uint8" or float(space.high.max()) > 1.0 else x


# --- trainable encoders -------------------------------------------------------------------------------------------


@register_encoder("identity")
class IdentityEncoder(nn.Module):
    """Flatten the key and pass it through."""

    def __init__(self, space: gymnasium.Space) -> None:
        super().__init__()
        self.output_dim = _space_size(space)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0], -1).float()


@register_encoder("mlp")
class MLPEncoder(nn.Module):
    """Flatten the key, then an MLP (low-dimensional state, proprioception, a belief vector)."""

    def __init__(self, space: gymnasium.Space, hidden: list[int] = (64,)) -> None:
        super().__init__()
        self.net = _mlp([_space_size(space), *hidden])
        self.output_dim = hidden[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.reshape(x.shape[0], -1).float())


@register_encoder("cnn")
class CNNEncoder(nn.Module):
    """Nature-DQN convolution stack for small images (``[H, W, C]`` or ``[C, H, W]``), trained from scratch."""

    def __init__(self, space: gymnasium.Space, features: int = 256) -> None:
        super().__init__()
        self.space = space
        c = space.shape[-1] if space.shape[-1] in (1, 3, 4) and space.shape[0] not in (1, 3, 4) else space.shape[0]
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )  # fmt: skip
        with torch.no_grad():
            n = self.conv(_channels_first(torch.zeros(1, *space.shape))).shape[-1]
        self.head = nn.Sequential(nn.Linear(n, features), nn.ReLU())
        self.output_dim = features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(_to_unit_range(_channels_first(x), self.space)))


@register_encoder("pointnet")
class PointNetEncoder(nn.Module):
    """PointNet: a shared per-point MLP then a max pool over the points, so the feature is invariant to the point
    order. The key is ``[N, C]`` (N points with C = 3 coordinates, or more with normals/colours)."""

    def __init__(self, space: gymnasium.Space, hidden: list[int] = (64, 128), features: int | None = None) -> None:
        super().__init__()
        self.point_dim = space.shape[-1]
        self.point_mlp = _mlp([self.point_dim, *hidden], activation=nn.ReLU)
        self.head = nn.Linear(hidden[-1], features) if features else nn.Identity()
        self.output_dim = features or hidden[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.shape[0], -1, self.point_dim).float()
        return self.head(self.point_mlp(x).max(dim=1).values)


# --- pretrained foundation encoders (optional dependencies, imported when used) ----------------------------------

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _load_dinov2(name: str) -> nn.Module:
    return torch.hub.load("facebookresearch/dinov2", name)


def _load_open_clip(name: str, pretrained: str) -> nn.Module:
    try:
        import open_clip
    except ImportError as e:
        raise ImportError("The 'clip_image' and 'clip_text' encoders need open_clip: pip install open_clip_torch") from e
    model, _, _ = open_clip.create_model_and_transforms(name, pretrained=pretrained)
    return model


def clip_tokenize(texts: list[str], name: str = "ViT-B-32", context_length: int = 77) -> torch.Tensor:
    """Token ids ``[len(texts), context_length]`` for the ``clip_text`` encoder (store them as an observation key,
    e.g. a ``Box(0, 49407, (77,), int64)`` instruction)."""
    import open_clip

    return open_clip.get_tokenizer(name)(texts, context_length=context_length)


class _PretrainedImageEncoder(nn.Module):
    """Resize to the backbone resolution, normalize, run the (frozen by default) backbone."""

    def __init__(self, space, backbone: nn.Module, output_dim: int, size: int, mean, std, freeze: bool) -> None:
        super().__init__()
        self.space, self.backbone, self.output_dim, self.size, self.freeze = space, backbone, output_dim, size, freeze
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)
        if freeze:
            self.backbone.requires_grad_(False).eval()

    def train(self, mode: bool = True) -> "_PretrainedImageEncoder":
        super().train(mode)
        if self.freeze:
            self.backbone.eval()
        return self

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _to_unit_range(_channels_first(x), self.space)
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        x = F.interpolate(x[:, :3], size=(self.size, self.size), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        with torch.set_grad_enabled(not self.freeze and torch.is_grad_enabled()):
            return self.embed(x).float()


@register_encoder("dinov2")
class DINOv2Encoder(_PretrainedImageEncoder):
    """DINOv2 ViT CLS embedding (``dinov2_vits14``: 384, ``vitb14``: 768, ...) from torch.hub."""

    def __init__(self, space, name: str = "dinov2_vits14", size: int = 224, freeze: bool = True) -> None:
        backbone = _load_dinov2(name)
        super().__init__(space, backbone, backbone.embed_dim, size, _IMAGENET_MEAN, _IMAGENET_STD, freeze)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


@register_encoder("clip_image")
class CLIPImageEncoder(_PretrainedImageEncoder):
    """CLIP image embedding (open_clip ``encode_image``), in the same space as ``clip_text``."""

    def __init__(self, space, name: str = "ViT-B-32", pretrained: str = "openai", freeze: bool = True) -> None:
        model = _load_open_clip(name, pretrained)
        size = model.visual.image_size
        size = size[0] if isinstance(size, (tuple, list)) else size
        super().__init__(space, model, model.visual.output_dim, size, _CLIP_MEAN, _CLIP_STD, freeze)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.encode_image(x)


@register_encoder("clip_text")
class CLIPTextEncoder(nn.Module):
    """CLIP text embedding of token ids (see :func:`clip_tokenize`), e.g. a language instruction."""

    def __init__(self, space, name: str = "ViT-B-32", pretrained: str = "openai", freeze: bool = True) -> None:
        super().__init__()
        self.backbone, self.freeze = _load_open_clip(name, pretrained), freeze
        self.output_dim = self.backbone.text_projection.shape[-1]
        if freeze:
            self.backbone.requires_grad_(False).eval()

    def train(self, mode: bool = True) -> "CLIPTextEncoder":
        super().train(mode)
        if self.freeze:
            self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.set_grad_enabled(not self.freeze and torch.is_grad_enabled()):
            return self.backbone.encode_text(x.reshape(x.shape[0], -1).long()).float()


# --- fusion -------------------------------------------------------------------------------------------------------


class FiLM(nn.Module):
    """Feature-wise linear modulation: ``h' = (1 + γ(c)) ⊙ h + β(c)`` (γ, β initialized to 0, so h' = h at start)."""

    def __init__(self, context_dim: int, feature_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(context_dim, 2 * feature_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.proj(c).chunk(2, dim=-1)
        return (1.0 + gamma) * h + beta


class DictEncoder(nn.Module):
    """Encode a flattened Dict observation key by key and fuse the features.

    :param observation_space: The Dict observation space (the flat input is unflattened with it).
    :param encoders: Encoder spec per key (:func:`make_encoder`). Keys without a spec use ``identity``.
    :param fusion: ``"concat"``, or ``{"type": "film", "context": <key>}``: the context key's feature modulates
        every other key's feature with its own :class:`FiLM` layer, and the output concatenates the modulated
        features (the context feature itself is not repeated).
    """

    def __init__(
        self, observation_space: gymnasium.spaces.Dict, encoders: dict | None = None, fusion: str | dict = "concat"
    ) -> None:
        super().__init__()
        if not isinstance(observation_space, gymnasium.spaces.Dict):
            raise ValueError(f"DictEncoder needs a Dict observation space, got {observation_space}")
        self.observation_space = observation_space
        encoders = encoders or {}
        unknown = set(encoders) - set(observation_space.keys())
        if unknown:
            raise ValueError(f"Encoders for keys {sorted(unknown)} not in the observation space {list(observation_space)}")
        self.encoders = nn.ModuleDict(
            {k: make_encoder(encoders.get(k, "identity"), s) for k, s in observation_space.items()}
        )
        fusion = {"type": fusion} if isinstance(fusion, str) else dict(fusion)
        self.context = fusion.get("context") if fusion["type"].lower() == "film" else None
        if fusion["type"].lower() not in ("concat", "film"):
            raise ValueError(f"Unknown fusion '{fusion['type']}' (concat or film)")
        if fusion["type"].lower() == "film" and self.context not in self.encoders:
            raise ValueError(f"FiLM context key '{self.context}' is not an observation key")
        self._order = [k for k in sorted(self.encoders) if k != self.context]
        if self.context is not None:
            c = self.encoders[self.context].output_dim
            self.film = nn.ModuleDict({k: FiLM(c, self.encoders[k].output_dim) for k in self._order})
        self.output_dim = sum(self.encoders[k].output_dim for k in self._order)

    def forward(self, x: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        if not isinstance(x, dict):
            x = unflatten_tensorized_space(self.observation_space, x)
        if self.context is None:
            return torch.cat([self.encoders[k](x[k]) for k in self._order], dim=-1)
        c = self.encoders[self.context](x[self.context])
        return torch.cat([self.film[k](self.encoders[k](x[k]), c) for k in self._order], dim=-1)


# --- models -------------------------------------------------------------------------------------------------------


class _EncoderNet(nn.Module):
    def __init__(self, observation_space, encoders, fusion, hidden: list[int], output_dim: int) -> None:
        super().__init__()
        self.encoder = DictEncoder(observation_space, encoders, fusion)
        self.mlp = _mlp([self.encoder.output_dim, *hidden])
        self.head = nn.Linear(hidden[-1] if hidden else self.encoder.output_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.mlp(self.encoder(x)))


class EncoderGaussianPolicy(GaussianMixin, Model):
    """Gaussian policy: :class:`DictEncoder` → MLP → mean actions, state-independent log std."""

    def __init__(
        self,
        observation_space,
        action_space,
        device=None,
        *,
        encoders: dict | None = None,
        fusion: str | dict = "concat",
        hidden: list[int] = (256, 128),
        clip_actions: bool = False,
        initial_log_std: float = 0.0,
    ) -> None:
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        GaussianMixin.__init__(self, clip_actions=clip_actions, clip_log_std=True, min_log_std=-20.0, max_log_std=2.0)
        self.net = _EncoderNet(observation_space, encoders, fusion, list(hidden), self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.full((self.num_actions,), float(initial_log_std)))

    def compute(self, inputs: dict[str, Any], role: str = "") -> tuple[torch.Tensor, dict[str, Any]]:
        return self.net(inputs["observations"]), {"log_std": self.log_std_parameter}


class EncoderCategoricalPolicy(CategoricalMixin, Model):
    """Categorical policy: :class:`DictEncoder` → MLP → logits."""

    def __init__(
        self,
        observation_space,
        action_space,
        device=None,
        *,
        encoders: dict | None = None,
        fusion: str | dict = "concat",
        hidden: list[int] = (256, 128),
    ) -> None:
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        CategoricalMixin.__init__(self, unnormalized_log_prob=True)
        self.net = _EncoderNet(observation_space, encoders, fusion, list(hidden), self.num_actions)

    def compute(self, inputs: dict[str, Any], role: str = "") -> tuple[torch.Tensor, dict[str, Any]]:
        return self.net(inputs["observations"]), {}


class EncoderDeterministic(DeterministicMixin, Model):
    """Deterministic model (e.g. the value function): :class:`DictEncoder` → MLP → ``output_size`` outputs."""

    def __init__(
        self,
        observation_space,
        action_space,
        device=None,
        *,
        encoders: dict | None = None,
        fusion: str | dict = "concat",
        hidden: list[int] = (256, 128),
        output_size: int = 1,
    ) -> None:
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self)
        self.net = _EncoderNet(observation_space, encoders, fusion, list(hidden), output_size)

    def compute(self, inputs: dict[str, Any], role: str = "") -> tuple[torch.Tensor, dict[str, Any]]:
        return self.net(inputs["observations"]), {}
