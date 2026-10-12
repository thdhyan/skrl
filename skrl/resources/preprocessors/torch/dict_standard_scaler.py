from __future__ import annotations

import gymnasium
import numpy as np

import torch
import torch.nn as nn

from skrl.resources.preprocessors.torch.running_standard_scaler import RunningStandardScaler
from skrl.utils.spaces.torch import compute_space_size


def _is_low_dimensional(space: gymnasium.Space) -> bool:
    """Float Box with at most one dimension: proprioception, poses, belief vectors (not images, points or tokens)."""
    return isinstance(space, gymnasium.spaces.Box) and np.issubdtype(space.dtype, np.floating) and len(space.shape) <= 1


class DictStandardScaler(nn.Module):
    def __init__(
        self,
        size: gymnasium.spaces.Dict,
        *,
        keys: list[str] | None = None,
        epsilon: float = 1e-8,
        clip_threshold: float = 5.0,
        device: str | torch.device | None = None,
    ) -> None:
        """Running standardization of selected keys of a flattened Dict observation; the other keys (images, point
        clouds, token ids) pass through unchanged.

        Drop-in for :class:`RunningStandardScaler` as ``observation_preprocessor`` (same call signature and
        checkpoint buffers), e.g. in the YAML::

            observation_preprocessor: DictStandardScaler
            observation_preprocessor_kwargs: {keys: [proprio, goal]}

        :param size: The Dict observation space (flattened with its keys in sorted order, as skrl does).
        :param keys: Keys to standardize. Default: every float Box key with at most one dimension.
        :param epsilon: Small number to avoid division by zero.
        :param clip_threshold: Threshold to clip the data.
        :param device: Data allocation and computation device.
        """
        super().__init__()
        if not isinstance(size, gymnasium.spaces.Dict):
            raise ValueError(f"DictStandardScaler needs a Dict observation space, got {size}")
        if keys is None:
            keys = [k for k, s in size.items() if _is_low_dimensional(s)]
        unknown = set(keys) - set(size.keys())
        if unknown:
            raise ValueError(f"Keys {sorted(unknown)} not in the observation space {list(size)}")
        self.keys = sorted(keys)
        index, start = [], 0
        for k in sorted(size.keys()):
            n = compute_space_size(size[k], occupied_size=True)
            if k in keys:
                index.append(torch.arange(start, start + n))
            start += n
        self.size = start
        self.scaler = RunningStandardScaler(
            sum(len(i) for i in index), epsilon=epsilon, clip_threshold=clip_threshold, device=device
        )
        self.register_buffer("index", torch.cat(index) if index else torch.zeros(0, dtype=torch.long), persistent=False)
        self.index = self.index.to(self.scaler.device)

    def forward(
        self, x: torch.Tensor | None, *, train: bool = False, inverse: bool = False, no_grad: bool = True
    ) -> torch.Tensor | None:
        if x is None or not len(self.index):
            return x
        out = x.clone()
        out[..., self.index] = self.scaler(x[..., self.index], train=train, inverse=inverse, no_grad=no_grad)
        return out
