import pytest

import gymnasium
import torch

from skrl.models.torch.transformer import TransformerEncoderTrunk, TransformerGaussianPolicy, TransformerValue


OBS_DIM, ACT_DIM = 17, 6


def _spaces():
    observation_space = gymnasium.spaces.Box(-float("inf"), float("inf"), (OBS_DIM,))
    action_space = gymnasium.spaces.Box(-1.0, 1.0, (ACT_DIM,))
    return observation_space, action_space


@pytest.mark.parametrize("batch_size", [1, 8])
def test_policy_output_shapes(batch_size):
    observation_space, action_space = _spaces()
    policy = TransformerGaussianPolicy(observation_space, action_space, "cpu")
    inputs = {"observations": torch.randn(batch_size, OBS_DIM)}

    actions, outputs = policy.act(inputs, role="policy")

    assert actions.shape == (batch_size, ACT_DIM)
    assert outputs["log_prob"].shape == (batch_size, 1)
    assert outputs["mean_actions"].shape == (batch_size, ACT_DIM)
    assert outputs["log_std"].shape == (ACT_DIM,)


@pytest.mark.parametrize("batch_size", [1, 8])
def test_value_output_shape(batch_size):
    observation_space, action_space = _spaces()
    value = TransformerValue(observation_space, action_space, "cpu")
    values, outputs = value.act({"observations": torch.randn(batch_size, OBS_DIM)}, role="value")

    assert values.shape == (batch_size, 1)
    assert outputs == {}


def test_trunk_returns_cls_embedding():
    trunk = TransformerEncoderTrunk(OBS_DIM, d_model=32, n_heads=4, n_layers=1, dim_ff=64)
    assert trunk(torch.randn(5, OBS_DIM)).shape == (5, 32)


def test_gradients_flow_to_all_parameters():
    observation_space, action_space = _spaces()
    policy = TransformerGaussianPolicy(observation_space, action_space, "cpu")
    value = TransformerValue(observation_space, action_space, "cpu")
    obs = torch.randn(8, OBS_DIM)

    actions, outputs = policy.act({"observations": obs}, role="policy")
    loss = -outputs["log_prob"].mean() + value.act({"observations": obs}, role="value")[0].pow(2).mean()
    loss.backward()

    for name, param in list(policy.named_parameters()) + list(value.named_parameters()):
        assert param.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(param.grad).all(), f"non-finite gradient for {name}"
    assert policy.log_std_parameter.grad.abs().sum() > 0
