import pytest
import torch

from generate_sethl_wan import calibrate_posterior


def test_temperature_preserves_specified_and_flattens_hidden():
    posterior = torch.tensor([[[.8, .2], [1.0, 0.0]]])
    specified = torch.tensor([[False, True]])
    result = calibrate_posterior(posterior, specified, 2.0)
    assert torch.equal(result[0, 1], posterior[0, 1])
    assert result[0, 0, 0] < posterior[0, 0, 0]
    assert torch.allclose(result.sum(-1), torch.ones_like(result[..., 0]))


def test_temperature_must_be_positive():
    with pytest.raises(ValueError):
        calibrate_posterior(torch.ones(1, 2) / 2, torch.tensor([False]), 0.0)

