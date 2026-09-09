import pytest
import torch

from wan_training_contract import (
    assemble_control_y,
    assert_wan_input_contract,
    flow_matching_batch,
)


def test_control_y_and_48_channel_contract():
    control = torch.randn(2, 16, 5, 32, 32)
    first = torch.randn_like(control)
    noisy = torch.randn_like(control)
    y = assemble_control_y(control, first)
    assert y.shape == (2, 32, 5, 32, 32)
    assert_wan_input_contract(noisy, y)


def test_contract_rejects_wrong_channel_count():
    with pytest.raises(ValueError, match="16 channels"):
        assemble_control_y(torch.randn(1, 8, 5, 8, 8), torch.randn(1, 8, 5, 8, 8))


def test_flow_matching_endpoints_and_target():
    clean = torch.randn(2, 16, 2, 4, 4)
    noise = torch.randn_like(clean)
    at_clean, target = flow_matching_batch(clean, torch.zeros(2), noise)
    at_noise, _ = flow_matching_batch(clean, torch.ones(2), noise)
    assert torch.equal(at_clean, clean)
    assert torch.equal(at_noise, noise)
    assert torch.equal(target, noise - clean)
