import torch

from continuous_control_latent_bridge import ContinuousCondition, ContinuousControlLatentBridge
from hl_control_latent_bridge import HLControlLatentBridge


def test_continuous_bridge_contract_and_exact_parameter_match():
    direction = torch.nn.functional.normalize(torch.randn(2, 17, 40, 3), dim=-1)
    xy = torch.rand(2, 17, 40, 2) * 2 - 1
    condition = ContinuousCondition(direction, xy, parent_xy=torch.zeros_like(xy))
    baseline = ContinuousControlLatentBridge(token_width=32)
    hl = HLControlLatentBridge(token_width=32)
    output = baseline(condition, 16, 16)
    assert output.shape == (2, 16, 5, 16, 16)
    assert torch.equal(output, torch.zeros_like(output))
    assert baseline.parameter_count == hl.parameter_count
