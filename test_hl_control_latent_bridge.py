import torch

from hl_control_latent_bridge import HLControlLatentBridge, bilinear_splat
from hl_wan_adapter import HLCondition


def condition(batch=2, frames=17):
    ids = torch.randint(0, 26, (batch, frames, 40))
    return HLCondition(
        posterior=torch.nn.functional.one_hot(ids, 26).float(),
        joint_xy=torch.rand(batch, frames, 40, 2) * 2 - 1,
    )


def test_splat_shape_and_mass():
    feature = torch.ones(1, 2, 3, 4)
    xy = torch.zeros(1, 2, 3, 2)
    grid = bilinear_splat(feature, xy, 8, 8)
    assert grid.shape == (1, 4, 2, 8, 8)
    assert torch.isfinite(grid).all()
    assert grid.abs().sum() > 0


def test_bridge_matches_wan_control_latent_contract():
    model = HLControlLatentBridge(token_width=32)
    output = model(condition(), latent_height=16, latent_width=16)
    assert output.shape == (2, 16, 5, 16, 16)
    assert torch.equal(output, torch.zeros_like(output))
    output.square().mean().backward()
    assert model.decoder[-1].weight.grad is not None


def test_bridge_is_small():
    model = HLControlLatentBridge()
    assert model.parameter_count < 3_000_000


def test_parent_coordinates_enable_bone_segment_splatting():
    item = condition(batch=1)
    item = HLCondition(item.posterior, item.joint_xy,
                       parent_xy=torch.zeros_like(item.joint_xy))
    model = HLControlLatentBridge(token_width=32, line_samples=5)
    with torch.no_grad():
        model.decoder[-1].weight.normal_(std=1e-3)
    output = model(item, 16, 16)
    assert output.shape == (1, 16, 5, 16, 16)
    assert torch.isfinite(output).all()
