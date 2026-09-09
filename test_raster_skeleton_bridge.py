import torch
from raster_skeleton_bridge import RasterSkeletonCondition, RasterSkeletonControlLatentBridge


def test_raster_bridge_shape_gradient_and_mask():
    bridge = RasterSkeletonControlLatentBridge()
    joints = torch.rand(1, 17, 2, 21, 2) * 2 - 1
    mask = torch.ones(1, 17, 2, 20, dtype=torch.bool); mask[:, 5:10] = False
    out = bridge(RasterSkeletonCondition(joints, mask), 16, 16)
    assert out.shape == (1, 16, 5, 16, 16)
    out.square().mean().backward()
    assert any(p.grad is not None for p in bridge.parameters())


def test_hidden_joint_coordinates_cannot_leak():
    torch.manual_seed(4)
    bridge = RasterSkeletonControlLatentBridge().eval()
    joints = torch.rand(1, 17, 2, 21, 2) * 2 - 1
    mask = torch.ones(1, 17, 2, 20, dtype=torch.bool)
    mask[..., 12:16] = False
    perturbed = joints.clone()
    # Bones 12--15 are one finger chain.  Changing its non-root joints must
    # not alter the output while every bone in that chain is unspecified.
    perturbed[..., 13:17, :] = torch.rand_like(perturbed[..., 13:17, :]) * 2 - 1
    with torch.no_grad():
        a = bridge(RasterSkeletonCondition(joints, mask), 8, 8)
        b = bridge(RasterSkeletonCondition(perturbed, mask), 8, 8)
    torch.testing.assert_close(a, b)
