import torch

from kinematic_hl_bridge import KinematicHLCondition, KinematicHLControlLatentBridge


def test_compact_stream_to_wan_latent_and_gradient():
    ids = torch.randint(0, 26, (1, 17, 2, 20))
    posterior = torch.nn.functional.one_hot(ids, 26).float().requires_grad_()
    condition = KinematicHLCondition(
        posterior=posterior,
        bone_length=torch.rand(1, 17, 2, 20) * 0.1,
        root_xy=torch.rand(1, 17, 2, 2) * 2 - 1,
        palm_rotation=torch.eye(3).view(1, 1, 1, 3, 3).expand(1, 17, 2, -1, -1),
        projection_scale=2.0,
    )
    model = KinematicHLControlLatentBridge(token_width=32)
    with torch.no_grad():
        model.bridge.decoder[-1].weight.normal_(std=1e-3)
    output = model(condition, 16, 16)
    assert output.shape == (1, 16, 5, 16, 16)
    output.square().mean().backward()
    assert posterior.grad is not None and posterior.grad.abs().sum() > 0
