import torch
from torch import nn

from hl_wan_training_module import HLWanTrainingModule
from kinematic_hl_bridge import KinematicHLCondition, KinematicHLControlLatentBridge


class FakeWan(nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_like = nn.Parameter(torch.tensor(1.0))
        self.last_y = None

    def forward(self, x, t, context, seq_len, y, **kwargs):
        self.last_y = y
        return x + self.lora_like * y[:, :16]


def test_joint_wrapper_uses_native_y_and_trains_both_paths():
    ids = torch.randint(0, 26, (1, 17, 2, 20))
    posterior = torch.nn.functional.one_hot(ids, 26).float()
    condition = KinematicHLCondition(
        posterior=posterior,
        bone_length=torch.rand(1, 17, 2, 20) * 0.1,
        root_xy=torch.zeros(1, 17, 2, 2),
        palm_rotation=torch.eye(3).view(1, 1, 1, 3, 3).expand(1, 17, 2, -1, -1),
        projection_scale=2.0,
    )
    bridge = KinematicHLControlLatentBridge(token_width=32)
    with torch.no_grad():
        bridge.bridge.decoder[-1].weight.normal_(std=1e-3)
    wan = FakeWan()
    model = HLWanTrainingModule(wan, bridge)
    noisy = torch.randn(1, 16, 5, 16, 16)
    first = torch.zeros_like(noisy)
    output = model(noisy, first, condition, torch.ones(1), [torch.zeros(2, 4)], 320)
    assert wan.last_y.shape == (1, 32, 5, 16, 16)
    output.square().mean().backward()
    assert wan.lora_like.grad is not None
    assert bridge.bridge.decoder[-1].weight.grad is not None
