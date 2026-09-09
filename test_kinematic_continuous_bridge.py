import torch

from kinematic_continuous_bridge import (KinematicContinuousCondition,
                                           KinematicContinuousControlLatentBridge)
from masked_continuous_completion import (MaskedContinuousCompletion,
                                           continuous_completion_loss)


def test_continuous_kinematic_contract_and_completion_gradients():
    b, f = 1, 5
    direction = torch.nn.functional.normalize(torch.randn(b, f, 2, 20, 3), dim=-1)
    specified = torch.rand(b, f, 2, 20) > 0.5
    observed = torch.where(specified[..., None], direction, torch.zeros_like(direction))
    completion = MaskedContinuousCompletion(width=24, layers=1, heads=4, max_frames=f)
    completed, mean, std = completion(observed, specified)
    loss = continuous_completion_loss(mean, std, direction, specified)
    loss.backward()
    assert loss.isfinite()
    bridge = KinematicContinuousControlLatentBridge(token_width=32)
    condition = KinematicContinuousCondition(
        completed, torch.ones(b, f, 2, 20) * 0.03,
        torch.zeros(b, f, 2, 2), torch.eye(3).expand(b, f, 2, 3, 3), 10.0,
        torch.ones(b, f, 2, 20))
    assert bridge(condition, 16, 16).shape == (b, 16, 2, 16, 16)
