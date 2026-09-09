import torch

from hl_programs import apply_partial_mask, joints_to_hl, structured_mask


def plausible_hand(batch=2, frames=5):
    # Non-degenerate synthetic hand in MediaPipe/FreiHAND joint order.
    joints = torch.zeros(batch, frames, 21, 3)
    bases = [1, 5, 9, 13, 17]
    xs = [-0.7, -0.35, 0.0, 0.35, 0.7]
    for base, x in zip(bases, xs):
        for k in range(4):
            joints[..., base + k, :] = torch.tensor([x, 0.35 + 0.25 * k, 0.08 * x])
    return joints


def test_hl_program_probabilities_and_frames():
    p = joints_to_hl(plausible_hand())
    assert p.posterior.shape == (2, 5, 20, 26)
    assert torch.allclose(p.posterior.sum(-1), torch.ones(2, 5, 20), atol=1e-5)
    eye = p.palm_rotation.transpose(-1, -2) @ p.palm_rotation
    assert torch.allclose(eye, torch.eye(3).expand_as(eye), atol=1e-5)


def test_partial_mask_is_explicit_uniform_not_coordinate_bypass():
    p = joints_to_hl(plausible_hand(batch=1, frames=6))
    mask = structured_mask(6, "distal", device=torch.device("cpu"))
    q = apply_partial_mask(p, mask.unsqueeze(0))
    assert (q.symbol[..., [3, 7, 11, 15, 19]] == -1).all()
    expected = torch.full((26,), 1 / 26)
    assert torch.allclose(q.posterior[0, 0, 3], expected)
    assert q.specified[..., [3, 7, 11, 15, 19]].sum() == 0
    observed = q.specified
    assert torch.equal(q.posterior[observed].argmax(-1), p.symbol[observed])
    assert torch.all((q.posterior[observed] == 0) | (q.posterior[observed] == 1))


def test_interval_mask_is_contiguous():
    mask = structured_mask(17, "interval", device=torch.device("cpu"),
                           generator=torch.Generator().manual_seed(7))
    absent = (~mask).all(-1).nonzero().flatten()
    assert len(absent) == 5
    assert torch.equal(absent, torch.arange(absent[0], absent[0] + 5))
