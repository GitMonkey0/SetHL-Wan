import torch

from kinematic_spatializer import (hl_centers, posterior_moments,
                                   relaxed_hl_sample, sample_within_hl_cell,
                                   spatialize_bimanual)


def test_spatializer_contract_and_gradient():
    direction = torch.nn.functional.normalize(torch.randn(2, 5, 2, 20, 3), dim=-1)
    direction.requires_grad_()
    lengths = torch.rand(2, 5, 2, 20) * 0.1
    roots = torch.rand(2, 5, 2, 2) * 2 - 1
    rotation = torch.eye(3).view(1, 1, 1, 3, 3).expand(2, 5, 2, -1, -1)
    child, parent = spatialize_bimanual(direction, lengths, roots, rotation, 2.0)
    assert child.shape == parent.shape == (2, 5, 40, 2)
    child.square().mean().backward()
    assert direction.grad is not None and direction.grad.abs().sum() > 0


def test_zero_lengths_collapse_to_root():
    direction = torch.zeros(1, 1, 2, 20, 3)
    direction[..., 0] = 1
    lengths = torch.zeros(1, 1, 2, 20)
    roots = torch.tensor([[[[0.2, 0.3], [-0.4, 0.5]]]])
    rotation = torch.eye(3).view(1, 1, 1, 3, 3).expand(1, 1, 2, -1, -1)
    child, parent = spatialize_bimanual(direction, lengths, roots, rotation, 1.0)
    assert torch.allclose(child.view(1, 1, 2, 20, 2), roots[..., None, :].expand_as(child.view(1, 1, 2, 20, 2)))
    assert torch.allclose(parent, child)


def test_posterior_moments_preserve_concentration():
    codebook = hl_centers(torch.device("cpu"), torch.float32)
    one_hot = torch.nn.functional.one_hot(torch.tensor([0]), 26).float()
    direction, concentration, covariance = posterior_moments(one_hot, codebook)
    assert torch.allclose(direction, codebook[:1])
    assert torch.allclose(concentration, torch.ones_like(concentration))
    assert torch.allclose(covariance, torch.zeros_like(covariance), atol=1e-6)
    uniform = torch.full((1, 26), 1 / 26)
    _, diffuse_concentration, diffuse_covariance = posterior_moments(uniform, codebook)
    assert diffuse_concentration.item() < 1e-6
    assert torch.trace(diffuse_covariance[0]) > 0.9


def test_relaxed_sample_is_unit_length_and_differentiable():
    logits = torch.randn(4, 26, requires_grad=True)
    posterior = logits.softmax(-1)
    sample = relaxed_hl_sample(posterior, hl_centers(torch.device("cpu"), torch.float32))
    assert torch.allclose(torch.linalg.vector_norm(sample, dim=-1), torch.ones(4), atol=1e-5)
    sample.sum().backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0


def test_within_cell_sample_preserves_symbol_varies_and_has_gradient():
    torch.manual_seed(7)
    codebook = hl_centers(torch.device("cpu"), torch.float32)
    target = torch.arange(26).repeat_interleave(32)
    logits = torch.full((len(target), 26), -20.0, requires_grad=True)
    # Make the selected forward symbol deterministic while retaining the
    # straight-through path through logits.
    with torch.no_grad():
        logits.scatter_(1, target[:, None], 20.0)
    posterior = logits.softmax(-1)
    sample = sample_within_hl_cell(posterior, codebook)
    assert torch.allclose(sample.norm(dim=-1), torch.ones(len(target)), atol=1e-5)
    assert torch.equal((sample @ codebook.T).argmax(-1), target)
    assert sample.std(dim=0).sum() > 0.01
    sample.square().mul(torch.tensor([1.0, 2.0, 3.0])).sum().backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0
