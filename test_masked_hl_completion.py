import torch

from masked_hl_completion import MaskedHLCompletion, hierarchical_completion_loss


def test_completion_preserves_observed_cells_and_trains_hidden_cells():
    model = MaskedHLCompletion(width=24, layers=1, heads=4, max_frames=5)
    symbol = torch.randint(0, 26, (2, 5, 2, 20))
    truth = torch.nn.functional.one_hot(symbol, 26).float()
    specified = torch.rand(2, 5, 2, 20) > 0.4
    observed = torch.where(specified[..., None], truth, torch.full_like(truth, 1 / 26))
    completed, logits = model(observed, specified)
    assert completed.shape == truth.shape
    assert torch.equal(completed[specified], truth[specified])
    assert torch.allclose(completed.sum(-1), torch.ones_like(completed[..., 0]), atol=1e-5)
    loss = hierarchical_completion_loss(logits, symbol, specified)
    loss.backward()
    assert loss.isfinite()
    assert all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters())


def test_no_hidden_cells_returns_differentiable_zero():
    model = MaskedHLCompletion(width=24, layers=1, heads=4, max_frames=2)
    posterior = torch.full((1, 2, 2, 20, 26), 1 / 26)
    completed, logits = model(posterior, torch.ones(1, 2, 2, 20, dtype=torch.bool))
    loss = hierarchical_completion_loss(logits, torch.zeros(1, 2, 2, 20, dtype=torch.long),
                                        torch.ones(1, 2, 2, 20, dtype=torch.bool))
    loss.backward()
    assert float(loss.detach()) == 0.0
    assert torch.allclose(completed, posterior)


def test_invalid_hidden_cells_do_not_contribute():
    logits = torch.randn(1, 2, 2, 20, 26, requires_grad=True)
    target = torch.randint(0, 26, (1, 2, 2, 20))
    specified = torch.zeros_like(target, dtype=torch.bool)
    valid = torch.zeros_like(specified)
    loss = hierarchical_completion_loss(logits, target, specified, valid=valid)
    assert float(loss.detach()) == 0.0
    loss.backward()
