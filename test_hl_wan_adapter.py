import torch

from hl_wan_adapter import HLCondition, HLWanAdapter


def sample_condition(batch=2, frames=5, bones=40, codes=26):
    logits = torch.randn(batch, frames, bones, codes)
    return HLCondition(
        posterior=logits.softmax(-1),
        joint_xy=torch.rand(batch, frames, bones, 2) * 2 - 1,
        residual=torch.rand(batch, frames, bones),
        confidence=torch.rand(batch, frames, bones),
    )


def test_shapes_and_zero_init():
    model = HLWanAdapter(injection_layers=(1, 3), wan_width=96, token_width=32)
    motion = model.encode(sample_condition())
    video = torch.randn(2, 80, 96)
    assert motion.shape == (2, 200, 32)
    assert torch.equal(model.residual(0, video, motion), torch.zeros_like(video))
    assert torch.equal(model.residual(1, video, motion), torch.zeros_like(video))


def test_gradient_reaches_adapter_then_encoder():
    model = HLWanAdapter(injection_layers=(1,), wan_width=96, token_width=32)
    condition = sample_condition(batch=1)
    video = torch.randn(1, 20, 96)
    motion = model.encode(condition)
    model.residual(1, video, motion).sum().backward()
    assert model.adapters["1"].output.weight.grad.abs().sum() > 0
    # A zero output projection deliberately shields the encoder on step zero.
    assert model.encoder.codebook.grad is None or model.encoder.codebook.grad.abs().sum() == 0

    with torch.no_grad():
        model.adapters["1"].output.weight.normal_(std=1e-3)
    model.zero_grad(set_to_none=True)
    model.residual(1, video, model.encode(condition)).square().mean().backward()
    assert model.encoder.codebook.grad.abs().sum() > 0


def test_parameter_budget_for_wan_1_3b():
    model = HLWanAdapter()
    assert model.parameter_count < 10_000_000
    assert model.parameter_count / 1_300_000_000 < 0.008
