import torch
from torch import nn

from hl_wan_adapter import HLCondition, HLWanAdapter
from wan_adapter_wrapper import HLConditionedWan, ResidualInjectedBlock


class ToyBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.linear = nn.Linear(width, width)

    def forward(self, x):
        return self.linear(x)


class ToyWan(nn.Module):
    def __init__(self, width=32, depth=4):
        super().__init__()
        self.blocks = nn.ModuleList(ToyBlock(width) for _ in range(depth))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


def condition():
    ids = torch.randint(0, 26, (2, 5, 40))
    return HLCondition(
        posterior=torch.nn.functional.one_hot(ids, 26).float(),
        joint_xy=torch.rand(2, 5, 40, 2),
    )


def test_wrap_freezes_backbone_and_preserves_initial_output():
    torch.manual_seed(7)
    base = ToyWan()
    reference = ToyWan()
    reference.load_state_dict(base.state_dict())
    adapter = HLWanAdapter(injection_layers=(1, 3), wan_width=32, token_width=32)
    model = HLConditionedWan(base, adapter)
    x = torch.randn(2, 11, 32)
    assert torch.equal(model(x, hl_condition=condition()), reference(x))
    assert all(not p.requires_grad for p in model.backbone.parameters())
    assert all(p.requires_grad for p in model.adapter.parameters())
    assert isinstance(model.backbone.blocks[1], ResidualInjectedBlock)


def test_only_adapter_accumulates_gradients():
    model = HLConditionedWan(
        ToyWan(), HLWanAdapter(injection_layers=(1,), wan_width=32, token_width=32)
    )
    model(torch.randn(2, 11, 32), hl_condition=condition()).sum().backward()
    assert all(p.grad is None for p in model.backbone.parameters())
    assert model.adapter.adapters["1"].output.weight.grad.abs().sum() > 0
