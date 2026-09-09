"""Non-invasive Wan wrapper for HL residual injection.

The wrapper does not edit the vendored VideoX-Fun source. It replaces selected
transformer blocks with transparent wrappers and freezes every backbone
parameter. Adapter-only checkpoints should save ``adapter.state_dict()``.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from hl_wan_adapter import HLCondition, HLWanAdapter


class ResidualInjectedBlock(nn.Module):
    def __init__(self, block: nn.Module, adapter: HLWanAdapter, layer: int) -> None:
        super().__init__()
        self.block = block
        # Avoid registering the shared adapter once per block. The owner wrapper
        # registers it; object.__setattr__ stores only a runtime reference here.
        object.__setattr__(self, "_adapter_ref", adapter)
        self.layer = layer
        self.motion_tokens: Tensor | None = None

    def forward(self, x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        x = self.block(x, *args, **kwargs)
        if self.motion_tokens is None:
            raise RuntimeError("HL condition was not set before Wan forward")
        return x + self._adapter_ref.residual(self.layer, x, self.motion_tokens)


class HLConditionedWan(nn.Module):
    """Wrap a Wan-like backbone exposing a ``blocks`` ModuleList."""

    def __init__(self, backbone: nn.Module, adapter: HLWanAdapter) -> None:
        super().__init__()
        self.backbone = backbone
        self.adapter = adapter
        self.backbone.requires_grad_(False)
        if not hasattr(self.backbone, "blocks"):
            raise TypeError("backbone must expose a blocks ModuleList")
        if max(adapter.injection_layers) >= len(self.backbone.blocks):
            raise ValueError("injection layer exceeds backbone depth")
        self.injected: list[ResidualInjectedBlock] = []
        for layer in adapter.injection_layers:
            wrapped = ResidualInjectedBlock(self.backbone.blocks[layer], adapter, layer)
            self.backbone.blocks[layer] = wrapped
            self.injected.append(wrapped)

    def forward(self, *args: Any, hl_condition: HLCondition, **kwargs: Any) -> Any:
        motion_tokens = self.adapter.encode(hl_condition)
        # Retain the graph for non-reentrant gradient-checkpoint recomputation.
        # Training is intentionally single-forward-at-a-time per module replica.
        for block in self.injected:
            block.motion_tokens = motion_tokens
        return self.backbone(*args, **kwargs)

    def trainable_parameters(self):
        return self.adapter.parameters()

    def adapter_state_dict(self):
        return self.adapter.state_dict()
