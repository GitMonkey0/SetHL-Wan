"""Thin joint-training wrapper; keeps the vendored Wan implementation untouched."""

from __future__ import annotations

from torch import Tensor, nn

from kinematic_hl_bridge import KinematicHLCondition, KinematicHLControlLatentBridge
from wan_training_contract import assemble_control_y, assert_wan_input_contract


class HLWanTrainingModule(nn.Module):
    """Feed the learned kinematic control latent through Wan's native `y` path.

    The caller is responsible for freezing the base checkpoint and attaching
    LoRA to selected Wan modules before constructing this wrapper.
    """

    def __init__(self, wan: nn.Module, bridge: KinematicHLControlLatentBridge,
                 expected_in_channels: int = 48) -> None:
        super().__init__()
        self.wan = wan
        self.bridge = bridge
        self.expected_in_channels = expected_in_channels

    def forward(self, noisy_latent: Tensor, first_frame_latent: Tensor,
                condition: KinematicHLCondition, timesteps: Tensor,
                context, seq_len: int, **wan_kwargs):
        control = self.bridge(condition, noisy_latent.shape[-2], noisy_latent.shape[-1])
        y = assemble_control_y(control.to(noisy_latent), first_frame_latent)
        assert_wan_input_contract(noisy_latent, y, self.expected_in_channels)
        return self.wan(x=noisy_latent, t=timesteps, context=context,
                        seq_len=seq_len, y=y, **wan_kwargs)
