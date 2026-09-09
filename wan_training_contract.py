"""Device-agnostic tensor contract for joint HL-bridge/Wan training."""

from __future__ import annotations

import torch
from torch import Tensor


def assemble_control_y(control_latent: Tensor, first_frame_latent: Tensor) -> Tensor:
    """Return Wan-Control's 32-channel ``y`` tensor.

    Wan concatenates this tensor with the 16-channel noisy video latent before
    its 48-channel patch embedding. Both inputs must already share the VAE
    temporal and spatial grid.
    """
    if control_latent.ndim != 5 or first_frame_latent.ndim != 5:
        raise ValueError("control and first-frame latents must be [B,C,T,H,W]")
    if control_latent.shape != first_frame_latent.shape:
        raise ValueError("control and first-frame latents must have identical shapes")
    if control_latent.shape[1] != 16:
        raise ValueError("Wan-Control expects 16 channels for each latent branch")
    return torch.cat((control_latent, first_frame_latent), dim=1)


def flow_matching_batch(clean_latent: Tensor, sigmas: Tensor,
                        noise: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Construct the exact interpolation and velocity target used by Wan."""
    if noise is None:
        noise = torch.randn_like(clean_latent)
    if noise.shape != clean_latent.shape:
        raise ValueError("noise and clean latent must have identical shapes")
    while sigmas.ndim < clean_latent.ndim:
        sigmas = sigmas.unsqueeze(-1)
    noisy = (1.0 - sigmas) * clean_latent + sigmas * noise
    target = noise - clean_latent
    return noisy, target


def assert_wan_input_contract(noisy_latent: Tensor, y: Tensor,
                              expected_in_channels: int = 48) -> None:
    if noisy_latent.ndim != 5 or y.ndim != 5:
        raise ValueError("Wan inputs must be [B,C,T,H,W]")
    if noisy_latent.shape[0] != y.shape[0] or noisy_latent.shape[2:] != y.shape[2:]:
        raise ValueError("Wan noisy and condition latents must share batch/grid")
    actual = noisy_latent.shape[1] + y.shape[1]
    if actual != expected_in_channels:
        raise ValueError(f"expected {expected_in_channels} concatenated channels, got {actual}")
