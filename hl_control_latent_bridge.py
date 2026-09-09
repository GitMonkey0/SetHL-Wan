"""HL-to-Wan control-latent bridge used by the recommended two-stage design."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from hl_wan_adapter import HLCondition, HLTokenEncoder


def bilinear_splat(features: Tensor, xy: Tensor, height: int, width: int) -> Tensor:
    """Splat ``[B,F,J,C]`` tokens to ``[B,C,F,H,W]`` with bilinear weights."""
    batch, frames, joints, channels = features.shape
    px = (xy[..., 0].clamp(-1, 1) + 1) * (width - 1) / 2
    py = (xy[..., 1].clamp(-1, 1) + 1) * (height - 1) / 2
    x0, y0 = px.floor().long(), py.floor().long()
    x1, y1 = (x0 + 1).clamp_max(width - 1), (y0 + 1).clamp_max(height - 1)
    wx, wy = px - x0, py - y0

    canvas = features.new_zeros(batch, frames, height * width, channels)
    mass = features.new_zeros(batch, frames, height * width, 1)
    for x, y, weight in (
        (x0, y0, (1 - wx) * (1 - wy)),
        (x1, y0, wx * (1 - wy)),
        (x0, y1, (1 - wx) * wy),
        (x1, y1, wx * wy),
    ):
        weight = weight.to(features.dtype)
        index = (y * width + x)[..., None]
        canvas.scatter_add_(2, index.expand(-1, -1, -1, channels), features * weight[..., None])
        mass.scatter_add_(2, index, weight[..., None])
    canvas = canvas / mass.clamp_min(1.0)
    return canvas.view(batch, frames, height, width, channels).permute(0, 4, 1, 2, 3)


class HLControlLatentBridge(nn.Module):
    """Predict the 16-channel VAE control latent expected by Wan-Control.

    Stage 1 minimizes distance to frozen-VAE latents of skeleton control videos.
    Stage 2 jointly optimizes this bridge and Wan LoRA with flow matching.
    """

    def __init__(self, token_width: int = 128, latent_channels: int = 16,
                 bones: int = 40, line_samples: int = 5) -> None:
        super().__init__()
        if line_samples < 1:
            raise ValueError("line_samples must be positive")
        self.line_samples = line_samples
        self.encoder = HLTokenEncoder(width=token_width, depth=2, heads=4, bones=bones)
        self.background = nn.Parameter(torch.zeros(1, token_width, 1, 1, 1))
        self.decoder = nn.Sequential(
            nn.Conv3d(token_width, token_width, 3, padding=1),
            nn.GroupNorm(8, token_width),
            nn.SiLU(),
            nn.Conv3d(token_width, token_width // 2, 3, padding=1),
            nn.GroupNorm(8, token_width // 2),
            nn.SiLU(),
            nn.Conv3d(token_width // 2, latent_channels, 3, padding=1),
        )
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, condition: HLCondition, latent_height: int, latent_width: int) -> Tensor:
        batch, frames, bones = condition.posterior.shape[:3]
        tokens = self.encoder(condition).view(batch, frames, bones, -1)
        # Confidence is also a hard spatial visibility gate.  Merely feeding it
        # as a token feature still splats absent-hand embeddings onto the image.
        if condition.confidence is not None:
            tokens = tokens * condition.confidence[..., None]
        if condition.parent_xy is None or self.line_samples == 1:
            volume = bilinear_splat(tokens, condition.joint_xy, latent_height, latent_width)
        else:
            if condition.parent_xy.shape != condition.joint_xy.shape:
                raise ValueError("parent_xy must have shape [B,F,J,2]")
            alpha = torch.linspace(0, 1, self.line_samples, device=tokens.device,
                                   dtype=tokens.dtype).view(1, 1, 1, -1, 1)
            xy = condition.parent_xy[..., None, :] * (1 - alpha) + \
                condition.joint_xy[..., None, :] * alpha
            xy = xy.flatten(2, 3)
            line_tokens = tokens[..., None, :].expand(-1, -1, -1, self.line_samples, -1)
            line_tokens = line_tokens.flatten(2, 3)
            volume = bilinear_splat(line_tokens, xy, latent_height, latent_width)
        volume = volume + self.background
        # Wan's VAE maps 4n+1 RGB frames to n+1 latent frames.  Explicit
        # interpolation avoids an unsupported strided-Conv3D backward kernel
        # on Ascend while preserving the exact temporal contract.
        latent_frames = (frames - 1) // 4 + 1
        volume = torch.nn.functional.interpolate(
            volume, size=(latent_frames, latent_height, latent_width),
            mode="trilinear", align_corners=False)
        return self.decoder(volume)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
