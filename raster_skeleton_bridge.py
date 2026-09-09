"""Matched-capacity direct 2-D skeleton baseline for Wan control."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from hl_control_latent_bridge import bilinear_splat
from kinematic_spatializer import HL_PARENTS


@dataclass(frozen=True)
class RasterSkeletonCondition:
    joints_xy: Tensor  # [B,F,2,21,2], normalized to [-1,1]
    specified: Tensor  # [B,F,2,20]


class RasterSkeletonControlLatentBridge(nn.Module):
    """Rasterize observed 2-D bones and lift them to Wan's 16 channels."""

    def __init__(self, width: int = 192, token_width: int = 128,
                 max_frames: int = 81, line_samples: int = 5):
        super().__init__(); self.max_frames = max_frames; self.line_samples = line_samples
        self.coordinate = nn.Linear(4, width)
        self.bone = nn.Embedding(20, width); self.hand = nn.Embedding(2, width)
        self.time = nn.Embedding(max_frames, width); self.mask = nn.Embedding(2, width)
        layer = nn.TransformerEncoderLayer(width, 6, 4 * width, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, 4)
        self.project = nn.Linear(width, token_width)
        self.background = nn.Parameter(torch.zeros(1, token_width, 1, 1, 1))
        self.decoder = nn.Sequential(
            nn.Conv3d(token_width, token_width, 3, padding=1), nn.GroupNorm(8, token_width), nn.SiLU(),
            nn.Conv3d(token_width, token_width // 2, 3, padding=1),
            nn.GroupNorm(8, token_width // 2), nn.SiLU(),
            nn.Conv3d(token_width // 2, 16, 3, padding=1))
        nn.init.zeros_(self.decoder[-1].weight); nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, condition: RasterSkeletonCondition, latent_height: int,
                latent_width: int) -> Tensor:
        joints, specified = condition.joints_xy, condition.specified
        batch, frames = joints.shape[:2]
        parents = torch.as_tensor(HL_PARENTS, device=joints.device)
        child = joints[..., 1:, :]; parent = joints[..., parents[1:], :]
        # Hidden cells must not affect any token, including through global
        # self-attention.  Zero their coordinates before the encoder rather
        # than only suppressing their raster contribution afterwards.
        endpoints = torch.cat((parent, child), -1)
        endpoints = torch.where(specified[..., None], endpoints,
                                torch.zeros_like(endpoints))
        token = self.coordinate(endpoints)
        token = token + self.time(torch.arange(frames, device=joints.device))[None, :, None, None]
        token = token + self.hand(torch.arange(2, device=joints.device))[None, None, :, None]
        token = token + self.bone(torch.arange(20, device=joints.device))[None, None, None]
        token = token + self.mask(specified.long())
        token = self.project(self.encoder(token.reshape(batch, frames * 40, -1)))
        token = token.reshape(batch, frames, 40, -1) * specified.flatten(2, 3)[..., None]
        child = child.flatten(2, 3); parent = parent.flatten(2, 3)
        alpha = torch.linspace(0, 1, self.line_samples, device=joints.device,
                               dtype=joints.dtype).view(1, 1, 1, -1, 1)
        xy = (parent[..., None, :] * (1 - alpha) + child[..., None, :] * alpha).flatten(2, 3)
        token = token[..., None, :].expand(-1, -1, -1, self.line_samples, -1).flatten(2, 3)
        volume = bilinear_splat(token, xy, latent_height, latent_width) + self.background
        latent_frames = (frames - 1) // 4 + 1
        volume = torch.nn.functional.interpolate(volume,
            size=(latent_frames, latent_height, latent_width), mode="trilinear",
            align_corners=False)
        return self.decoder(volume)

    @property
    def parameter_count(self): return sum(p.numel() for p in self.parameters())
