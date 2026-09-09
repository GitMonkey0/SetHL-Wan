"""Matched continuous-direction baseline for the HL control-latent bridge."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from hl_control_latent_bridge import bilinear_splat
from hl_wan_adapter import sinusoidal_time


@dataclass(frozen=True)
class ContinuousCondition:
    direction: Tensor          # [B,F,J,3], unit vectors in the palm frame
    joint_xy: Tensor           # [B,F,J,2], normalized image coordinates
    parent_xy: Tensor | None = None
    residual: Tensor | None = None
    confidence: Tensor | None = None


def hl_centers(device: torch.device, dtype: torch.dtype) -> Tensor:
    values = torch.tensor([(x, y, z) for x in (-1.0, 0.0, 1.0)
                           for y in (-1.0, 0.0, 1.0)
                           for z in (-1.0, 0.0, 1.0)
                           if (x, y, z) != (0.0, 0.0, 0.0)],
                          device=device, dtype=dtype)
    return torch.nn.functional.normalize(values, dim=-1)


class ContinuousTokenEncoder(nn.Module):
    """Continuous baseline with the same semantic-bank capacity as HL.

    Raw unit directions are lifted to their 26 cosine similarities. Unlike the
    HL branch, no argmax, cell posterior, or residual quantization is applied.
    """

    def __init__(self, bones: int = 40, width: int = 128,
                 depth: int = 2, heads: int = 4) -> None:
        super().__init__()
        self.bones, self.width = bones, width
        self.semantic_bank = nn.Parameter(torch.empty(26, width))
        nn.init.normal_(self.semantic_bank, std=0.02)
        self.bone_embedding = nn.Embedding(bones, width)
        self.geometry = nn.Sequential(nn.Linear(5, width), nn.SiLU(), nn.Linear(width, width))
        layer = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=0.0,
                                           activation="gelu", batch_first=True,
                                           norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(width)

    def forward(self, condition: ContinuousCondition) -> Tensor:
        direction, xy = condition.direction, condition.joint_xy
        if direction.ndim != 4 or direction.shape[-1] != 3:
            raise ValueError("direction must have shape [B,F,J,3]")
        if xy.shape != direction.shape[:3] + (2,):
            raise ValueError("joint_xy must have shape [B,F,J,2]")
        batch, frames, bones, _ = direction.shape
        if bones != self.bones:
            raise ValueError(f"expected {self.bones} bones, got {bones}")
        residual = direction.new_zeros(batch, frames, bones) if condition.residual is None else condition.residual
        confidence = direction.new_ones(batch, frames, bones) if condition.confidence is None else condition.confidence
        cosine = direction @ hl_centers(direction.device, direction.dtype).T
        semantic = cosine @ self.semantic_bank / math.sqrt(26.0)
        bone = self.bone_embedding(torch.arange(bones, device=direction.device))[None, None]
        time = sinusoidal_time(frames, self.width, direction.device, direction.dtype)[None, :, None]
        norm_error = (torch.linalg.vector_norm(direction, dim=-1) - 1).abs()
        geometry = torch.cat((xy, residual[..., None], confidence[..., None],
                              norm_error[..., None]), dim=-1)
        tokens = semantic + bone + time + self.geometry(geometry)
        return self.norm(self.temporal(tokens.reshape(batch, frames * bones, self.width)))


class ContinuousControlLatentBridge(nn.Module):
    def __init__(self, token_width: int = 128, latent_channels: int = 16,
                 bones: int = 40, line_samples: int = 5) -> None:
        super().__init__()
        self.line_samples = line_samples
        self.encoder = ContinuousTokenEncoder(bones, token_width)
        self.background = nn.Parameter(torch.zeros(1, token_width, 1, 1, 1))
        self.decoder = nn.Sequential(
            nn.Conv3d(token_width, token_width, 3, padding=1),
            nn.GroupNorm(8, token_width), nn.SiLU(),
            nn.Conv3d(token_width, token_width // 2, 3, padding=1),
            nn.GroupNorm(8, token_width // 2), nn.SiLU(),
            nn.Conv3d(token_width // 2, latent_channels, 3, padding=1),
        )
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, condition: ContinuousCondition, height: int, width: int) -> Tensor:
        batch, frames, bones = condition.direction.shape[:3]
        tokens = self.encoder(condition).view(batch, frames, bones, -1)
        if condition.parent_xy is None or self.line_samples == 1:
            volume = bilinear_splat(tokens, condition.joint_xy, height, width)
        else:
            alpha = torch.linspace(0, 1, self.line_samples, device=tokens.device,
                                   dtype=tokens.dtype).view(1, 1, 1, -1, 1)
            xy = (condition.parent_xy[..., None, :] * (1 - alpha) +
                  condition.joint_xy[..., None, :] * alpha).flatten(2, 3)
            features = tokens[..., None, :].expand(-1, -1, -1, self.line_samples, -1)
            volume = bilinear_splat(features.flatten(2, 3), xy, height, width)
        volume = volume + self.background
        latent_frames = (frames - 1) // 4 + 1
        volume = torch.nn.functional.interpolate(
            volume, size=(latent_frames, height, width),
            mode="trilinear", align_corners=False)
        return self.decoder(volume)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
