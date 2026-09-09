"""Continuous directional completion baseline for partial hand control."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class MaskedContinuousCompletion(nn.Module):
    def __init__(self, width: int = 192, layers: int = 4, heads: int = 6,
                 max_frames: int = 81, dropout: float = 0.0) -> None:
        super().__init__()
        self.max_frames = max_frames
        self.input = nn.Linear(3, width)
        self.bone_embedding = nn.Embedding(20, width)
        self.hand_embedding = nn.Embedding(2, width)
        self.time_embedding = nn.Embedding(max_frames, width)
        self.mask_embedding = nn.Embedding(2, width)
        block = nn.TransformerEncoderLayer(
            width, heads, 4 * width, dropout=dropout, batch_first=True,
            norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(block, layers)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 4)  # directional mean and log standard deviation

    def forward(self, observed_direction: Tensor, specified: Tensor,
                sample: bool = True, generator: torch.Generator | None = None
                ) -> tuple[Tensor, Tensor, Tensor]:
        if observed_direction.ndim != 5 or observed_direction.shape[-3:] != (2, 20, 3):
            raise ValueError("observed_direction must be [B,F,2,20,3]")
        batch, frames = observed_direction.shape[:2]
        if specified.shape != observed_direction.shape[:-1]:
            raise ValueError("specified must be [B,F,2,20]")
        if frames > self.max_frames:
            raise ValueError("sequence exceeds max_frames")
        x = self.input(observed_direction)
        x = x + self.time_embedding(torch.arange(frames, device=x.device))[None, :, None, None]
        x = x + self.hand_embedding(torch.arange(2, device=x.device))[None, None, :, None]
        x = x + self.bone_embedding(torch.arange(20, device=x.device))[None, None, None]
        x = x + self.mask_embedding(specified.long())
        raw = self.head(self.norm(self.encoder(x.reshape(batch, frames * 40, -1))))
        raw = raw.reshape(batch, frames, 2, 20, 4)
        mean = torch.nn.functional.normalize(raw[..., :3], dim=-1)
        std = raw[..., 3:4].clamp(-5, 1).exp()
        if sample:
            noise = torch.randn(mean.shape, device=mean.device, dtype=mean.dtype,
                                generator=generator)
            prediction = torch.nn.functional.normalize(mean + std * noise, dim=-1)
        else:
            prediction = mean
        completed = torch.where(specified[..., None], observed_direction, prediction)
        return completed, mean, std


def continuous_completion_loss(mean: Tensor, std: Tensor, target: Tensor,
                               specified: Tensor, valid: Tensor | None = None) -> Tensor:
    hidden = ~specified.bool()
    if valid is not None:
        if valid.shape != specified.shape:
            raise ValueError("valid must match specified")
        hidden = hidden & valid.bool()
    if not bool(hidden.any()):
        return mean.sum() * 0.0
    error = 1.0 - (mean[hidden].float() * target[hidden].float()).sum(-1).clamp(-1, 1)
    variance = std[hidden].float().square().squeeze(-1).clamp_min(1e-5)
    return (error / variance + variance.log()).mean()
