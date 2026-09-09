"""Parameter-efficient structured hand-motion conditioning for Wan.

This module is intentionally independent of the frozen HL-Repair repository.  It
turns per-bone HL posteriors and compact geometric side information into tokens
that can be injected into selected Wan DiT blocks without rasterizing a skeleton
video.  The integration hook is kept small so it can be audited before modifying
the vendored Wan implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class HLCondition:
    """A batch of rate-scalable hand-control tokens.

    Shapes are ``posterior [B,F,J,K]``, ``joint_xy [B,F,J,2]``, and optional
    scalar fields ``[B,F,J]``.  ``posterior`` may be one-hot (hard HL) or soft.
    """

    posterior: Tensor
    joint_xy: Tensor
    parent_xy: Tensor | None = None
    residual: Tensor | None = None
    confidence: Tensor | None = None


def sinusoidal_time(frames: int, width: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    if width % 2:
        raise ValueError("time width must be even")
    position = torch.arange(frames, device=device, dtype=torch.float32)[:, None]
    scale = torch.exp(
        torch.arange(0, width, 2, device=device, dtype=torch.float32)
        * (-math.log(10_000.0) / width)
    )
    encoding = torch.cat((torch.sin(position * scale), torch.cos(position * scale)), dim=-1)
    return encoding.to(dtype=dtype)


class HLTokenEncoder(nn.Module):
    """Encode semantic direction distributions plus geometry into motion tokens."""

    def __init__(
        self,
        codebook_size: int = 26,
        bones: int = 40,
        width: int = 256,
        depth: int = 3,
        heads: int = 8,
    ) -> None:
        super().__init__()
        self.codebook_size = codebook_size
        self.bones = bones
        self.width = width
        self.codebook = nn.Parameter(torch.empty(codebook_size, width))
        nn.init.normal_(self.codebook, std=0.02)
        self.bone_embedding = nn.Embedding(bones, width)
        self.geometry = nn.Sequential(nn.Linear(5, width), nn.SiLU(), nn.Linear(width, width))
        layer = nn.TransformerEncoderLayer(
            width, heads, 4 * width, dropout=0.0, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(width)

    def forward(self, condition: HLCondition) -> Tensor:
        posterior, xy = condition.posterior, condition.joint_xy
        if posterior.ndim != 4 or posterior.shape[-1] != self.codebook_size:
            raise ValueError("posterior must have shape [B,F,J,K]")
        if xy.shape != posterior.shape[:3] + (2,):
            raise ValueError("joint_xy must have shape [B,F,J,2]")
        batch, frames, bones, _ = posterior.shape
        if bones != self.bones:
            raise ValueError(f"expected {self.bones} bones, got {bones}")

        residual = posterior.new_zeros(batch, frames, bones) if condition.residual is None else condition.residual
        confidence = posterior.new_ones(batch, frames, bones) if condition.confidence is None else condition.confidence
        if residual.shape != posterior.shape[:3] or confidence.shape != posterior.shape[:3]:
            raise ValueError("residual and confidence must have shape [B,F,J]")

        semantic = posterior @ self.codebook
        bone_ids = torch.arange(bones, device=posterior.device)
        bone = self.bone_embedding(bone_ids)[None, None]
        time = sinusoidal_time(frames, self.width, posterior.device, posterior.dtype)[None, :, None]
        entropy = -(posterior.clamp_min(1e-8) * posterior.clamp_min(1e-8).log()).sum(-1)
        entropy = entropy / math.log(self.codebook_size)
        geometry = torch.cat(
            (xy, residual[..., None], confidence[..., None], entropy[..., None]), dim=-1
        )
        tokens = semantic + bone + time + self.geometry(geometry)
        tokens = tokens.reshape(batch, frames * bones, self.width)
        return self.norm(self.temporal(tokens))


class HLWanResidualAdapter(nn.Module):
    """Low-rank cross-attention residual for selected frozen Wan blocks.

    ``video_tokens`` are Wan hidden states ``[B,N,D]``. The output has the same
    shape and starts as an exact zero, preserving the pretrained model before
    adapter training.
    """

    def __init__(self, wan_width: int = 1536, token_width: int = 256, heads: int = 8) -> None:
        super().__init__()
        self.video_norm = nn.LayerNorm(wan_width)
        self.token_norm = nn.LayerNorm(token_width)
        self.query = nn.Linear(wan_width, token_width, bias=False)
        self.attention = nn.MultiheadAttention(token_width, heads, batch_first=True)
        self.output = nn.Linear(token_width, wan_width, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, video_tokens: Tensor, motion_tokens: Tensor) -> Tensor:
        query = self.query(self.video_norm(video_tokens))
        key_value = self.token_norm(motion_tokens)
        attended, _ = self.attention(query, key_value, key_value, need_weights=False)
        return self.output(attended)


class HLWanAdapter(nn.Module):
    """Shared HL encoder and one zero-init residual adapter per injection point."""

    def __init__(
        self,
        injection_layers: tuple[int, ...] = (4, 9, 14, 19, 24, 29),
        wan_width: int = 1536,
        token_width: int = 256,
        bones: int = 40,
    ) -> None:
        super().__init__()
        self.injection_layers = injection_layers
        self.encoder = HLTokenEncoder(width=token_width, bones=bones)
        self.adapters = nn.ModuleDict({
            str(layer): HLWanResidualAdapter(wan_width, token_width)
            for layer in injection_layers
        })

    def encode(self, condition: HLCondition) -> Tensor:
        return self.encoder(condition)

    def residual(self, layer: int, video_tokens: Tensor, motion_tokens: Tensor) -> Tensor:
        if str(layer) not in self.adapters:
            return torch.zeros_like(video_tokens)
        return self.adapters[str(layer)](video_tokens, motion_tokens)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
