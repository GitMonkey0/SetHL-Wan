"""Structured completion model for partially specified HL programs."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from kinematic_spatializer import hl_centers


class MaskedHLCompletion(nn.Module):
    """Predict distributions for unspecified bone/time cells.

    Observed cells are copied exactly.  Missing cells are completed jointly over
    time and the two 20-bone hand trees, preventing the independent-uniform
    sampling that otherwise creates anatomically implausible fingers.
    """

    def __init__(self, width: int = 192, layers: int = 4, heads: int = 6,
                 max_frames: int = 81, dropout: float = 0.0) -> None:
        super().__init__()
        self.max_frames = max_frames
        self.code_embedding = nn.Parameter(torch.randn(26, width) * width ** -0.5)
        self.bone_embedding = nn.Embedding(20, width)
        self.hand_embedding = nn.Embedding(2, width)
        self.time_embedding = nn.Embedding(max_frames, width)
        self.mask_embedding = nn.Embedding(2, width)
        block = nn.TransformerEncoderLayer(
            width, heads, 4 * width, dropout=dropout, batch_first=True,
            norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(block, layers)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 26)

    def forward(self, observed_posterior: Tensor, specified: Tensor,
                temperature: float = 1.0) -> tuple[Tensor, Tensor]:
        if observed_posterior.ndim != 5 or observed_posterior.shape[-3:] != (2, 20, 26):
            raise ValueError("observed_posterior must be [B,F,2,20,26]")
        if specified.shape != observed_posterior.shape[:-1]:
            raise ValueError("specified must be [B,F,2,20]")
        if observed_posterior.shape[1] > self.max_frames:
            raise ValueError("sequence exceeds max_frames")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        batch, frames = observed_posterior.shape[:2]
        token = observed_posterior @ self.code_embedding
        time = self.time_embedding(torch.arange(frames, device=token.device))[None, :, None, None]
        hand = self.hand_embedding(torch.arange(2, device=token.device))[None, None, :, None]
        bone = self.bone_embedding(torch.arange(20, device=token.device))[None, None, None]
        token = token + time + hand + bone + self.mask_embedding(specified.long())
        token = token.reshape(batch, frames * 40, -1)
        logits = self.head(self.norm(self.encoder(token))).reshape(batch, frames, 2, 20, 26)
        predicted = (logits / temperature).softmax(-1)
        completed = torch.where(specified[..., None], observed_posterior, predicted)
        return completed, logits


def hierarchical_completion_loss(logits: Tensor, target_symbol: Tensor,
                                 specified: Tensor, angular_weight: float = 0.25,
                                 valid: Tensor | None = None,
                                 codebook: Tensor | None = None) -> Tensor:
    """Cross entropy plus expected spherical distance, only on hidden cells."""
    hidden = ~specified.bool()
    if valid is not None:
        if valid.shape != specified.shape:
            raise ValueError("valid must match specified")
        hidden = hidden & valid.bool()
    if not bool(hidden.any()):
        return logits.sum() * 0.0
    flat_logits = logits[hidden]
    flat_target = target_symbol[hidden]
    ce = torch.nn.functional.cross_entropy(flat_logits, flat_target)
    codebook = (hl_centers(logits.device, logits.dtype) if codebook is None else
                codebook.to(device=logits.device, dtype=logits.dtype))
    if codebook.shape != (26, 3):
        raise ValueError("codebook must have shape [26,3]")
    target_direction = codebook[flat_target]
    expected_direction = flat_logits.softmax(-1) @ codebook
    cosine = torch.nn.functional.cosine_similarity(
        expected_direction.float(), target_direction.float(), dim=-1)
    # 1-cos(theta) has the same spherical ordering as acos but avoids the
    # unbounded boundary derivative that produces NaNs in BF16.
    spherical = (1.0 - cosine.clamp(-1.0, 1.0)).mean()
    return ce + angular_weight * spherical
