"""End-to-end compact kinematic HL stream to Wan control latent."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from hl_control_latent_bridge import HLControlLatentBridge
from hl_wan_adapter import HLCondition
from kinematic_spatializer import (hl_centers, posterior_moments,
                                   relaxed_hl_sample, sample_within_hl_cell,
                                   spatialize_bimanual)


@dataclass(frozen=True)
class KinematicHLCondition:
    posterior: Tensor          # [B,F,2,20,26]
    bone_length: Tensor        # [B,F,2,20]
    root_xy: Tensor            # [B,F,2,2]
    palm_rotation: Tensor      # [B,F,2,3,3]
    projection_scale: Tensor | float
    residual: Tensor | None = None
    confidence: Tensor | None = None


class KinematicHLControlLatentBridge(nn.Module):
    """Spatialize compact HL kinematics, then predict Wan's control latent."""

    def __init__(self, token_width: int = 128, latent_channels: int = 16,
                 line_samples: int = 5, spatial_mode: str = "mean",
                 sample_temperature: float = 0.5,
                 within_cell: bool = False) -> None:
        super().__init__()
        if spatial_mode not in {"mean", "sample"}:
            raise ValueError("spatial_mode must be 'mean' or 'sample'")
        self.spatial_mode = spatial_mode
        self.sample_temperature = sample_temperature
        self.within_cell = within_cell
        self.register_buffer("direction_codebook", hl_centers(torch.device("cpu"),
                                                               torch.float32))
        self.bridge = HLControlLatentBridge(token_width, latent_channels, 40,
                                            line_samples)

    def forward(self, condition: KinematicHLCondition, latent_height: int,
                latent_width: int) -> Tensor:
        if condition.posterior.ndim != 5 or condition.posterior.shape[-3:] != (2, 20, 26):
            raise ValueError("posterior must have shape [B,F,2,20,26]")
        mean_direction, concentration, _ = posterior_moments(
            condition.posterior, self.direction_codebook.to(condition.posterior))
        if self.spatial_mode == "mean":
            direction = mean_direction
        elif self.within_cell:
            direction = sample_within_hl_cell(
                condition.posterior, self.direction_codebook.to(condition.posterior),
                self.sample_temperature)
        else:
            direction = relaxed_hl_sample(
                condition.posterior, self.direction_codebook.to(condition.posterior),
                self.sample_temperature, hard=True)
        child_xy, parent_xy = spatialize_bimanual(
            direction, condition.bone_length, condition.root_xy,
            condition.palm_rotation, condition.projection_scale)
        batch, frames = condition.posterior.shape[:2]
        residual = (None if condition.residual is None else
                    condition.residual.reshape(batch, frames, 40))
        confidence = (concentration.reshape(batch, frames, 40)
                      if condition.confidence is None else
                      condition.confidence.reshape(batch, frames, 40))
        bridge_condition = HLCondition(
            posterior=condition.posterior.reshape(batch, frames, 40, 26),
            joint_xy=child_xy,
            parent_xy=parent_xy,
            residual=residual,
            confidence=confidence,
        )
        return self.bridge(bridge_condition, latent_height, latent_width)

    @property
    def parameter_count(self) -> int:
        return self.bridge.parameter_count
