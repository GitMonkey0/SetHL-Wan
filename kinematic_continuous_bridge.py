"""Coordinate-bypass-free continuous baseline matched to the HL bridge."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from continuous_control_latent_bridge import ContinuousCondition, ContinuousControlLatentBridge
from kinematic_spatializer import spatialize_bimanual


@dataclass(frozen=True)
class KinematicContinuousCondition:
    direction: Tensor       # [B,F,2,20,3]
    bone_length: Tensor     # [B,F,2,20]
    root_xy: Tensor         # [B,F,2,2]
    palm_rotation: Tensor   # [B,F,2,3,3]
    projection_scale: Tensor | float
    confidence: Tensor | None = None


class KinematicContinuousControlLatentBridge(nn.Module):
    def __init__(self, token_width: int = 128, latent_channels: int = 16,
                 line_samples: int = 5) -> None:
        super().__init__()
        self.bridge = ContinuousControlLatentBridge(
            token_width=token_width, latent_channels=latent_channels,
            bones=40, line_samples=line_samples)

    def forward(self, condition: KinematicContinuousCondition,
                latent_height: int, latent_width: int) -> Tensor:
        child, parent = spatialize_bimanual(
            condition.direction, condition.bone_length, condition.root_xy,
            condition.palm_rotation, condition.projection_scale)
        batch, frames = condition.direction.shape[:2]
        flat = ContinuousCondition(
            direction=condition.direction.reshape(batch, frames, 40, 3),
            joint_xy=child,
            parent_xy=parent,
            confidence=(None if condition.confidence is None else
                        condition.confidence.reshape(batch, frames, 40)),
        )
        return self.bridge(flat, latent_height, latent_width)

    @property
    def parameter_count(self) -> int:
        return self.bridge.parameter_count

