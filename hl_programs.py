"""Convert tracked hand joints into partial Hand-Labanotation programs.

Joint order follows MediaPipe/FreiHAND: wrist, four thumb joints, then four
joints each for index, middle, ring, and little fingers.  The HL palm topology
matches Fig. 3 of Li et al. (MM 2024).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from kinematic_spatializer import HL_PARENTS, hl_centers


@dataclass(frozen=True)
class PartialHLProgram:
    posterior: Tensor       # [..., 20, 26]
    symbol: Tensor          # [..., 20]
    specified: Tensor       # [..., 20], bool
    local_direction: Tensor # [..., 20, 3], retained only as supervision target
    bone_length: Tensor     # [..., 20]
    palm_rotation: Tensor   # [..., 3, 3], local-to-camera


def palm_frame(joints: Tensor) -> Tensor:
    """Construct the right-handed, hand-local frame defined by the HL paper."""
    if joints.shape[-2:] != (21, 3):
        raise ValueError("joints must end in [21,3]")
    y = torch.nn.functional.normalize(joints[..., 9, :] - joints[..., 0, :], dim=-1)
    across = torch.nn.functional.normalize(joints[..., 17, :] - joints[..., 5, :], dim=-1)
    z = torch.nn.functional.normalize(torch.linalg.cross(across, y, dim=-1), dim=-1)
    x = torch.linalg.cross(y, z, dim=-1)
    return torch.stack((x, y, z), dim=-1)


def joints_to_hl(joints: Tensor, temperature: float = 0.12,
                 codebook: Tensor | None = None) -> PartialHLProgram:
    """Return soft HL assignments and kinematic quantities for 3-D joints."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    frame = palm_frame(joints)
    parents = torch.as_tensor(HL_PARENTS, device=joints.device)
    bones = joints[..., 1:, :] - joints[..., parents[1:], :]
    length = torch.linalg.vector_norm(bones, dim=-1)
    world_direction = bones / length[..., None].clamp_min(1e-8)
    local_direction = torch.einsum("...ji,...bi->...bj", frame, world_direction)
    codebook = (hl_centers(joints.device, joints.dtype) if codebook is None else
                codebook.to(device=joints.device, dtype=joints.dtype))
    if codebook.shape != (26, 3):
        raise ValueError("codebook must have shape [26,3]")
    logits = torch.einsum("...bd,kd->...bk", local_direction, codebook) / temperature
    posterior = logits.softmax(dim=-1)
    symbol = logits.argmax(dim=-1)
    specified = torch.isfinite(joints).all(dim=(-1, -2))[..., None].expand_as(symbol)
    return PartialHLProgram(posterior, symbol, specified, local_direction, length, frame)


def apply_partial_mask(program: PartialHLProgram, specified: Tensor,
                       hard_observed: bool = True) -> PartialHLProgram:
    """Build a partial program without leaking sub-cell joint directions.

    A genuine HL interface supplies symbols, not the continuous direction used
    to derive them. Observed cells are therefore one-hot by default; missing
    cells are maximum entropy. ``hard_observed=False`` is a soft-oracle ablation.
    """
    if specified.shape != program.symbol.shape:
        raise ValueError("specified mask must match symbol shape")
    specified = specified.bool() & program.specified
    uniform = torch.full_like(program.posterior, 1.0 / program.posterior.shape[-1])
    observed = (torch.nn.functional.one_hot(program.symbol, 26).to(program.posterior)
                if hard_observed else program.posterior)
    posterior = torch.where(specified[..., None], observed, uniform)
    symbol = torch.where(specified, program.symbol, torch.full_like(program.symbol, -1))
    return PartialHLProgram(posterior, symbol, specified, program.local_direction,
                            program.bone_length, program.palm_rotation)


def structured_mask(num_frames: int, policy: str, *, device: torch.device,
                    generator: torch.Generator | None = None) -> Tensor:
    """Create one of the preregistered `[F,20]` partial-control masks."""
    mask = torch.ones(num_frames, 20, dtype=torch.bool, device=device)
    if policy == "full":
        return mask
    if policy == "distal":
        mask[:, [3, 7, 11, 15, 19]] = False
    elif policy == "finger":
        finger = int(torch.randint(0, 5, (), generator=generator, device=device))
        mask[:, 4 * finger: 4 + 4 * finger] = False
    elif policy == "interval":
        width = max(1, num_frames // 3)
        start = int(torch.randint(0, num_frames - width + 1, (),
                                  generator=generator, device=device))
        mask[start:start + width] = False
    else:
        raise ValueError(f"unknown masking policy: {policy}")
    return mask
