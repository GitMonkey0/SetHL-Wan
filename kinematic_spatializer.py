"""Differentiable kinematic spatialization without transmitting every joint XY."""

from __future__ import annotations

import torch
from torch import Tensor


HL_PARENTS = (0, 0, 1, 2, 3, 9, 5, 6, 7, 0, 9, 10, 11,
              9, 13, 14, 15, 13, 17, 18, 19)


def hl_centers(device: torch.device, dtype: torch.dtype) -> Tensor:
    values = torch.tensor([(x, y, z) for x in (-1.0, 0.0, 1.0)
                           for y in (-1.0, 0.0, 1.0)
                           for z in (-1.0, 0.0, 1.0)
                           if (x, y, z) != (0.0, 0.0, 0.0)],
                          device=device, dtype=dtype)
    return torch.nn.functional.normalize(values, dim=-1)


def expected_hl_direction(posterior: Tensor, codebook: Tensor) -> Tensor:
    """Posterior mean direction, normalized back to the unit sphere."""
    if posterior.shape[-1] != codebook.shape[0] or codebook.shape[-1] != 3:
        raise ValueError("posterior/codebook shape mismatch")
    return torch.nn.functional.normalize(posterior @ codebook, dim=-1)


def posterior_moments(posterior: Tensor, codebook: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Return spherical mean direction, concentration, and 3x3 covariance."""
    if posterior.shape[-1] != codebook.shape[0] or codebook.shape[-1] != 3:
        raise ValueError("posterior/codebook shape mismatch")
    mean = posterior @ codebook
    concentration = torch.linalg.vector_norm(mean, dim=-1)
    direction = torch.nn.functional.normalize(mean, dim=-1)
    second = torch.einsum("...k,ki,kj->...ij", posterior, codebook, codebook)
    covariance = second - mean[..., :, None] * mean[..., None, :]
    return direction, concentration, covariance


def support_preserving_log_probabilities(posterior: Tensor) -> Tensor:
    """Convert probabilities to logits without reviving zero-probability symbols."""
    if not posterior.is_floating_point():
        raise ValueError("posterior must be floating point")
    if bool((posterior < 0).any()) or bool((posterior.sum(-1) <= 0).any()):
        raise ValueError("posterior must be nonnegative with nonempty support")
    safe = posterior.clamp_min(torch.finfo(posterior.dtype).tiny).log()
    return safe.masked_fill(posterior == 0, -torch.inf)


def relaxed_hl_sample(posterior: Tensor, codebook: Tensor, temperature: float = 0.5,
                      hard: bool = False) -> Tensor:
    """Differentiable categorical direction sample for uncertainty training."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    weights = torch.nn.functional.gumbel_softmax(
        support_preserving_log_probabilities(posterior),
        tau=temperature, hard=hard, dim=-1)
    return torch.nn.functional.normalize(weights @ codebook, dim=-1)


def sample_within_hl_cell(posterior: Tensor, codebook: Tensor,
                          temperature: float = 0.5,
                          radius_fraction: float = 0.9) -> Tensor:
    """Sample continuously inside the selected HL symbol's spherical cell.

    A hard straight-through Gumbel sample chooses the symbol.  The sampled
    direction is then perturbed in its tangent plane by less than half the
    selected centre's nearest-neighbour angle.  Consequently its nearest HL
    centre is guaranteed to remain the selected symbol (away from ties), while
    gradients still reach the categorical posterior.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0 < radius_fraction < 1:
        raise ValueError("radius_fraction must lie in (0, 1)")
    if posterior.shape[-1] != codebook.shape[0] or codebook.shape[-1] != 3:
        raise ValueError("posterior/codebook shape mismatch")
    weights = torch.nn.functional.gumbel_softmax(
        support_preserving_log_probabilities(posterior),
        tau=temperature, hard=True, dim=-1)
    centre = torch.nn.functional.normalize(weights @ codebook, dim=-1)

    similarity = (codebook @ codebook.T).clamp(-1.0, 1.0)
    similarity.fill_diagonal_(-1.0)
    nearest_angle = similarity.max(dim=-1).values.acos()
    max_angle = (weights @ nearest_angle) * (0.5 * radius_fraction)

    # Construct a stable tangent basis from the coordinate axis least aligned
    # with the selected centre.  Projecting an unconstrained random vector can
    # become arbitrarily close to zero and yielded non-finite NPU gradients.
    axis = torch.nn.functional.one_hot(
        centre.detach().abs().argmin(dim=-1), 3).to(centre)
    tangent_x = torch.nn.functional.normalize(
        torch.linalg.cross(centre, axis, dim=-1), dim=-1, eps=1e-6)
    tangent_y = torch.linalg.cross(centre, tangent_x, dim=-1)
    azimuth = torch.rand_like(max_angle) * (2.0 * torch.pi)
    tangent = (azimuth.cos()[..., None] * tangent_x +
               azimuth.sin()[..., None] * tangent_y)
    # Stay away from the zero-radius pole, whose spherical parameterization has
    # an undefined azimuthal derivative, while remaining inside the cell.
    theta = max_angle * (0.1 + 0.9 * torch.rand_like(max_angle))
    return torch.nn.functional.normalize(
        theta.cos()[..., None] * centre + theta.sin()[..., None] * tangent, dim=-1)


def spatialize_bimanual(direction: Tensor, bone_length: Tensor, root_xy: Tensor,
                        palm_rotation: Tensor, projection_scale: Tensor | float
                        ) -> tuple[Tensor, Tensor]:
    """Return child and parent XY for 40 bones from a compact kinematic stream.

    Args:
        direction: `[B,F,2,20,3]` local unit directions.
        bone_length: `[B,F,2,20]` metric or normalized lengths.
        root_xy: `[B,F,2,2]` wrist locations in normalized image coordinates.
        palm_rotation: `[B,F,2,3,3]` local-to-camera rotations.
        projection_scale: scalar, `[B,F,2]`, or `[B,F,2,2]` XY scale from
            length units to normalized image coordinates (orthographic prototype).
    """
    if direction.shape[-3:] != (2, 20, 3):
        raise ValueError("direction must have shape [B,F,2,20,3]")
    if bone_length.shape != direction.shape[:-1]:
        raise ValueError("bone_length must have shape [B,F,2,20]")
    if root_xy.shape != direction.shape[:2] + (2, 2):
        raise ValueError("root_xy must have shape [B,F,2,2]")
    if palm_rotation.shape != direction.shape[:3] + (3, 3):
        raise ValueError("palm_rotation must have shape [B,F,2,3,3]")
    world = torch.einsum("bfhij,bfhkj->bfhki", palm_rotation, direction)
    scale = torch.as_tensor(projection_scale, device=direction.device,
                            dtype=direction.dtype)
    while scale.ndim < 3:
        scale = scale.unsqueeze(0)
    if scale.ndim == 3:
        scale = scale[..., None]
    if scale.ndim != 4 or scale.shape[-1] not in (1, 2):
        raise ValueError("projection_scale must be scalar, [B,F,2], or [B,F,2,2]")
    delta = world[..., :2] * bone_length[..., None] * scale[..., None, :]
    joints: list[Tensor | None] = [None] * 21
    joints[0] = root_xy
    remaining = set(range(1, 21))
    while remaining:
        ready = [child for child in sorted(remaining) if joints[HL_PARENTS[child]] is not None]
        if not ready:
            raise RuntimeError("invalid kinematic topology")
        for child in ready:
            joints[child] = joints[HL_PARENTS[child]] + delta[..., child - 1, :]
            remaining.remove(child)
    child_xy = torch.stack([joints[child] for child in range(1, 21)], dim=-2)
    parent_xy = torch.stack([joints[HL_PARENTS[child]] for child in range(1, 21)], dim=-2)
    # Fold the hand dimension into the bone dimension used by the bridge.
    return child_xy.flatten(2, 3), parent_xy.flatten(2, 3)
