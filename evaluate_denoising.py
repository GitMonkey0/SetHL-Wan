#!/usr/bin/env python3
"""Paired held-out evaluation for SetHL/continuous Wan checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from diffusers import FlowMatchEulerDiscreteScheduler
from peft import LoraConfig, inject_adapter_in_model

ROOT = Path(__file__).resolve().parents[1]
VIDEOX_FUN_ROOT = Path(os.environ.get("VIDEOX_FUN_ROOT", ROOT / "vendor" / "VideoX-Fun"))
for path in (ROOT / "vendor" / "python", VIDEOX_FUN_ROOT, ROOT):
    sys.path.insert(0, str(path))

from hl_programs import apply_partial_mask, joints_to_hl, structured_mask  # noqa: E402
from kinematic_continuous_bridge import KinematicContinuousCondition, KinematicContinuousControlLatentBridge  # noqa: E402
from kinematic_hl_bridge import KinematicHLCondition, KinematicHLControlLatentBridge  # noqa: E402
from kinematic_spatializer import hl_centers, posterior_moments  # noqa: E402
from masked_continuous_completion import MaskedContinuousCompletion  # noqa: E402
from masked_hl_completion import MaskedHLCompletion  # noqa: E402
from raster_skeleton_bridge import (RasterSkeletonCondition,
                                    RasterSkeletonControlLatentBridge)  # noqa: E402
from train_sethl_wan import fill_missing  # noqa: E402
from videox_fun.models import WanTransformer3DModel  # noqa: E402
from wan_training_contract import assemble_control_y, flow_matching_batch  # noqa: E402


def json_safe(value):
    """Recursively replace non-finite floats with strict-JSON null values."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def angular_degrees(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = torch.nn.functional.normalize(a, dim=-1)
    b = torch.nn.functional.normalize(b, dim=-1)
    return torch.rad2deg(torch.acos((a * b).sum(-1).clamp(-1 + 1e-6, 1 - 1e-6)))


def load_modules(args, device):
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    wan = WanTransformer3DModel.from_pretrained(str(args.model)).to(dtype=torch.bfloat16)
    wan.requires_grad_(False)
    wan = inject_adapter_in_model(LoraConfig(
        r=args.rank, lora_alpha=args.rank,
        target_modules=["q", "k", "v", "ffn.0", "ffn.2"], lora_dropout=0.0), wan)
    wan.load_state_dict(checkpoint["wan_trainable"], strict=False)
    if args.representation in {"sethl", "sethl_center", "hardhl", "sethl_nohier", "vq"}:
        completion = MaskedHLCompletion()
        bridge = KinematicHLControlLatentBridge(spatial_mode="mean")
        if args.representation == "vq":
            learned = torch.load(args.codebook, weights_only=True)["codebook"]
            bridge.direction_codebook.copy_(learned)
    elif args.representation == "continuous":
        completion = MaskedContinuousCompletion()
        bridge = KinematicContinuousControlLatentBridge()
    else:
        completion = None
        bridge = RasterSkeletonControlLatentBridge()
    if completion is not None:
        completion.load_state_dict(checkpoint["completion"])
    bridge.load_state_dict(checkpoint["bridge"])
    metadata = {"training_seed": checkpoint.get("seed"),
                "training_step": checkpoint.get("step")}
    return (wan.to(device).eval(), None if completion is None else completion.to(device).eval(),
            bridge.to(device).eval(), metadata)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latents", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path(os.environ.get(
        "WAN_MODEL_PATH", ROOT / "models/Wan2.1-Fun-V1.1-1.3B-Control")))
    parser.add_argument("--representation", choices=("sethl", "sethl_center", "hardhl", "sethl_nohier", "vq", "continuous", "raster"), required=True)
    parser.add_argument("--codebook", type=Path)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--split", default="test")
    parser.add_argument("--mask-policy", choices=("full", "distal", "finger", "interval"), default="interval")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20270909)
    parser.add_argument("--joint-noise", type=float, default=0.0,
                        help="Gaussian landmark-coordinate noise applied only to control input")
    parser.add_argument("--compare-zero-control", action="store_true",
                        help="also evaluate the same noise with a zero control latent")
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    device = torch.device(args.device)
    torch.manual_seed(args.seed); random.seed(args.seed); torch.npu.manual_seed_all(args.seed)
    wan, completion, bridge, checkpoint_metadata = load_modules(args, device)
    rows = [json.loads(line) for line in (args.latents / "manifest.jsonl").read_text().splitlines()]
    rows = [row for row in rows if row.get("split") == args.split]
    if args.limit: rows = rows[:args.limit]
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=5.0)
    records = []
    codebook = (torch.load(args.codebook, weights_only=True)["codebook"].to(device)
                if args.representation == "vq" else hl_centers(device, torch.float32))
    with torch.no_grad():
        for sample_index, row in enumerate(rows):
            sample = torch.load(args.latents / row["path"], weights_only=True)
            clean = sample["clean_latent"].to(device=device, dtype=torch.bfloat16)
            first = sample["first_frame_latent"].to(device=device, dtype=torch.bfloat16)
            joints = fill_missing(sample["joints"].unsqueeze(0).to(device).float(),
                                  sample["visible"].unsqueeze(0).to(device).float())
            visible = sample["visible"].unsqueeze(0).to(device).float()
            truth = joints_to_hl(joints, codebook=codebook)
            control_gen = torch.Generator(device=device).manual_seed(
                args.seed + 2000000 + sample_index)
            control_joints = joints + args.joint_noise * torch.randn(
                joints.shape, generator=control_gen, device=device, dtype=joints.dtype)
            control_program = joints_to_hl(control_joints, codebook=codebook)
            frames = joints.shape[1]
            mask_gen = torch.Generator(device=device).manual_seed(args.seed + sample_index)
            masks = torch.stack([structured_mask(frames, args.mask_policy, device=device,
                                                  generator=mask_gen) for _ in range(2)])
            masks = masks.view(1, 2, frames, 20).transpose(1, 2)
            masks = masks & (visible[..., None] > .5)
            if args.representation in {"sethl", "sethl_center", "hardhl", "sethl_nohier", "vq"}:
                partial = apply_partial_mask(control_program, masks)
                posterior, _ = completion(partial.posterior, partial.specified)
                if args.representation == "hardhl":
                    posterior = torch.nn.functional.one_hot(
                        posterior.argmax(-1), 26).to(posterior)
                direction, concentration, covariance = posterior_moments(posterior, codebook)
                evaluation_direction = direction
                condition = KinematicHLCondition(
                    posterior, control_program.bone_length,
                    control_joints[..., 0, :2] * 2 - 1,
                    control_program.palm_rotation, torch.tensor(2., device=device),
                    confidence=visible[..., None].expand_as(masks).float())
                variance = covariance.diagonal(dim1=-2, dim2=-1).sum(-1)
                uncertainty = float(variance[~masks].mean().cpu())
            elif args.representation == "continuous":
                observed = torch.where(masks[..., None], control_program.local_direction,
                                       torch.zeros_like(control_program.local_direction))
                sample_gen = torch.Generator(device=device).manual_seed(args.seed + 100000 + sample_index)
                direction, mean, _ = completion(observed, masks, sample=True, generator=sample_gen)
                evaluation_direction = torch.where(masks[..., None],
                                                   truth.local_direction, mean)
                condition = KinematicContinuousCondition(
                    direction, control_program.bone_length,
                    control_joints[..., 0, :2] * 2 - 1,
                    control_program.palm_rotation, torch.tensor(2., device=device),
                    confidence=visible[..., None].expand_as(masks).float())
                uncertainty = float("nan")
            else:
                condition = RasterSkeletonCondition(
                    control_joints[..., :2] * 2.0 - 1.0, masks)
                uncertainty = float("nan")
            valid = visible[..., None].expand_as(masks) > .5
            if args.representation != "raster":
                angular = angular_degrees(evaluation_direction, truth.local_direction)
            with torch.autocast(device_type="npu", dtype=torch.bfloat16):
                control = bridge(condition, clean.shape[-2], clean.shape[-1]).to(torch.bfloat16)
            y = assemble_control_y(control, first)
            flow_values = []
            zero_flow_values = []
            for train_index in (100, 500, 900):
                noise_gen = torch.Generator(device=device).manual_seed(
                    args.seed + 1000000 * sample_index + train_index)
                noise = torch.randn(clean.shape, generator=noise_gen, device=device,
                                    dtype=torch.bfloat16)
                sigma = scheduler.sigmas[train_index].to(device=device, dtype=torch.bfloat16)
                noisy, target = flow_matching_batch(clean, sigma, noise)
                timestep = scheduler.timesteps[train_index].view(1).to(device)
                seq_len = math.ceil(clean.shape[-3] * clean.shape[-2] * clean.shape[-1] / 4)
                with torch.autocast(device_type="npu", dtype=torch.bfloat16):
                    prediction = wan(x=noisy, t=timestep,
                                     context=torch.zeros(1, 1, 4096, device=device, dtype=torch.bfloat16),
                                     seq_len=seq_len, y=y,
                                     clip_fea=torch.zeros(1, 257, 1280, device=device,
                                                          dtype=torch.bfloat16))
                flow_values.append(float(torch.nn.functional.mse_loss(
                    prediction.float(), target.float()).cpu()))
                if args.compare_zero_control:
                    zero_y = assemble_control_y(torch.zeros_like(control), first)
                    with torch.autocast(device_type="npu", dtype=torch.bfloat16):
                        zero_prediction = wan(
                            x=noisy, t=timestep, context=torch.zeros(
                                1, 1, 4096, device=device, dtype=torch.bfloat16),
                            seq_len=seq_len, y=zero_y,
                            clip_fea=torch.zeros(1, 257, 1280, device=device,
                                                 dtype=torch.bfloat16))
                    zero_flow_values.append(float(torch.nn.functional.mse_loss(
                        zero_prediction.float(), target.float()).cpu()))
            record = {"id": row["id"],
                      "flow_mse": sum(flow_values) / len(flow_values)}
            if args.compare_zero_control:
                record["zero_control_flow_mse"] = sum(zero_flow_values) / len(zero_flow_values)
                record["control_gain"] = record["zero_control_flow_mse"] - record["flow_mse"]
            if args.representation != "raster":
                record.update({
                    "hidden_angular_deg": float(angular[(~masks) & valid].mean().cpu()),
                    "observed_angular_deg": float(angular[masks & valid].mean().cpu()),
                    "hidden_uncertainty": uncertainty,
                })
            records.append(record)
            print(json.dumps(json_safe(records[-1]), allow_nan=False), flush=True)
    keys = (("flow_mse",) if args.representation == "raster" else
            ("flow_mse", "hidden_angular_deg", "observed_angular_deg"))
    if args.compare_zero_control:
        keys += ("zero_control_flow_mse", "control_gain")
    report = {"representation": args.representation, "split": args.split,
              **checkpoint_metadata,
              "joint_noise": args.joint_noise, "mask_policy": args.mask_policy,
              "samples": len(records), "per_sample": records,
              "means": {key: sum(row[key] for row in records) / len(records) for key in keys}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = json_safe(report)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "per_sample"},
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
