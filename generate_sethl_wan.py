#!/usr/bin/env python3
"""Generate reproducible Wan videos from partial SetHL or continuous programs."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch_npu  # noqa: F401
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
VIDEOX_FUN_ROOT = Path(os.environ.get("VIDEOX_FUN_ROOT", ROOT / "vendor/VideoX-Fun"))
for path in (ROOT / "vendor/python", VIDEOX_FUN_ROOT, ROOT):
    sys.path.insert(0, str(path))

from hl_programs import apply_partial_mask, joints_to_hl, structured_mask  # noqa: E402
from kinematic_continuous_bridge import KinematicContinuousCondition  # noqa: E402
from kinematic_hl_bridge import KinematicHLCondition  # noqa: E402
from masked_continuous_completion import MaskedContinuousCompletion  # noqa: E402
from masked_hl_completion import MaskedHLCompletion  # noqa: E402
from raster_skeleton_bridge import RasterSkeletonCondition  # noqa: E402
from wan_training_contract import assemble_control_y  # noqa: E402


def calibrate_posterior(posterior: torch.Tensor, specified: torch.Tensor,
                        temperature: float) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("posterior temperature must be positive")
    if temperature == 1.0:
        return posterior
    calibrated = torch.softmax(posterior.clamp_min(1e-12).log() / temperature, dim=-1)
    return torch.where(specified[..., None], posterior, calibrated)


def write_video(frames: np.ndarray, path: Path, fps: float = 8.0):
    # frames [3,F,H,W] in [0,1]
    path.parent.mkdir(parents=True, exist_ok=True)
    _, count, height, width = frames.shape
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (width, height))
    if not writer.isOpened(): raise RuntimeError(f"cannot open video writer {path}")
    for index in range(count):
        rgb = np.clip(frames[:, index].transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
        writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    writer.release()


def main():
    # Heavy model imports stay local so lightweight utilities such as posterior
    # calibration remain importable before VideoX-Fun is installed.
    from evaluate_denoising import load_modules
    from train_sethl_wan import fill_missing
    from videox_fun.models import AutoencoderKLWan
    parser = argparse.ArgumentParser()
    parser.add_argument("--latents", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--representation", choices=("sethl", "sethl_center", "hardhl", "sethl_nohier", "vq", "continuous", "raster"), required=True)
    parser.add_argument("--codebook", type=Path)
    parser.add_argument("--model", type=Path, default=Path(os.environ.get(
        "WAN_MODEL_PATH", ROOT / "models/Wan2.1-Fun-V1.1-1.3B-Control")))
    parser.add_argument("--config", type=Path, default=VIDEOX_FUN_ROOT / "config/wan2.1/wan_civitai.yaml")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--split", default="test")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--mask-policy", choices=("full", "distal", "finger", "interval"), default="interval")
    parser.add_argument("--inference-steps", type=int, default=30)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--control-scales", type=float, nargs="+", default=[1.0])
    parser.add_argument("--posterior-temperature", type=float, default=1.0,
                        help="categorical completion temperature; specified cells stay one-hot")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.npu.set_device(args.device); device = torch.device(args.device)
    wan, completion, bridge, checkpoint_metadata = load_modules(args, device)
    cfg = OmegaConf.load(args.config)
    vae = AutoencoderKLWan.from_pretrained(
        str(args.model / cfg.vae_kwargs.vae_subpath),
        additional_kwargs=OmegaConf.to_container(cfg.vae_kwargs)).to(device).eval()
    rows = [json.loads(line) for line in (args.latents / "manifest.jsonl").read_text().splitlines()]
    rows = [row for row in rows if row.get("split") == args.split]
    row = rows[args.sample_index]
    sample = torch.load(args.latents / row["path"], weights_only=True)
    clean = sample["clean_latent"].to(device=device, dtype=torch.bfloat16)
    first = sample["first_frame_latent"].to(device=device, dtype=torch.bfloat16)
    visible = sample["visible"].unsqueeze(0).to(device).float()
    joints = fill_missing(sample["joints"].unsqueeze(0).to(device).float(), visible)
    codebook = (torch.load(args.codebook, weights_only=True)["codebook"].to(device)
                if args.representation == "vq" else None)
    truth = joints_to_hl(joints, codebook=codebook); frames = joints.shape[1]
    mask_gen = torch.Generator(device=device).manual_seed(20270909 + args.sample_index)
    masks = torch.stack([structured_mask(frames, args.mask_policy, device=device,
                                          generator=mask_gen) for _ in range(2)])
    masks = masks.view(1, 2, frames, 20).transpose(1, 2)
    masks = masks & (visible[..., None] > .5)
    with torch.no_grad():
        if args.representation in {"sethl", "sethl_center", "hardhl", "sethl_nohier", "vq"}:
            partial = apply_partial_mask(truth, masks)
            posterior, _ = completion(partial.posterior, partial.specified)
            if args.posterior_temperature != 1.0 and args.representation != "hardhl":
                posterior = calibrate_posterior(
                    posterior, masks, args.posterior_temperature)
            if args.representation == "hardhl":
                posterior = torch.nn.functional.one_hot(
                    posterior.argmax(-1), 26).to(posterior)
            # Training samples the set-valued control.  At inference, draw one
            # compatible hidden articulation per output; specified cells are
            # one-hot and therefore remain invariant across samples.
            bridge.spatial_mode = "sample"
            bridge.within_cell = args.representation in {"sethl", "sethl_nohier"}
        elif args.representation == "continuous":
            observed = torch.where(masks[..., None], truth.local_direction,
                                   torch.zeros_like(truth.local_direction))
        scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=5.0)
        seq_len = math.ceil(clean.shape[-3] * clean.shape[-2] * clean.shape[-1] / 4)
        outputs = []
        for control_scale in args.control_scales:
          controls = []
          for seed in args.seeds:
            # Decouple the control-completion draw from diffusion noise while
            # preserving paired diffusion seeds across representations.
            torch.manual_seed(seed + 1_000_000)
            torch.npu.manual_seed_all(seed + 1_000_000)
            if args.representation in {"sethl", "sethl_center", "hardhl", "sethl_nohier", "vq"}:
                condition = KinematicHLCondition(
                    posterior, truth.bone_length, joints[..., 0, :2] * 2 - 1,
                    truth.palm_rotation, torch.tensor(2., device=device),
                    confidence=visible[..., None].expand_as(masks).float())
            elif args.representation == "continuous":
                completion_gen = torch.Generator(device=device).manual_seed(
                    seed + 1_000_000)
                direction, _, _ = completion(observed, masks, sample=True,
                                               generator=completion_gen)
                condition = KinematicContinuousCondition(
                    direction, truth.bone_length, joints[..., 0, :2] * 2 - 1,
                    truth.palm_rotation, torch.tensor(2., device=device),
                    confidence=visible[..., None].expand_as(masks).float())
            else:
                condition = RasterSkeletonCondition(
                    joints[..., :2] * 2.0 - 1.0, masks)
            with torch.autocast(device_type="npu", dtype=torch.bfloat16):
                controls.append(bridge(
                    condition, clean.shape[-2], clean.shape[-1]).to(torch.bfloat16))
          batch = len(args.seeds)
          control = torch.cat(controls, dim=0)
          first_batch = first.expand(batch, *first.shape[1:])
          y = assemble_control_y(control * control_scale, first_batch)
          scheduler.set_timesteps(args.inference_steps, device=device)
          latents = torch.cat([
              torch.randn(clean.shape,
                          generator=torch.Generator(device=device).manual_seed(seed),
                          device=device, dtype=torch.bfloat16)
              for seed in args.seeds], dim=0)
          context = torch.zeros(batch, 1, 4096, device=device, dtype=torch.bfloat16)
          clip = torch.zeros(batch, 257, 1280, device=device, dtype=torch.bfloat16)
          for timestep in scheduler.timesteps:
              model_input = (scheduler.scale_model_input(latents, timestep)
                             if hasattr(scheduler, "scale_model_input") else latents)
              with torch.autocast(device_type="npu", dtype=torch.bfloat16):
                  prediction = wan(x=model_input, t=timestep.expand(batch), context=context,
                                   seq_len=seq_len, y=y, clip_fea=clip)
              latents = scheduler.step(prediction, timestep, latents,
                                       return_dict=False)[0]
          decoded = vae.decode(latents.to(dtype=vae.dtype)).sample
          decoded = (decoded / 2 + .5).clamp(0, 1).cpu().float().numpy()
          for batch_index, seed in enumerate(args.seeds):
            scale_tag = str(control_scale).replace(".", "p")
            path = args.output / f"{row['id']}_scale{scale_tag}_seed{seed}.mp4"
            write_video(decoded[batch_index], path)
            outputs.append({"seed": seed, "control_scale": control_scale,
                            "video": path.name})
            print(json.dumps(outputs[-1]), flush=True)
    (args.output / "generation.json").write_text(json.dumps({
        "representation": args.representation, "source": row["id"],
        **checkpoint_metadata,
        "checkpoint": str(args.checkpoint),
        "mask_policy": args.mask_policy, "specified_fraction": float(masks.float().mean().cpu()),
        "posterior_temperature": args.posterior_temperature,
        "specified_mask": masks.cpu().tolist(),
        "inference_steps": args.inference_steps, "outputs": outputs}, indent=2) + "\n")


if __name__ == "__main__":
    main()
