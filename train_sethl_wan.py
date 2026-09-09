#!/usr/bin/env python3
"""Train SetHL completion/bridge and Wan LoRA from cached video latents."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
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
from kinematic_continuous_bridge import (KinematicContinuousCondition,
                                           KinematicContinuousControlLatentBridge)  # noqa: E402
from kinematic_hl_bridge import KinematicHLCondition, KinematicHLControlLatentBridge  # noqa: E402
from masked_continuous_completion import (MaskedContinuousCompletion,
                                           continuous_completion_loss)  # noqa: E402
from masked_hl_completion import MaskedHLCompletion, hierarchical_completion_loss  # noqa: E402
from raster_skeleton_bridge import (RasterSkeletonCondition,
                                    RasterSkeletonControlLatentBridge)  # noqa: E402
from videox_fun.models import WanTransformer3DModel  # noqa: E402
from wan_training_contract import assemble_control_y, flow_matching_batch  # noqa: E402


def fill_missing(joints: torch.Tensor, visible: torch.Tensor) -> torch.Tensor:
    """Nearest-fill missing frames; use the other hand only as masked geometry."""
    joints = joints.clone()
    batch, frames = joints.shape[:2]
    for b in range(batch):
        for hand in range(2):
            valid = torch.nonzero(visible[b, :, hand] > 0.5).flatten()
            if len(valid) == 0:
                other = 1 - hand
                other_valid = torch.nonzero(visible[b, :, other] > 0.5).flatten()
                if len(other_valid):
                    joints[b, :, hand] = joints[b, :, other]
                    joints[b, :, hand, :, 0] = 1.0 - joints[b, :, hand, :, 0]
                else:
                    raise ValueError("sample has no visible hand")
                continue
            for f in range(frames):
                if visible[b, f, hand] <= 0.5:
                    nearest = valid[(valid - f).abs().argmin()]
                    joints[b, f, hand] = joints[b, nearest, hand]
    return joints


def make_condition(joints: torch.Tensor, visible: torch.Tensor, policy: str,
                   completion: MaskedHLCompletion, generator: torch.Generator,
                   hard_completion: bool = False, angular_weight: float = 0.25,
                   codebook: torch.Tensor | None = None
                   ) -> tuple[KinematicHLCondition, torch.Tensor]:
    joints = fill_missing(joints, visible)
    program = joints_to_hl(joints, codebook=codebook)
    batch, frames = joints.shape[:2]
    masks = torch.stack([
        structured_mask(frames, policy, device=joints.device, generator=generator)
        for _ in range(batch * 2)
    ]).view(batch, 2, frames, 20).transpose(1, 2)
    masks = masks & (visible[..., None] > 0.5)
    partial = apply_partial_mask(program, masks)
    completed, logits = completion(partial.posterior, partial.specified)
    if hard_completion:
        hard = torch.nn.functional.one_hot(completed.argmax(-1), 26).to(completed)
        completed = hard + completed - completed.detach()
    confidence = visible[..., None].expand(batch, frames, 2, 20)
    condition = KinematicHLCondition(
        posterior=completed,
        bone_length=program.bone_length,
        root_xy=joints[..., 0, :2] * 2.0 - 1.0,
        palm_rotation=program.palm_rotation,
        projection_scale=torch.tensor(2.0, device=joints.device, dtype=joints.dtype),
        confidence=confidence,
    )
    auxiliary = hierarchical_completion_loss(
        logits, program.symbol, partial.specified, angular_weight=angular_weight,
        valid=visible[..., None].expand_as(masks), codebook=codebook)
    return condition, auxiliary


def make_continuous_condition(joints: torch.Tensor, visible: torch.Tensor, policy: str,
                              completion: MaskedContinuousCompletion,
                              generator: torch.Generator
                              ) -> tuple[KinematicContinuousCondition, torch.Tensor]:
    joints = fill_missing(joints, visible)
    program = joints_to_hl(joints)
    batch, frames = joints.shape[:2]
    masks = torch.stack([
        structured_mask(frames, policy, device=joints.device, generator=generator)
        for _ in range(batch * 2)
    ]).view(batch, 2, frames, 20).transpose(1, 2)
    masks = masks & (visible[..., None] > 0.5)
    observed = torch.where(masks[..., None], program.local_direction,
                           torch.zeros_like(program.local_direction))
    completed, mean, std = completion(observed, masks, sample=True, generator=generator)
    confidence = visible[..., None].expand(batch, frames, 2, 20)
    condition = KinematicContinuousCondition(
        direction=completed,
        bone_length=program.bone_length,
        root_xy=joints[..., 0, :2] * 2.0 - 1.0,
        palm_rotation=program.palm_rotation,
        projection_scale=torch.tensor(2.0, device=joints.device, dtype=joints.dtype),
        confidence=confidence,
    )
    auxiliary = continuous_completion_loss(
        mean, std, program.local_direction, masks,
        valid=visible[..., None].expand_as(masks))
    return condition, auxiliary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latents", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path(os.environ.get(
        "WAN_MODEL_PATH", ROOT / "models" / "Wan2.1-Fun-V1.1-1.3B-Control")))
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--split", default="train")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--completion-lr", type=float,
                        help="completion-head LR; defaults to --lr")
    parser.add_argument("--wan-lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20270909)
    parser.add_argument("--mask-policy", choices=("distal", "finger", "interval", "mixed"),
                        default="mixed")
    parser.add_argument("--representation", choices=("sethl", "hardhl", "sethl_nohier", "vq", "continuous", "raster"),
                        default="sethl")
    parser.add_argument("--codebook", type=Path)
    parser.add_argument("--completion-init", type=Path)
    parser.add_argument("--angular-weight", type=float, default=0.25)
    parser.add_argument("--save-every", type=int, default=250)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.npu.manual_seed_all(args.seed)
    torch.npu.set_device(args.device)
    device, dtype = torch.device(args.device), torch.bfloat16

    rows = [json.loads(x) for x in (args.latents / "manifest.jsonl").read_text().splitlines()]
    if rows and "split" in rows[0]:
        rows = [row for row in rows if row["split"] == args.split]
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        raise ValueError("empty latent manifest")
    random.Random(args.seed).shuffle(rows)

    wan = WanTransformer3DModel.from_pretrained(str(args.model)).to(dtype=dtype)
    wan.requires_grad_(False)
    wan = inject_adapter_in_model(LoraConfig(
        r=args.rank, lora_alpha=args.rank,
        target_modules=["q", "k", "v", "ffn.0", "ffn.2"], lora_dropout=0.0), wan)
    wan.enable_gradient_checkpointing()
    wan.to(device).train()
    # Keep the small newly initialized modules and their Adam states in FP32;
    # the frozen Wan backbone and its LoRA matmuls operate in BF16.
    learned_codebook = None
    if args.representation == "vq":
        if args.codebook is None: raise ValueError("--codebook is required for vq")
        learned_codebook = torch.load(args.codebook, weights_only=True)["codebook"].to(device)
    if args.representation in {"sethl", "hardhl", "sethl_nohier", "vq"}:
        completion = MaskedHLCompletion().to(device=device, dtype=torch.float32).train()
        bridge = KinematicHLControlLatentBridge(
            spatial_mode="sample",
            within_cell=args.representation in {"sethl", "sethl_nohier"}).to(
            device=device, dtype=torch.float32).train()
        if learned_codebook is not None:
            bridge.direction_codebook.copy_(learned_codebook)
    elif args.representation == "continuous":
        completion = MaskedContinuousCompletion().to(device=device, dtype=torch.float32).train()
        bridge = KinematicContinuousControlLatentBridge().to(
            device=device, dtype=torch.float32).train()
    else:
        completion = None
        bridge = RasterSkeletonControlLatentBridge().to(
            device=device, dtype=torch.float32).train()
    if args.completion_init is not None:
        if completion is None:
            raise ValueError("raster representation has no completion module")
        initial = torch.load(args.completion_init, map_location="cpu", weights_only=True)
        completion.load_state_dict(initial["completion"])
    wan_parameters = [p for p in wan.parameters() if p.requires_grad]
    completion_parameters = ([] if completion is None else list(completion.parameters()))
    bridge_parameters = list(bridge.parameters())
    new_parameters = completion_parameters + bridge_parameters
    parameters = wan_parameters + new_parameters
    groups = [{"params": wan_parameters, "lr": args.wan_lr},
              {"params": bridge_parameters, "lr": args.lr}]
    if completion_parameters:
        groups.append({"params": completion_parameters,
                       "lr": args.lr if args.completion_lr is None else args.completion_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=5.0)
    rng = torch.Generator(device=device).manual_seed(args.seed)
    losses = []
    started = time.time()

    for step in range(args.steps):
        row = rows[step % len(rows)]
        policy = (random.choice(("distal", "finger", "interval"))
                  if args.mask_policy == "mixed" else args.mask_policy)
        sample = torch.load(args.latents / row["path"], weights_only=True)
        clean = sample["clean_latent"].to(device=device, dtype=dtype)
        first = sample["first_frame_latent"].to(device=device, dtype=dtype)
        joints = sample["joints"].unsqueeze(0).to(device=device, dtype=torch.float32)
        visible = sample["visible"].unsqueeze(0).to(device=device, dtype=torch.float32)
        with torch.autocast(device_type="npu", dtype=dtype):
            if args.representation in {"sethl", "hardhl", "sethl_nohier", "vq"}:
                condition, auxiliary = make_condition(joints, visible, policy,
                                                      completion, rng,
                                                      hard_completion=args.representation == "hardhl",
                                                      angular_weight=(0.0 if args.representation == "sethl_nohier"
                                                                      else args.angular_weight),
                                                      codebook=learned_codebook)
            elif args.representation == "continuous":
                condition, auxiliary = make_continuous_condition(
                    joints, visible, policy, completion, rng)
            else:
                joints = fill_missing(joints, visible)
                frames = joints.shape[1]
                masks = torch.stack([
                    structured_mask(frames, policy, device=device, generator=rng)
                    for _ in range(joints.shape[0] * 2)
                ]).view(joints.shape[0], 2, frames, 20).transpose(1, 2)
                masks = masks & (visible[..., None] > 0.5)
                condition = RasterSkeletonCondition(joints[..., :2] * 2.0 - 1.0, masks)
                auxiliary = joints.sum() * 0.0
            control = bridge(condition, clean.shape[-2], clean.shape[-1]).to(dtype)
        y = assemble_control_y(control, first)

        index = torch.randint(0, len(scheduler.timesteps), (1,), generator=rng,
                              device=device).cpu()
        timestep = scheduler.timesteps[index].to(device)
        sigma = scheduler.sigmas[index].to(device=device, dtype=dtype)
        noisy, target = flow_matching_batch(clean, sigma,
                                             torch.randn(clean.shape, generator=rng,
                                                         device=device, dtype=dtype))
        seq_len = math.ceil(clean.shape[-3] * clean.shape[-2] * clean.shape[-1] / 4)
        context = torch.zeros(1, 1, 4096, device=device, dtype=dtype)
        clip = torch.zeros(1, 257, 1280, device=device, dtype=dtype)
        prediction = wan(x=noisy, t=timestep, context=context, seq_len=seq_len,
                         y=y, clip_fea=clip)
        flow_loss = torch.nn.functional.mse_loss(prediction.float(), target.float())
        loss = flow_loss + 0.1 * auxiliary.float()
        if not bool(loss.isfinite().item()):
            raise FloatingPointError(
                f"non-finite loss at step {step + 1}: flow={float(flow_loss.detach().cpu())}, "
                f"completion={float(auxiliary.detach().cpu())}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        if not all(p.grad is None or bool(torch.isfinite(p.grad).all().item()) for p in parameters):
            raise FloatingPointError(f"non-finite gradient at step {step + 1}")
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if (step + 1) % 10 == 0 or step == 0:
            print(json.dumps({"step": step + 1, "loss": losses[-1],
                              "flow": float(flow_loss.detach().cpu()),
                              "completion": float(auxiliary.detach().cpu())}), flush=True)
        if (step + 1) % args.save_every == 0 or step + 1 == args.steps:
            torch.save({
                "wan_trainable": {k: v.detach().cpu() for k, v in wan.state_dict().items()
                                  if "lora_" in k},
                "completion": None if completion is None else completion.state_dict(),
                "bridge": bridge.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step + 1,
                "seed": args.seed,
            }, args.output / f"checkpoint-{step + 1}.pt")

    torch.npu.synchronize()
    report = {
        "status": "complete", "steps": args.steps, "seed": args.seed,
        "examples": len(rows), "mask_policy": args.mask_policy,
        "representation": args.representation,
        "initial_loss": losses[0], "final_loss": losses[-1],
        "last_20_mean_loss": sum(losses[-20:]) / min(20, len(losses)),
        "seconds": time.time() - started,
        "peak_allocated_bytes": int(torch.npu.max_memory_allocated(device)),
        "trainable_parameters": sum(p.numel() for p in parameters),
        "rank": args.rank, "learning_rate": args.lr,
        "completion_learning_rate": (args.lr if args.completion_lr is None
                                     else args.completion_lr),
        "wan_learning_rate": args.wan_lr,
        "completion_init": (None if args.completion_init is None else
                            str(args.completion_init)),
        "angular_weight": args.angular_weight,
        "latent_manifest_sha256": hashlib.sha256(
            (args.latents / "manifest.jsonl").read_bytes()).hexdigest(),
        "scope": "training run; quantitative generation evaluation required separately",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
