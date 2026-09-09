#!/usr/bin/env python3
"""Pretrain masked hand-program completion before joint Wan adaptation."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch_npu  # noqa: F401

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from hl_programs import apply_partial_mask, joints_to_hl, structured_mask
from masked_continuous_completion import MaskedContinuousCompletion, continuous_completion_loss
from masked_hl_completion import MaskedHLCompletion, hierarchical_completion_loss
from train_sethl_wan import fill_missing


def load_landmarks(directory: Path, split: str):
    rows = [json.loads(x) for x in (directory / "manifest.jsonl").read_text().splitlines()]
    rows = [x for x in rows if x.get("split") == split]
    samples = [torch.load(directory / row["path"], weights_only=True) for row in rows]
    return (torch.stack([x["joints"] for x in samples]),
            torch.stack([x["visible"] for x in samples]))


def batch_masks(batch: int, frames: int, device: torch.device,
                generator: torch.Generator) -> torch.Tensor:
    policies = ("distal", "finger", "interval")
    return torch.stack([structured_mask(frames, policies[int(torch.randint(
        0, len(policies), (), generator=generator, device=device))], device=device,
        generator=generator) for _ in range(batch * 2)]).view(
            batch, 2, frames, 20).transpose(1, 2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--latents", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--representation", choices=("sethl", "vq", "continuous"), required=True)
    p.add_argument("--codebook", type=Path)
    p.add_argument("--device", default="npu:0")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--angular-weight", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=2027)
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.device.startswith("npu"):
        torch.npu.set_device(args.device)
        torch.npu.manual_seed_all(args.seed)
    device = torch.device(args.device)
    torch.manual_seed(args.seed); random.seed(args.seed)
    joints, visible = load_landmarks(args.latents, "train")
    codebook = (torch.load(args.codebook, weights_only=True)["codebook"].to(device)
                if args.representation == "vq" else None)
    model = (MaskedContinuousCompletion() if args.representation == "continuous"
             else MaskedHLCompletion()).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    losses = []
    for step in range(args.steps):
        ids = torch.randint(0, len(joints), (args.batch_size,), generator=generator,
                            device=device).cpu()
        batch_joints = joints[ids].to(device).float()
        batch_visible = visible[ids].to(device).float()
        batch_joints = fill_missing(batch_joints, batch_visible)
        program = joints_to_hl(batch_joints, codebook=codebook)
        masks = batch_masks(args.batch_size, batch_joints.shape[1], device, generator)
        masks &= batch_visible[..., None].bool()
        valid = batch_visible[..., None].expand_as(masks)
        if args.representation == "continuous":
            observed = torch.where(masks[..., None], program.local_direction,
                                   torch.zeros_like(program.local_direction))
            _, mean, std = model(observed, masks, sample=False)
            loss = continuous_completion_loss(mean, std, program.local_direction,
                                              masks, valid=valid)
        else:
            partial = apply_partial_mask(program, masks)
            _, logits = model(partial.posterior, masks)
            loss = hierarchical_completion_loss(logits, program.symbol, masks,
                                                 angular_weight=args.angular_weight,
                                                 valid=valid, codebook=codebook)
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if step == 0 or (step + 1) % 100 == 0:
            print(json.dumps({"step": step + 1, "loss": losses[-1]}), flush=True)
    torch.save({"completion": model.state_dict(), "representation": args.representation,
                "steps": args.steps, "seed": args.seed,
                "angular_weight": args.angular_weight}, args.output)
    print(json.dumps({"status": "complete", "steps": args.steps,
                      "final_loss": losses[-1],
                      "last_100_mean": sum(losses[-100:]) / min(100, len(losses))}, indent=2))


if __name__ == "__main__":
    main()
