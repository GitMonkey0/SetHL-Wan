#!/usr/bin/env python3
"""Fast paired evaluation of the learned partial-control completion heads."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from hl_programs import apply_partial_mask, joints_to_hl, structured_mask
from kinematic_spatializer import hl_centers, posterior_moments
from masked_continuous_completion import MaskedContinuousCompletion
from masked_hl_completion import MaskedHLCompletion
from train_sethl_wan import fill_missing


def angle(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = torch.nn.functional.normalize(a, dim=-1)
    b = torch.nn.functional.normalize(b, dim=-1)
    return torch.rad2deg(torch.acos((a * b).sum(-1).clamp(-1 + 1e-6, 1 - 1e-6)))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--latents", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--representation", choices=("sethl", "hardhl", "sethl_nohier", "vq", "continuous"), required=True)
    p.add_argument("--codebook", type=Path)
    p.add_argument("--mask-policy", choices=("distal", "finger", "interval"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20270909)
    p.add_argument("--split", default="test")
    p.add_argument("--joint-noise", type=float, default=0.0)
    args = p.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if args.representation == "continuous":
        completion = MaskedContinuousCompletion()
    else:
        completion = MaskedHLCompletion()
    completion.load_state_dict(checkpoint["completion"]); completion.eval()
    codebook = (torch.load(args.codebook, weights_only=True)["codebook"]
                if args.representation == "vq" else hl_centers(torch.device("cpu"), torch.float32))
    rows = [json.loads(x) for x in (args.latents / "manifest.jsonl").read_text().splitlines()]
    rows = [x for x in rows if x.get("split") == args.split]
    records = []
    with torch.no_grad():
        for index, row in enumerate(rows):
            sample = torch.load(args.latents / row["path"], weights_only=True)
            visible = sample["visible"].unsqueeze(0).float()
            joints = fill_missing(sample["joints"].unsqueeze(0).float(), visible)
            truth = joints_to_hl(joints, codebook=codebook)
            noise_gen = torch.Generator().manual_seed(args.seed + 1000000 + index)
            control_joints = joints + args.joint_noise * torch.randn(
                joints.shape, generator=noise_gen, dtype=joints.dtype)
            control = joints_to_hl(control_joints, codebook=codebook)
            gen = torch.Generator().manual_seed(args.seed + index)
            masks = torch.stack([structured_mask(joints.shape[1], args.mask_policy,
                                                  device=torch.device("cpu"), generator=gen)
                                 for _ in range(2)]).view(1, 2, joints.shape[1], 20).transpose(1, 2)
            masks &= visible[..., None].bool()
            if args.representation == "continuous":
                observed = torch.where(masks[..., None], control.local_direction,
                                       torch.zeros_like(control.local_direction))
                _, direction, _ = completion(observed, masks, sample=False)
            else:
                partial = apply_partial_mask(control, masks)
                posterior, _ = completion(partial.posterior, masks)
                direction, _, _ = posterior_moments(posterior, codebook)
            hidden = (~masks) & visible[..., None].bool()
            angular = angle(direction, truth.local_direction)
            predicted_symbol = torch.einsum("...d,kd->...k", direction, codebook).argmax(-1)
            records.append({"id": row["id"],
                            "hidden_angular_deg": float(angular[hidden].mean()),
                            "hidden_symbol_accuracy": float((predicted_symbol[hidden] == truth.symbol[hidden]).float().mean())})
    report = {"representation": args.representation, "mask_policy": args.mask_policy,
              "split": args.split,
              "joint_noise": args.joint_noise,
              "samples": len(records), "per_sample": records,
              "means": {key: sum(x[key] for x in records) / len(records)
                        for key in ("hidden_angular_deg", "hidden_symbol_accuracy")}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "per_sample"}, indent=2))


if __name__ == "__main__":
    main()
