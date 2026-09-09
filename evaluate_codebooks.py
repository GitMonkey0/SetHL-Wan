#!/usr/bin/env python3
"""Compare fixed HL and learned-VQ direction quantizers on held-out videos."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hl_programs import joints_to_hl  # noqa: E402
from kinematic_spatializer import hl_centers  # noqa: E402
from train_sethl_wan import fill_missing  # noqa: E402


def errors(direction, codebook):
    assignment = (direction @ codebook.T).argmax(-1)
    cosine = (direction * codebook[assignment]).sum(-1).clamp(-1, 1)
    return assignment, torch.rad2deg(torch.acos(cosine))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--latents", type=Path, required=True)
    p.add_argument("--vq", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", default="test")
    args = p.parse_args()
    rows = [json.loads(x) for x in (args.latents / "manifest.jsonl").read_text().splitlines()]
    rows = [x for x in rows if x["split"] == args.split]
    books = {"HL": hl_centers(torch.device("cpu"), torch.float32),
             "VQ": torch.load(args.vq, weights_only=True)["codebook"]}
    result = {name: {"angular": [], "stability": {str(s): [] for s in (.0025, .005, .01)}}
              for name in books}
    generator = torch.Generator().manual_seed(2027)
    for row in rows:
        s = torch.load(args.latents / row["path"], weights_only=True)
        j = fill_missing(s["joints"].unsqueeze(0).float(), s["visible"].unsqueeze(0).float())
        direction = joints_to_hl(j).local_direction.reshape(-1, 3)
        noisy_directions = {}
        for sigma in (.0025, .005, .01):
            noisy = j + sigma * torch.randn(j.shape, generator=generator)
            noisy_directions[sigma] = joints_to_hl(noisy).local_direction.reshape(-1, 3)
        for name, book in books.items():
            base, angle = errors(direction, book)
            result[name]["angular"].extend(angle.tolist())
            for sigma in (.0025, .005, .01):
                changed, _ = errors(noisy_directions[sigma], book)
                result[name]["stability"][str(sigma)].append(float((changed == base).float().mean()))
    report = {name: {
        "mean_quantization_angle_deg": float(np.mean(values["angular"])),
        "p95_quantization_angle_deg": float(np.percentile(values["angular"], 95)),
        "symbol_stability": {s: float(np.mean(v)) for s, v in values["stability"].items()},
    } for name, values in result.items()}
    report["samples"] = len(rows); report["split"] = args.split
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
