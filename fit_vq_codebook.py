#!/usr/bin/env python3
"""Fit a 26-entry directional VQ baseline on training videos only."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hl_programs import joints_to_hl  # noqa: E402
from train_sethl_wan import fill_missing  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--latents", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=2027)
    p.add_argument("--max-directions", type=int, default=500000)
    args = p.parse_args()
    rows = [json.loads(x) for x in (args.latents / "manifest.jsonl").read_text().splitlines()]
    rows = [x for x in rows if x["split"] == "train"]
    rng = np.random.default_rng(args.seed); rng.shuffle(rows)
    chunks, count = [], 0
    for row in rows:
        sample = torch.load(args.latents / row["path"], weights_only=True)
        joints = fill_missing(sample["joints"].unsqueeze(0).float(),
                              sample["visible"].unsqueeze(0).float())
        directions = joints_to_hl(joints).local_direction.reshape(-1, 3).numpy()
        chunks.append(directions); count += len(directions)
        if count >= args.max_directions: break
    data = np.concatenate(chunks)[:args.max_directions]
    kmeans = MiniBatchKMeans(n_clusters=26, random_state=args.seed, batch_size=8192,
                             n_init=10, max_iter=300).fit(data)
    centers = kmeans.cluster_centers_.astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True).clip(1e-8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"codebook": torch.from_numpy(centers), "seed": args.seed,
                "training_directions": len(data), "training_videos": len(rows)}, args.output)
    print(json.dumps({"output": str(args.output), "directions": len(data),
                      "inertia": float(kmeans.inertia_)}, indent=2))


if __name__ == "__main__": main()
