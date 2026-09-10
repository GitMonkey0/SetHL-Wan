#!/usr/bin/env python3
"""Create a content-disjoint WLASL latent view using exact RGB hashes.

WLASL contains identical source videos under multiple gloss entries.  To make
the generative evaluation strictly content-disjoint, one entry is retained per
decoded 17-frame RGB tensor.  Test entries have priority over validation and
training entries, so evaluation coverage is preserved; ties are resolved by
the lexicographically smallest clip id.  Latent tensors are hard-linked rather
than copied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import torch


PRIORITY = {"test": 0, "val": 1, "train": 2}


def rgb_hash(path: Path) -> str:
    sample = torch.load(path, map_location="cpu", weights_only=True)
    rgb = sample["rgb"].contiguous().cpu().numpy()
    return hashlib.sha256(rgb.tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--latents", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.source_manifest.read_text().splitlines()
            if line.strip()]
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[rgb_hash(args.windows / row["path"])].append(row)

    kept = [min(group, key=lambda row: (PRIORITY[row["split"]], str(row["id"])))
            for group in groups.values()]
    kept.sort(key=lambda row: str(row["id"]))
    keep_ids = {str(row["id"]) for row in kept}

    latent_rows = [json.loads(line)
                   for line in (args.latents / "manifest.jsonl").read_text().splitlines()
                   if line.strip()]
    latent_by_id = {str(row["id"]): row for row in latent_rows}
    if set(latent_by_id) != {str(row["id"]) for row in rows}:
        raise ValueError("source and latent manifests do not contain the same ids")

    args.output.mkdir(parents=True, exist_ok=True)
    output_rows = []
    for clip_id in sorted(keep_ids):
        row = latent_by_id[clip_id]
        source = args.latents / row["path"]
        target = args.output / row["path"]
        if target.exists():
            if not os.path.samefile(source, target):
                raise FileExistsError(f"non-matching target exists: {target}")
        else:
            os.link(source, target)
        output_rows.append(row)

    manifest = "".join(json.dumps(row, sort_keys=True) + "\n" for row in output_rows)
    (args.output / "manifest.jsonl").write_text(manifest)
    report = {
        "policy": "exact decoded-RGB SHA-256; split priority test, val, train; id tie-break",
        "input_rows": len(rows),
        "unique_rgb": len(groups),
        "removed_rows": len(rows) - len(groups),
        "kept_by_split": dict(sorted(Counter(row["split"] for row in kept).items())),
        "manifest_sha256": hashlib.sha256(manifest.encode()).hexdigest(),
    }
    (args.output / "deduplication_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
