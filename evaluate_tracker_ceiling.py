#!/usr/bin/env python3
"""Estimate the generated-video metric ceiling on re-encoded real clips."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

from evaluate_generated_videos import masked_mean, track_video
from hl_programs import joints_to_hl
from train_sethl_wan import fill_missing


def write_rgb_video(rgb: torch.Tensor, path: Path, fps: float = 8.0) -> None:
    frames, _, height, width = rgb.shape
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open {path}")
    for frame in rgb:
        writer.write(cv2.cvtColor(frame.permute(1, 2, 0).numpy(),
                                  cv2.COLOR_RGB2BGR))
    writer.release()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selected = json.loads(args.subset.read_text())["selected"]
    records = []
    with tempfile.TemporaryDirectory(prefix="sethl_tracker_") as temp:
        temp = Path(temp)
        for row in selected:
            sample = torch.load(args.windows / f"{row['id']}.pt", weights_only=True)
            path = temp / f"{row['id']}.mp4"
            write_rgb_video(sample["rgb"], path, sample.get("fps", 8.0))
            tracked, visible = track_video(path)
            target_visible = sample["visible"].unsqueeze(0).float()
            target_joints = fill_missing(sample["joints"].unsqueeze(0).float(),
                                         target_visible)
            tracked = fill_missing(tracked.unsqueeze(0), visible.unsqueeze(0))
            predicted, target = joints_to_hl(tracked), joints_to_hl(target_joints)
            valid = (visible.unsqueeze(0)[..., None].bool() &
                     target_visible[..., None].bool()).expand_as(target.symbol)
            cosine = (predicted.local_direction * target.local_direction).sum(-1).clamp(-1, 1)
            angle = torch.rad2deg(torch.acos(cosine))
            records.append({
                "source": row["id"],
                "detection_rate": float(visible.mean()),
                "hl_accuracy": masked_mean(predicted.symbol == target.symbol, valid),
                "angular_deg": masked_mean(angle, valid),
            })
    report = {"sources": len(records), "per_source": records,
              "means": {key: float(np.mean([r[key] for r in records]))
                        for key in ("detection_rate", "hl_accuracy", "angular_deg")}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["means"], indent=2))


if __name__ == "__main__":
    main()
