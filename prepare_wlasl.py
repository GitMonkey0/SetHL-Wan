#!/usr/bin/env python3
"""Prepare leakage-free 17-frame WLASL clips with aligned MediaPipe hands.

Each source video contributes at most one window.  The official WLASL split is
retained for comparability; signer/source identifiers are retained so that an
additional identity-disjoint audit can be constructed without decoding again.
MuteMotion V3 stores MediaPipe landmarks as right hand (0:21), left hand
(21:42), pose (42:75), and face (75:553).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch


def square_crop(bbox: list[int], width: int, height: int,
                padding: float) -> tuple[float, float, float]:
    x0, y0, x1, y1 = map(float, bbox)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(x1 - x0, y1 - y0) * (1 + 2 * padding)
    side = max(side, 2.0)
    return cx - side / 2, cy - side / 2, side


def hand_focused_crop(landmarks: np.ndarray, indices: np.ndarray, width: int,
                      height: int, person_bbox: list[int], padding: float
                      ) -> tuple[float, float, float]:
    hands = landmarks[indices, :42]
    valid = np.abs(hands).sum(axis=-1) > 1e-8
    xy = hands[..., :2][valid]
    if len(xy) < 15:
        return square_crop(person_bbox, width, height, padding)
    xy = xy * np.asarray([width, height], dtype=np.float32)
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    center = (lo + hi) / 2
    person_side = max(person_bbox[2] - person_bbox[0],
                      person_bbox[3] - person_bbox[1])
    side = max(float((hi - lo).max()) * (1 + 2 * padding),
               float(person_side) * 0.35, 64.0)
    return float(center[0] - side / 2), float(center[1] - side / 2), side


def decode_frames(path: Path, indices: np.ndarray, crop: tuple[float, float, float],
                  size: int) -> torch.Tensor:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {path}")
    x0, y0, side = crop
    matrix = np.asarray([[size / side, 0, -x0 * size / side],
                         [0, size / side, -y0 * size / side]], dtype=np.float32)
    frames, wanted = [], {int(frame): i for i, frame in enumerate(indices)}
    decoded = {}
    frame_index = 0
    while len(decoded) < len(wanted):
        ok, frame = capture.read()
        if not ok: break
        if frame_index in wanted:
            frame = cv2.warpAffine(frame, matrix, (size, size), flags=cv2.INTER_AREA,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=(0, 0, 0))
            decoded[frame_index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_index += 1
    capture.release()
    missing = sorted(set(wanted) - set(decoded))
    if missing:
        # A subset of legacy WLASL containers reports broken sequential time
        # stamps but remains seek-decodable. Recover those frames explicitly.
        fallback = cv2.VideoCapture(str(path))
        for index in missing:
            fallback.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = fallback.read()
            if not ok: continue
            frame = cv2.warpAffine(frame, matrix, (size, size), flags=cv2.INTER_AREA,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=(0, 0, 0))
            decoded[index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        fallback.release()
    if len(decoded) != len(wanted):
        missing = sorted(set(wanted) - set(decoded))
        raise RuntimeError(f"cannot decode frames {missing} from {path}")
    frames = [torch.from_numpy(decoded[int(index)]).permute(2, 0, 1)
              for index in indices]
    # Keep preparation storage compact; cache_video_latents normalizes uint8.
    return torch.stack(frames)


def transform_hands(landmarks: np.ndarray, indices: np.ndarray, width: int,
                    height: int, crop: tuple[float, float, float]
                    ) -> tuple[torch.Tensor, torch.Tensor]:
    # Output convention follows the rest of this repository: left, then right.
    selected = landmarks[indices]
    hands = np.stack((selected[:, 21:42], selected[:, :21]), axis=1).astype(np.float32)
    magnitude = np.abs(hands).sum(axis=-1)
    visible = (np.count_nonzero(magnitude > 1e-8, axis=-1) >= 15).astype(np.float32)
    x0, y0, side = crop
    hands[..., 0] = (hands[..., 0] * width - x0) / side
    hands[..., 1] = (hands[..., 1] * height - y0) / side
    # MediaPipe z is normalized by image width. Re-express it in crop units.
    hands[..., 2] = hands[..., 2] * width / side
    hands *= visible[..., None, None]
    return torch.from_numpy(hands), torch.from_numpy(visible)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--parsed", type=Path, required=True)
    parser.add_argument("--landmarks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--padding", type=float, default=0.12)
    parser.add_argument("--crop-mode", choices=("person", "hands"), default="person")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--min-visible", type=float, default=0.70)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    parsed = json.loads(args.parsed.read_text())
    original = json.loads(args.metadata.read_text())
    by_video = {instance["video_id"]: {"gloss": group["gloss"], **instance}
                for group in original for instance in group["instances"]}
    landmark_archive = np.load(args.landmarks, allow_pickle=False)
    rows, rejected = [], {}
    # Sort by source-video ID, not the shuffled NPZ key order.
    indexed = sorted(((Path(parsed[int(key)]["video_path"]).stem, key)
                      for key in landmark_archive.files))
    for video_id, key in indexed:
        if args.limit and len(rows) >= args.limit:
            break
        video_path = args.videos / f"{video_id}.mp4"
        meta = by_video[video_id]
        output_path = args.output / f"{video_id}.pt"
        if not video_path.exists():
            rejected["missing_video"] = rejected.get("missing_video", 0) + 1
            continue
        capture = cv2.VideoCapture(str(video_path))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture.release()
        landmark = landmark_archive[key]
        usable = min(count, len(landmark))
        if usable < args.frames or width <= 0 or height <= 0:
            rejected["short_or_unreadable"] = rejected.get("short_or_unreadable", 0) + 1
            continue
        indices = np.rint(np.linspace(0, usable - 1, args.frames)).astype(np.int64)
        crop = (hand_focused_crop(landmark, indices, width, height, meta["bbox"],
                                  args.padding) if args.crop_mode == "hands" else
                square_crop(meta["bbox"], width, height, args.padding))
        joints, visible = transform_hands(landmark, indices, width, height, crop)
        if float(visible.mean(dim=0).max()) < args.min_visible:
            rejected["insufficient_hand_track"] = rejected.get(
                "insufficient_hand_track", 0) + 1
            continue
        row = {
            "id": video_id, "path": output_path.name, "source_clip": video_id,
            "split": meta["split"], "signer_id": int(meta["signer_id"]),
            "source": meta["source"], "gloss": meta["gloss"],
            "frame_indices": indices.tolist(),
            "visible_fraction": visible.mean(dim=0).tolist(),
        }
        if output_path.exists():
            rows.append(row)
            continue
        try:
            rgb = decode_frames(video_path, indices, crop, args.size)
        except RuntimeError:
            rejected["decode_failure"] = rejected.get("decode_failure", 0) + 1
            continue
        torch.save({
            "rgb": rgb, "joints": joints, "visible": visible,
            "source_clip": video_id, "frame_indices": torch.from_numpy(indices),
            "fps": float(meta["fps"]), "signer_id": int(meta["signer_id"]),
            "source": meta["source"], "gloss": meta["gloss"],
            "crop_xy_side": torch.tensor(crop), "license": "C-UDA",
        }, output_path)
        rows.append(row)
        if len(rows) % 100 == 0:
            print(f"prepared {len(rows)}", flush=True)

    manifest = args.output / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    split_counts = {split: sum(row["split"] == split for row in rows)
                    for split in ("train", "val", "test")}
    signer_sets = {split: {row["signer_id"] for row in rows if row["split"] == split}
                   for split in split_counts}
    summary = {
        "windows": len(rows), "source_videos": len(rows), "split_counts": split_counts,
        "official_split_signer_overlap": {
            "train_val": len(signer_sets["train"] & signer_sets["val"]),
            "train_test": len(signer_sets["train"] & signer_sets["test"]),
            "val_test": len(signer_sets["val"] & signer_sets["test"]),
        },
        "rejected": rejected, "frames_per_window": args.frames, "size": args.size,
        "crop_mode": args.crop_mode,
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "split_unit": "one non-overlapping window per source video",
        "landmark_order": "MediaPipe V3 right,left,pose,face; output left,right",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
