#!/usr/bin/env python3
"""Measure constraint compliance and localized diversity in generated videos."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent / "eval_vendor"))
sys.path.insert(1, str(ROOT / "vendor/python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hl_programs import joints_to_hl, structured_mask  # noqa: E402


def track_video(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    # Keep the evaluator-only dependency lazy so metric/unit-test utilities can
    # be imported in the lighter training environment.
    import mediapipe as mp
    capture = cv2.VideoCapture(str(path)); frames = []
    while True:
        ok, frame = capture.read()
        if not ok: break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    joints = np.zeros((len(frames), 2, 21, 3), dtype=np.float32)
    visible = np.zeros((len(frames), 2), dtype=np.float32)
    with mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2,
                                  model_complexity=1, min_detection_confidence=.35,
                                  min_tracking_confidence=.35) as detector:
        for index, frame in enumerate(frames):
            result = detector.process(frame)
            if not result.multi_hand_landmarks: continue
            for landmarks, handedness in zip(result.multi_hand_landmarks,
                                              result.multi_handedness):
                # MediaPipe's handedness convention assumes a mirrored camera.
                # WLASL is not mirrored, hence invert its label.
                label = handedness.classification[0].label.lower()
                hand = 0 if label == "right" else 1  # output left,right
                joints[index, hand] = np.asarray(
                    [[p.x, p.y, p.z] for p in landmarks.landmark], dtype=np.float32)
                visible[index, hand] = 1
    return torch.from_numpy(joints), torch.from_numpy(visible)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    return float(values[mask].float().mean()) if bool(mask.any()) else float("nan")


def normalized_acceleration_error(pred: torch.Tensor, target: torch.Tensor,
                                  pred_visible: torch.Tensor,
                                  target_visible: torch.Tensor) -> float:
    """Second-difference error of wrist-relative articulation, palm normalized."""
    def normalize(joints):
        local = joints[..., :2] - joints[..., :1, :2]
        palm = torch.linalg.vector_norm(
            joints[..., 9, :2] - joints[..., 0, :2], dim=-1).clamp_min(1e-3)
        return local / palm[..., None, None]
    pred_local, target_local = normalize(pred), normalize(target)
    pred_acc = pred_local[:, 2:] - 2 * pred_local[:, 1:-1] + pred_local[:, :-2]
    target_acc = target_local[:, 2:] - 2 * target_local[:, 1:-1] + target_local[:, :-2]
    valid = (pred_visible[:, 2:].bool() & pred_visible[:, 1:-1].bool() &
             pred_visible[:, :-2].bool() & target_visible[:, 2:].bool() &
             target_visible[:, 1:-1].bool() & target_visible[:, :-2].bool())
    error = torch.linalg.vector_norm(pred_acc - target_acc, dim=-1).mean(-1)
    return masked_mean(error, valid)


def compliance_diversity_hypervolume(frontier: list[dict]) -> float:
    """Area dominated by compliance/diversity points relative to (0, 0).

    The union-of-rectangles definition rewards both reaching high compliance
    and retaining hidden-cell diversity, while discarding dominated operating
    points. Unlike a raw trapezoid over an arbitrary achieved x-range, all
    methods share the same fixed origin.
    """
    points = sorted((float(row["specified_hl_accuracy"]),
                     float(row["localized_diversity_deg"])) for row in frontier
                    if np.isfinite(row["specified_hl_accuracy"])
                    and np.isfinite(row["localized_diversity_deg"])
                    and row["specified_hl_accuracy"] >= 0
                    and row["localized_diversity_deg"] >= 0)
    if not points:
        return float("nan")
    area, previous = 0.0, 0.0
    for index, (accuracy, _) in enumerate(points):
        height = max(diversity for _, diversity in points[index:])
        area += max(0.0, accuracy - previous) * height
        previous = max(previous, accuracy)
    return area


def json_safe(value):
    """Recursively replace non-finite floats with JSON null."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def main():
    # Importing the Wan training entry point pulls in the upstream VideoX-Fun
    # package, which is unnecessary for the pure metric helpers above.
    from train_sethl_wan import fill_missing
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument("--latents", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads((args.generation / "generation.json").read_text())
    rows = {json.loads(line)["id"]: json.loads(line)
            for line in (args.latents / "manifest.jsonl").read_text().splitlines()}
    row = rows[spec["source"]]
    source = torch.load(args.latents / row["path"], weights_only=True)
    target_joints = fill_missing(source["joints"].unsqueeze(0).float(),
                                 source["visible"].unsqueeze(0).float())
    target_visible = source["visible"].unsqueeze(0).float()
    target = joints_to_hl(target_joints)
    frames = target_joints.shape[1]
    if "specified_mask" not in spec:
        raise ValueError("generation manifest predates explicit mask serialization")
    masks = torch.tensor(spec["specified_mask"], dtype=torch.bool)
    tracks, visibility, programs = [], [], []
    for item in spec["outputs"]:
        joint, visible = track_video(args.generation / item["video"])
        visible = visible.unsqueeze(0)
        try:
            joint = fill_missing(joint.unsqueeze(0), visible)
            program = joints_to_hl(joint)
        except ValueError:
            # A video with no detected hand is a compliance failure, not an
            # evaluator crash or a silently discarded sample.
            joint, program = None, None
        tracks.append(joint); visibility.append(visible)
        programs.append(program)
    per_video = []
    for item, program, visible in zip(spec["outputs"], programs, visibility):
        if program is None:
            per_video.append({
                "video": item["video"], "control_scale": item.get("control_scale", 1.0),
                "detection_rate": 0.0, "specified_hl_accuracy": 0.0,
                "specified_angular_deg": 180.0,
                "temporal_acceleration_error": float("nan")})
            continue
        valid = visible[..., None].expand_as(masks).bool()
        specified = masks & valid
        cosine = (program.local_direction * target.local_direction).sum(-1).clamp(-1, 1)
        angle = torch.rad2deg(torch.acos(cosine))
        per_video.append({
            "video": item["video"], "control_scale": item.get("control_scale", 1.0),
            "detection_rate": float(visible.mean()),
            "specified_hl_accuracy": masked_mean(program.symbol == target.symbol, specified),
            "specified_angular_deg": masked_mean(angle, specified),
            "temporal_acceleration_error": normalized_acceleration_error(
                tracks[len(per_video)], target_joints, visible, target_visible),
        })
    scales = sorted({item.get("control_scale", 1.0) for item in spec["outputs"]})
    frontier = []
    for scale in scales:
        members = [i for i, item in enumerate(spec["outputs"])
                   if item.get("control_scale", 1.0) == scale]
        specified_pairwise, specified_symbol_disagreement, hidden_pairwise = [], [], []
        for left, right in itertools.combinations(members, 2):
            if programs[left] is None or programs[right] is None:
                continue
            valid = (visibility[left][..., None].bool() & visibility[right][..., None].bool())
            valid = valid.expand_as(masks)
            cosine = (programs[left].local_direction * programs[right].local_direction).sum(-1).clamp(-1, 1)
            angle = torch.rad2deg(torch.acos(cosine))
            specified_pairwise.append(masked_mean(angle, masks & valid))
            specified_symbol_disagreement.append(masked_mean(
                programs[left].symbol != programs[right].symbol, masks & valid))
            hidden_pairwise.append(masked_mean(angle, (~masks) & valid))
        member_rows = [per_video[i] for i in members]
        detection = float(np.nanmean([x["detection_rate"] for x in member_rows]))
        frontier.append({
            "control_scale": scale,
            "detection_rate": detection,
            "specified_hl_accuracy": float(np.nanmean([x["specified_hl_accuracy"] for x in member_rows])),
            "specified_angular_deg": float(np.nanmean([x["specified_angular_deg"] for x in member_rows])),
            "temporal_acceleration_error": float(np.nanmean(
                [x["temporal_acceleration_error"] for x in member_rows])),
            "specified_pairwise_diversity_deg": float(np.nanmean(specified_pairwise)),
            "specified_pairwise_symbol_disagreement": float(
                np.nanmean(specified_symbol_disagreement)),
            "hidden_pairwise_diversity_deg": float(np.nanmean(hidden_pairwise)),
            # Reward variation on free cells only to the extent that paired
            # outputs retain the same symbols on specified cells. Angular
            # sub-cell variation in specified cells remains legal.
            "localized_diversity_deg": float(
                np.nanmean(hidden_pairwise) *
                (1.0 - np.nanmean(specified_symbol_disagreement)) * detection),
        })
    auc = compliance_diversity_hypervolume(frontier)
    unit = min(frontier, key=lambda x: abs(x["control_scale"] - 1.0))
    report = {
        "source": spec["source"], "representation": spec["representation"],
        "training_seed": spec.get("training_seed"),
        "training_step": spec.get("training_step"),
        "videos": len(programs), "per_video": per_video,
        "means": {
            "detection_rate": float(np.nanmean([x["detection_rate"] for x in per_video])),
            "specified_hl_accuracy": float(np.nanmean([x["specified_hl_accuracy"] for x in per_video])),
            "specified_angular_deg": float(np.nanmean([x["specified_angular_deg"] for x in per_video])),
            "hidden_pairwise_diversity_deg": unit["hidden_pairwise_diversity_deg"],
            "specified_pairwise_symbol_disagreement":
                unit["specified_pairwise_symbol_disagreement"],
            "temporal_acceleration_error": float(np.nanmean(
                [x["temporal_acceleration_error"] for x in per_video])),
        }, "operating_point": unit, "frontier": frontier,
        "frontier_hypervolume": auc}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = json_safe(report)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__": main()
