#!/usr/bin/env python3
"""Evaluate source-video consistency with Kinetics-pretrained R3D-18 features.

This is a secondary, representation-independent diagnostic.  Each generated
clip is compared with its source clip in the penultimate R3D-18 feature space;
statistics first average diffusion and training seeds within each source, then
bootstrap source videos.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torchvision.models.video import r3d_18


METHODS = ("raster", "continuous", "vq", "hardhl", "sethl_nohier", "sethl")
SEEDS = (2027, 2028, 2029)
MEAN = torch.tensor((0.43216, 0.394666, 0.37645)).view(3, 1, 1, 1)
STD = torch.tensor((0.22803, 0.22145, 0.216989)).view(3, 1, 1, 1)
WEIGHTS_SHA256 = "b3b3357ead25631ec9c57362ff2128a92d0427e01e2cd184951a44380c3f2e9d"


def load_video(path: Path) -> torch.Tensor:
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise ValueError(f"could not decode {path}")
    video = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 255
    video = torch.nn.functional.interpolate(
        video, size=(128, 128), mode="bilinear", align_corners=False)
    video = video[..., 8:120, 8:120].permute(1, 0, 2, 3)
    return (video - MEAN) / STD


def summarize(source_scores: dict[str, list[float]], rng: np.random.Generator,
              bootstrap: int) -> dict:
    sources = sorted(source_scores)
    values = np.asarray([np.mean(source_scores[source]) for source in sources])
    draws = rng.integers(0, len(values), size=(bootstrap, len(values)))
    return {"mean": float(values.mean()),
            "ci95": np.percentile(values[draws].mean(1), [2.5, 97.5]).tolist(),
            "sources": len(sources), "videos": sum(map(len, source_scores.values()))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--control-scale", type=float, default=1.0)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    actual_weights_sha = hashlib.sha256(args.weights.read_bytes()).hexdigest()
    if actual_weights_sha != WEIGHTS_SHA256:
        raise ValueError(f"unexpected R3D-18 weights SHA-256: {actual_weights_sha}")
    if args.device.startswith("npu"):
        import torch_npu  # noqa: F401
        torch.npu.set_device(args.device)
    device = torch.device(args.device)
    model = r3d_18(weights=None)
    model.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True))
    model.fc = nn.Identity()
    model.to(device).eval()

    cache: dict[Path, torch.Tensor] = {}

    def encode(path: Path) -> torch.Tensor:
        if path not in cache:
            video = load_video(path).unsqueeze(0).to(device)
            with torch.inference_mode():
                feature = torch.nn.functional.normalize(model(video).float(), dim=-1)
            cache[path] = feature[0].cpu()
        return cache[path]

    per_method: dict[str, dict[str, list[float]]] = {}
    methods = tuple(args.methods)
    seeds = tuple(args.seeds)
    for method in methods:
        scores: dict[str, list[float]] = {}
        for seed in seeds:
            for report_path in sorted(
                    args.generation_root.glob(f"{method}_seed{seed}/sample_*/generation.json")):
                report = json.loads(report_path.read_text())
                source = str(report["source"])
                # Window tensors retain the exact 17 RGB frames used to build
                # the conditioning latent.
                window = torch.load(args.windows / f"{source}.pt", weights_only=True)
                reference_path = args.windows / f"{source}.pt"
                if reference_path not in cache:
                    video = window["rgb"].float() / 255
                    video = torch.nn.functional.interpolate(
                        video, size=(128, 128), mode="bilinear", align_corners=False)
                    video = video[..., 8:120, 8:120].permute(1, 0, 2, 3)
                    video = (video - MEAN) / STD
                    with torch.inference_mode():
                        cache[reference_path] = torch.nn.functional.normalize(
                            model(video.unsqueeze(0).to(device)).float(), dim=-1)[0].cpu()
                reference = cache[reference_path]
                for output in report["outputs"]:
                    if abs(float(output["control_scale"]) - args.control_scale) > 1e-8:
                        continue
                    feature = encode(report_path.parent / output["video"])
                    scores.setdefault(source, []).append(float(torch.dot(reference, feature)))
        per_method[method] = scores

    rng = np.random.default_rng(args.seed)
    summaries = {method: summarize(scores, rng, args.bootstrap)
                 for method, scores in per_method.items()}
    comparisons = {}
    if "sethl" in per_method:
        sethl_values = {source: np.mean(values)
                        for source, values in per_method["sethl"].items()}
        for baseline in methods:
            if baseline == "sethl":
                continue
            sources = sorted(set(sethl_values) & set(per_method[baseline]))
            delta = np.asarray([sethl_values[source] - np.mean(per_method[baseline][source])
                                for source in sources])
            draws = rng.integers(0, len(delta), size=(args.bootstrap, len(delta)))
            comparisons[f"sethl_minus_{baseline}"] = {
                "mean": float(delta.mean()),
                "ci95": np.percentile(delta[draws].mean(1), [2.5, 97.5]).tolist(),
                "paired_sources": len(sources),
            }
    payload = {
        "feature_extractor": "torchvision R3D-18 Kinetics-400 V1 penultimate layer",
        "weights_sha256": WEIGHTS_SHA256,
        "control_scale": args.control_scale,
        "aggregation": "mean over diffusion/training seeds, then source bootstrap",
        "summary": summaries,
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
