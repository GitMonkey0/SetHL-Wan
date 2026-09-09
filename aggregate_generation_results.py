#!/usr/bin/env python3
"""Paired source-video bootstrap for generated-video control metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(paths: list[Path], expected_training_seeds: int) -> dict[str, list[dict]]:
    reports = [json.loads(path.read_text()) for path in paths]
    grouped: dict[str, list[dict]] = {}
    for report in reports:
        grouped.setdefault(report["source"], []).append(report)
    for source, members in grouped.items():
        seeds = [row.get("training_seed") for row in members]
        if len(members) != expected_training_seeds or len(set(seeds)) != len(seeds):
            raise ValueError(
                f"{source}: expected {expected_training_seeds} distinct training seeds, got {seeds}")
    return grouped


def one_metric(report: dict, key: str) -> float:
    if key == "frontier_hypervolume":
        raw = report[key]
    else:
        raw = report["operating_point"][key]
    return float("nan") if raw is None else float(raw)


def source_metric(reports: list[dict], key: str) -> float:
    """Average training seeds within source; source remains the sampling unit."""
    values = np.asarray([one_metric(report, key) for report in reports], dtype=float)
    return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", type=Path, nargs="+", required=True)
    p.add_argument("--baseline", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=2027)
    p.add_argument("--expected-training-seeds", type=int, default=3)
    args = p.parse_args()
    method = load(args.method, args.expected_training_seeds)
    baseline = load(args.baseline, args.expected_training_seeds)
    ids = sorted(set(method) & set(baseline))
    if not ids:
        raise ValueError("no paired source videos")
    metrics = ("frontier_hypervolume", "specified_hl_accuracy", "specified_angular_deg",
               "hidden_pairwise_diversity_deg",
               "specified_pairwise_symbol_disagreement", "detection_rate",
               "temporal_acceleration_error")
    rng = np.random.default_rng(args.seed); output = {}
    for key in metrics:
        a = np.asarray([source_metric(method[i], key) for i in ids])
        b = np.asarray([source_metric(baseline[i], key) for i in ids])
        finite = np.isfinite(a) & np.isfinite(b); a, b = a[finite], b[finite]
        if not len(a):
            output[key] = {"method_mean": None, "baseline_mean": None,
                           "paired_delta": None, "ci95": None,
                           "paired_sources": 0}
            continue
        delta = a - b
        draws = rng.integers(0, len(delta), size=(args.bootstrap, len(delta)))
        boot = delta[draws].mean(1)
        output[key] = {"method_mean": float(a.mean()), "baseline_mean": float(b.mean()),
                       "paired_delta": float(delta.mean()),
                       "ci95": np.percentile(boot, [2.5, 97.5]).tolist(),
                       "paired_sources": int(len(delta))}
    report = {"method": next(iter(method.values()))[0]["representation"],
              "baseline": next(iter(baseline.values()))[0]["representation"],
              "metrics": output, "bootstrap_replicates": args.bootstrap}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
