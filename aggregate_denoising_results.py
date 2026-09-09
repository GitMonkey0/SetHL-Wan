#!/usr/bin/env python3
"""Aggregate held-out Wan denoising reports at the source-video level."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_method(root: Path, method: str, seeds: tuple[int, ...]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for seed in seeds:
        path = root / f"{method}_seed{seed}.json"
        report = json.loads(path.read_text())
        if report.get("training_seed") != seed:
            raise ValueError(f"training seed mismatch in {path}")
        for row in report["per_sample"]:
            grouped.setdefault(row["id"], []).append(row)
    bad = {source: len(rows) for source, rows in grouped.items() if len(rows) != len(seeds)}
    if bad:
        raise ValueError(f"incomplete training-seed replicas for {method}: {bad}")
    return grouped


def paired_summary(method: dict[str, list[dict]], baseline: dict[str, list[dict]],
                   metric: str, rng: np.random.Generator,
                   bootstrap: int) -> dict:
    if set(method) != set(baseline):
        raise ValueError("method and baseline source sets differ")
    sources = sorted(method)
    left = np.asarray([np.mean([row[metric] for row in method[s]]) for s in sources])
    right = np.asarray([np.mean([row[metric] for row in baseline[s]]) for s in sources])
    delta = left - right
    draws = rng.integers(0, len(sources), size=(bootstrap, len(sources)))
    return {
        "method_mean": float(left.mean()),
        "baseline_mean": float(right.mean()),
        "paired_delta": float(delta.mean()),
        "ci95": np.percentile(delta[draws].mean(1), [2.5, 97.5]).tolist(),
        "paired_sources": len(sources),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--policy", choices=("interval", "full"), required=True)
    parser.add_argument("--method", default="sethl")
    parser.add_argument("--baseline", default="continuous")
    parser.add_argument("--seeds", type=int, nargs="+", default=(2027, 2028, 2029))
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seeds = tuple(args.seeds)
    policy_root = args.root / args.policy
    method = load_method(policy_root, args.method, seeds)
    baseline = load_method(policy_root, args.baseline, seeds)
    rng = np.random.default_rng(args.seed)
    metrics = {
        key: paired_summary(method, baseline, key, rng, args.bootstrap)
        for key in ("flow_mse", "zero_control_flow_mse", "control_gain")
    }
    output = {"policy": args.policy, "method": args.method,
              "baseline": args.baseline, "training_seeds": list(seeds),
              "metrics": metrics}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
