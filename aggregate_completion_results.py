#!/usr/bin/env python3
"""Paired source bootstrap across structured completion-mask policies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(paths: list[Path]) -> tuple[str, dict[str, dict[str, list[dict]]], int]:
    """Load replicate reports without treating training seeds as samples."""
    by_policy: dict[str, dict[str, list[dict]]] = {}
    representation = None
    for path in paths:
        report = json.loads(path.read_text())
        representation = representation or report["representation"]
        if report["representation"] != representation:
            raise ValueError("mixed representations")
        policy = report["mask_policy"]
        policy_rows = by_policy.setdefault(policy, {})
        for row in report["per_sample"]:
            policy_rows.setdefault(row["id"], []).append(row)
    replicate_counts = {len(rows) for sources in by_policy.values()
                        for rows in sources.values()}
    if len(replicate_counts) != 1:
        raise ValueError(f"unbalanced report grid: replicate counts={replicate_counts}")
    return representation, by_policy, replicate_counts.pop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=Path, nargs="+", required=True)
    parser.add_argument("--baseline", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    method_name, method, method_replicates = load(args.method)
    baseline_name, baseline, baseline_replicates = load(args.baseline)
    if method_replicates != baseline_replicates:
        raise ValueError("method and baseline have different replicate counts")
    policies = sorted(set(method) & set(baseline))
    if not policies:
        raise ValueError("no paired policies")
    ids = sorted(set.intersection(*(
        [set(method[p]) for p in policies] + [set(baseline[p]) for p in policies])))
    if not ids:
        raise ValueError("no paired source videos")
    rng = np.random.default_rng(args.seed)
    output = {}
    for metric in ("hidden_angular_deg", "hidden_symbol_accuracy"):
        # Average training seeds within each source before the paired source
        # bootstrap; otherwise repeated generations would inflate n.
        a = np.asarray([[np.mean([row[metric] for row in method[p][source]])
                         for p in policies] for source in ids]).mean(1)
        b = np.asarray([[np.mean([row[metric] for row in baseline[p][source]])
                         for p in policies] for source in ids]).mean(1)
        delta = a - b
        draws = rng.integers(0, len(ids), size=(args.bootstrap, len(ids)))
        bootstrap = delta[draws].mean(1)
        output[metric] = {
            "method_mean": float(a.mean()), "baseline_mean": float(b.mean()),
            "paired_delta": float(delta.mean()),
            "ci95": np.percentile(bootstrap, [2.5, 97.5]).tolist()}
    report = {"method": method_name, "baseline": baseline_name,
              "policies": policies, "paired_sources": len(ids),
              "training_replicates": method_replicates,
              "bootstrap_replicates": args.bootstrap, "metrics": output}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
