#!/usr/bin/env python3
"""Aggregate matched-seed reports with source-video paired bootstrap CIs."""

import argparse
import json
from pathlib import Path

import numpy as np


def load(paths):
    out = {}
    for path in paths:
        report = json.loads(path.read_text())
        out[path.parent.name] = {row["id"]: row for row in report["per_sample"]}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", type=Path, nargs="+", required=True)
    p.add_argument("--baseline", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=2027)
    args = p.parse_args()
    if len(args.method) != len(args.baseline):
        raise ValueError("method/baseline seed counts differ")
    methods, baselines = load(args.method), load(args.baseline)
    method_runs, baseline_runs = list(methods.values()), list(baselines.values())
    ids = sorted(set.intersection(*(set(x) for x in method_runs + baseline_runs)))
    metrics = ("flow_mse", "hidden_angular_deg", "observed_angular_deg")
    rng = np.random.default_rng(args.seed); result = {}
    for metric in metrics:
        method = np.asarray([[run[i][metric] for i in ids] for run in method_runs])
        baseline = np.asarray([[run[i][metric] for i in ids] for run in baseline_runs])
        paired = (method - baseline).mean(axis=0)
        draws = rng.integers(0, len(ids), size=(args.bootstrap, len(ids)))
        bootstrap = paired[draws].mean(axis=1)
        result[metric] = {
            "method_mean": float(method.mean()), "baseline_mean": float(baseline.mean()),
            "paired_delta_method_minus_baseline": float(paired.mean()),
            "ci95": np.percentile(bootstrap, [2.5, 97.5]).tolist(),
        }
    report = {"paired_source_videos": len(ids), "seeds": len(method_runs),
              "bootstrap_replicates": args.bootstrap, "metrics": result}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
