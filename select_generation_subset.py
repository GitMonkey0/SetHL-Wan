#!/usr/bin/env python3
"""Predeclare a deterministic, tracking-qualified generation subset."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--count", type=int, default=10)
    p.add_argument("--seed", type=int, default=2027)
    p.add_argument("--split", default="test")
    p.add_argument("--min-hand-visible", type=float, default=.9)
    args = p.parse_args()
    rows = [json.loads(x) for x in args.manifest.read_text().splitlines()]
    split_rows = [x for x in rows if x.get("split") == args.split]
    eligible = [(i, row) for i, row in enumerate(split_rows)
                if min(row["visible_fraction"]) >= args.min_hand_visible]
    def rank(item):
        return hashlib.sha256(f"{args.seed}:{item[1]['id']}".encode()).hexdigest()
    chosen = sorted(eligible, key=rank)[:args.count]
    report = {"seed": args.seed, "criterion": {
        "split": args.split, "both_hand_visible_fraction_gte": args.min_hand_visible},
        "eligible": len(eligible), "selected": [
            {"sample_index": i, "id": row["id"], "gloss": row["gloss"],
             "visible_fraction": row["visible_fraction"]} for i, row in chosen]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
