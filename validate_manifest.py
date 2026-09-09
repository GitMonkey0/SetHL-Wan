#!/usr/bin/env python3
"""Validate leakage-sensitive JSONL manifests before HL-Wan training."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


REQUIRED = {"clip_id", "split", "source_id", "start_frame", "end_frame",
            "rgb_path", "pose_path"}
ALLOWED_SPLITS = {"train", "val", "test"}


def read_manifest(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        missing = REQUIRED - row.keys()
        if missing:
            raise ValueError(f"line {line_number}: missing {sorted(missing)}")
        if row["split"] not in ALLOWED_SPLITS:
            raise ValueError(f"line {line_number}: invalid split {row['split']!r}")
        if not 0 <= int(row["start_frame"]) <= int(row["end_frame"]):
            raise ValueError(f"line {line_number}: invalid frame interval")
        rows.append(row)
    return rows


def audit(rows: list[dict], forbidden_sources: set[str] | None = None) -> dict:
    forbidden_sources = forbidden_sources or set()
    ids = [row["clip_id"] for row in rows]
    duplicated_ids = sorted(key for key, count in Counter(ids).items() if count > 1)
    source_splits: dict[str, set[str]] = defaultdict(set)
    intervals: dict[tuple[str, str], list[tuple[int, int, str]]] = defaultdict(list)
    for row in rows:
        source_splits[str(row["source_id"])].add(row["split"])
        intervals[(row["split"], str(row["source_id"]))].append(
            (int(row["start_frame"]), int(row["end_frame"]), str(row["clip_id"])))
    cross_split = {source: sorted(splits) for source, splits in source_splits.items()
                   if len(splits) > 1}
    overlaps = []
    for (split, source), spans in intervals.items():
        spans.sort()
        for previous, current in zip(spans, spans[1:]):
            if current[0] <= previous[1]:
                overlaps.append({"split": split, "source_id": source,
                                 "clips": [previous[2], current[2]]})
    forbidden_hits = sorted(set(source_splits) & forbidden_sources)
    result = {
        "clips": len(rows),
        "split_counts": dict(sorted(Counter(row["split"] for row in rows).items())),
        "source_counts": dict(sorted(Counter(row["source_id"] for row in rows).items())),
        "duplicate_clip_ids": duplicated_ids,
        "cross_split_sources": cross_split,
        "overlapping_windows": overlaps,
        "forbidden_source_hits": forbidden_hits,
    }
    result["valid"] = not any((duplicated_ids, cross_split, overlaps, forbidden_hits))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--forbid-source", action="append", default=[])
    args = parser.parse_args()
    result = audit(read_manifest(args.manifest), set(args.forbid_source))
    print(json.dumps(result, indent=2))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
