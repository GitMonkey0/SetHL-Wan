#!/usr/bin/env python3
"""Merge independently cached latent manifests and verify exact coverage."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--latents", type=Path, required=True)
    args = parser.parse_args()
    expected = [json.loads(x) for x in (args.windows / "manifest.jsonl").read_text().splitlines()]
    rows = []
    for path in sorted(args.latents.glob("manifest.shard-*.jsonl")):
        rows.extend(json.loads(x) for x in path.read_text().splitlines())
    by_id = {row["id"]: row for row in rows}
    expected_ids = {row["id"] for row in expected}
    if len(by_id) != len(rows): raise RuntimeError("duplicate IDs across shards")
    if set(by_id) != expected_ids:
        raise RuntimeError(f"coverage mismatch: missing={len(expected_ids-set(by_id))}, "
                           f"extra={len(set(by_id)-expected_ids)}")
    ordered = [by_id[row["id"]] for row in expected]
    manifest = args.latents / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in ordered))
    report = {"windows": len(ordered), "shards": len(list(args.latents.glob('manifest.shard-*.jsonl'))),
              "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
              "coverage": "exact"}
    (args.latents / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
