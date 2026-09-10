#!/usr/bin/env python3
"""Fail closed on the content-disjoint SetHL paper evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import pymupdf


ROOT = Path(__file__).resolve().parent
SEEDS = (2027, 2028, 2029)
METHODS = ("raster", "continuous", "vq", "hardhl", "sethl_nohier", "sethl")
RUNS = {
    "raster": "wlasl_hand_raster_seed{}",
    "continuous": "wlasl_hand_pre_continuous_seed{}",
    "vq": "wlasl_hand_stable_vq_seed{}",
    "hardhl": "wlasl_hand_pre_hardhl_seed{}",
    "sethl_nohier": "wlasl_hand_cell_no_spherical_seed{}",
    "sethl": "wlasl_hand_cell_sethl_seed{}",
}
LOCKED = {
    "data_sources/wlasl/latents_hand256_content_disjoint/manifest.jsonl":
        "a2f304db7d7e5a60ec7556c206f2f75f44ba175041d5430c1c82683c5a33b15d",
    "results/generation_subset_content_disjoint.json":
        "b4afea51b7886e9d239b0c10ed50f8d9768d5fcf27e60682f5435b031e284104",
    "results/temperature_validation_subset_content_disjoint.json":
        "f9263226730c347d9c3d112e17bb743850989dd99d28b54e9b748b26f02157a8",
    "data_sources/wlasl/vq26_codebook_content_disjoint.pt":
        "3bfbe6436169876e0ec3b97723ccfaeeda80909b98f7db97245c99c65f86fea5",
}
EXPECTED_TRAINABLE = {
    "raster": 21_440_720, "continuous": 21_839_508,
    "vq": 21_847_978, "hardhl": 21_847_978,
    "sethl_nohier": 21_847_978, "sethl": 21_847_978,
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def audit(evidence_only: bool) -> list[str]:
    failures: list[str] = []

    def require(ok: bool, message: str) -> None:
        if not ok:
            failures.append(message)

    for relative, expected in LOCKED.items():
        path = ROOT / relative
        require(path.is_file(), f"missing locked file: {relative}")
        if path.is_file():
            require(digest(path) == expected, f"hash mismatch: {relative}")

    manifest_path = ROOT / next(iter(LOCKED))
    rows = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    counts = {s: sum(row["split"] == s for row in rows) for s in ("train", "val", "test")}
    require(counts == {"train": 1402, "val": 392, "test": 265},
            f"wrong content-disjoint split: {counts}")
    require(len(rows) == len({row["id"] for row in rows}) == 2059,
            "manifest IDs are not 2,059 unique clips")
    report = json.loads((manifest_path.parent / "deduplication_report.json").read_text())
    require(report.get("unique_rgb") == 2059 and report.get("removed_rows") == 193,
            "deduplication report disagrees with lock")

    subset = json.loads((ROOT / "results/generation_subset_content_disjoint.json").read_text())
    sources = {str(row["id"]) for row in subset["selected"]}
    indices = {str(row["sample_index"]) for row in subset["selected"]}
    require(len(sources) == len(indices) == 25, "main subset is not 25 unique sources")

    for method, pattern in RUNS.items():
        for seed in SEEDS:
            run = ROOT / "runs_content_disjoint" / pattern.format(seed)
            info = json.loads((run / "report.json").read_text()) if (run / "report.json").is_file() else {}
            require(info.get("status") == "complete" and info.get("steps") == 3000,
                    f"incomplete training: {run.name}")
            require(info.get("examples") == 1402 and
                    info.get("latent_manifest_sha256") == LOCKED[next(iter(LOCKED))],
                    f"wrong training data provenance: {run.name}")
            require(info.get("trainable_parameters") == EXPECTED_TRAINABLE[method],
                    f"wrong trainable parameter count: {run.name}")
            if not evidence_only:
                checkpoint = run / "checkpoint-3000.pt"
                require(checkpoint.is_file(), f"missing checkpoint: {run.name}")
                if checkpoint.is_file():
                    import torch
                    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
                    lora = state.get("wan_trainable", {})
                    require(state.get("step") == 3000 and state.get("seed") == seed,
                            f"wrong checkpoint step/seed: {run.name}")
                    require(len(lora) == 480 and
                            sum(t.numel() for t in lora.values()) == 18_923_520 and
                            all("lora_" in key for key in lora),
                            f"wrong Wan LoRA tensors: {run.name}")
                    require(all(bool(torch.isfinite(t).all()) for t in lora.values()) and
                            any(bool(torch.count_nonzero(t)) for t in lora.values()),
                            f"non-finite or all-zero Wan LoRA: {run.name}")

    for method in METHODS:
        for seed in SEEDS:
            root = ROOT / "results/generated_content_disjoint/interval" / f"{method}_seed{seed}"
            reports = list(root.glob("sample_*.json"))
            generations = list(root.glob("sample_*/generation.json"))
            require(len(reports) == len(generations) == 25,
                    f"incomplete interval generation: {method}/{seed}")
            require({str(json.loads(p.read_text()).get("source")) for p in reports} == sources,
                    f"wrong interval source set: {method}/{seed}")
            require({p.stem.removeprefix("sample_") for p in reports} == indices,
                    f"wrong interval index set: {method}/{seed}")

    for method in ("continuous", "sethl"):
        for seed in SEEDS:
            root = ROOT / "results/generated_content_disjoint/full" / f"{method}_seed{seed}"
            require(len(list(root.glob("sample_*.json"))) == 25 and
                    len(list(root.glob("sample_*/generation.json"))) == 25,
                    f"incomplete full-control generation: {method}/{seed}")

    test_sources = {str(row["id"]) for row in rows if row["split"] == "test"}
    for policy in ("interval", "full"):
        for method in ("continuous", "sethl"):
            for seed in SEEDS:
                path = ROOT / "results/denoising_content_disjoint" / policy / f"{method}_seed{seed}.json"
                data = json.loads(path.read_text()) if path.is_file() else {}
                samples = data.get("per_sample", [])
                require(data.get("samples") == 265 and len(samples) == 265 and
                        {str(row.get("id")) for row in samples} == test_sources,
                        f"incomplete denoising report: {policy}/{method}/{seed}")
                require(all(math.isfinite(float(row.get("flow_mse", float("nan")))) for row in samples),
                        f"invalid denoising metric: {policy}/{method}/{seed}")

    for relative in (
        "results/final_generation_summary_content_disjoint.json",
        "results/final_denoising_sethl_vs_continuous_interval_content_disjoint.json",
        "results/final_video_feature_consistency_content_disjoint.json",
        "results/tracker_ceiling_content_disjoint.json",
        "paper/results_content_disjoint.tex", "paper/main.pdf",
    ):
        require((ROOT / relative).is_file(), f"missing final artifact: {relative}")

    pdf = ROOT / "paper/main.pdf"
    if pdf.is_file():
        doc = pymupdf.open(pdf)
        require(len(doc) <= 5, f"paper has {len(doc)} pages")
        require(all(abs(p.rect.width - 612) < 1 and abs(p.rect.height - 792) < 1 for p in doc),
                "paper is not US Letter")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-only", action="store_true")
    args = parser.parse_args()
    failures = audit(args.evidence_only)
    if failures:
        raise SystemExit("AUDIT FAILED\n- " + "\n- ".join(failures))
    print("AUDIT PASSED: content-disjoint paper evidence is complete.")


if __name__ == "__main__":
    main()
