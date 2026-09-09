#!/usr/bin/env python3
"""Fail closed unless every artifact required by the paper is present."""

from __future__ import annotations

import argparse
import json
import hashlib
import math
from pathlib import Path

import pymupdf


HERE = Path(__file__).resolve().parent
SEEDS = (2027, 2028, 2029)
SOURCES = 27
MAIN_METHODS = ("raster", "continuous", "vq", "hardhl", "sethl_nohier", "sethl")
RUNS = {
    "raster": "wlasl_hand_raster_seed{}",
    "continuous": "wlasl_hand_pre_continuous_seed{}",
    "vq": "wlasl_hand_stable_vq_seed{}",
    "hardhl": "wlasl_hand_pre_hardhl_seed{}",
    "sethl_nohier": "wlasl_hand_cell_no_spherical_seed{}",
    "sethl": "wlasl_hand_cell_sethl_seed{}",
}
MANIFEST_SHA = "5693326da395d89911e91194a49a36e44342f43d4bc91762fcd282edd99f3dd7"
GENERATION_SUBSET_SHA = "adbd86ff74b7485f3e8dec3f43dde5fecd9e1d106e665c9f201f87d5483a8cf8"
TEMPERATURE_SUBSET_SHA = "643c4f7c84ae84ecf161df8589ebc1059cc2256b3f4554a908a5fc277b25edfb"
VQ_CODEBOOK_SHA = "862c60ff9f2b8071e364727933d8ac1e178e2c4cd21f542f5e74fe514dce2e1c"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def audit(evidence_only: bool = False) -> list[str]:
    failures: list[str] = []
    locked_files = {
        HERE / "data_sources" / "wlasl" / "latents_hand256" / "manifest.jsonl": MANIFEST_SHA,
        HERE / "results" / "generation_subset.json": GENERATION_SUBSET_SHA,
        HERE / "results" / "temperature_validation_subset.json": TEMPERATURE_SUBSET_SHA,
        HERE / "data_sources" / "wlasl" / "vq26_codebook.pt": VQ_CODEBOOK_SHA,
    }
    for path, expected_sha in locked_files.items():
        require(path.is_file(), f"missing locked manifest: {path}", failures)
        if path.is_file():
            require(sha256(path) == expected_sha, f"hash mismatch: {path}", failures)
    subset_path = HERE / "results" / "generation_subset.json"
    expected_indices: set[str] = set()
    expected_sources: set[str] = set()
    expected_test_sources: set[str] = set()
    if subset_path.is_file():
        selected = json.loads(subset_path.read_text()).get("selected", [])
        expected_indices = {str(row["sample_index"]) for row in selected}
        expected_sources = {str(row["id"]) for row in selected}
        require(len(expected_indices) == SOURCES and len(expected_sources) == SOURCES,
                "generation subset does not contain 27 unique indices and sources", failures)
    latent_manifest = HERE / "data_sources" / "wlasl" / "latents_hand256" / "manifest.jsonl"
    if latent_manifest.is_file():
        manifest_rows = [json.loads(line) for line in latent_manifest.read_text().splitlines()]
        expected_test_sources = {str(row["id"]) for row in manifest_rows
                                 if row.get("split") == "test"}
        require(len(expected_test_sources) == 269,
                "latent manifest does not contain 269 unique test sources", failures)
    selection_path = HERE / "results" / "temperature_validation" / "selection.json"
    require(selection_path.is_file(), "posterior-temperature selection is missing", failures)
    if selection_path.is_file():
        selection = json.loads(selection_path.read_text())
        require(selection.get("selection_split") == "val",
                "posterior temperature was not selected on validation data", failures)
        require(set(selection.get("candidates", {})) == {"temp1", "temp1p5", "temp2"},
                "posterior-temperature candidate grid differs from lock", failures)
        require(selection.get("selected") == "temp1",
                f"unexpected selected temperature: {selection.get('selected')}", failures)
        require(all(row.get("sources") == 8
                    for row in selection.get("candidates", {}).values()),
                "temperature validation grid is incomplete", failures)
    for method, run_pattern in RUNS.items():
        for seed in SEEDS:
            run = HERE / "runs" / run_pattern.format(seed)
            report_path = run / "report.json"
            checkpoint = run / "checkpoint-3000.pt"
            require(report_path.is_file(), f"missing training report: {report_path}", failures)
            if not evidence_only:
                require(checkpoint.is_file(), f"missing checkpoint: {checkpoint}", failures)
            if report_path.is_file():
                report = json.loads(report_path.read_text())
                require(report.get("status") == "complete", f"incomplete run: {run}", failures)
                require(report.get("steps") == 3000, f"wrong step count: {run}", failures)
                require(report.get("latent_manifest_sha256") == MANIFEST_SHA,
                        f"wrong latent manifest: {run}", failures)

    for method in MAIN_METHODS:
        for seed in SEEDS:
            root = HERE / "results" / "generated" / "interval" / f"{method}_seed{seed}"
            generations = list(root.glob("sample_*/generation.json"))
            reports = list(root.glob("sample_*.json"))
            require(len(generations) == SOURCES,
                    f"{method}/{seed}: {len(generations)}/{SOURCES} interval generations", failures)
            require(len(reports) == SOURCES,
                    f"{method}/{seed}: {len(reports)}/{SOURCES} interval reports", failures)
            report_indices = {path.stem.removeprefix("sample_") for path in reports}
            require(report_indices == expected_indices,
                    f"{method}/{seed}: interval sample-index set differs from lock", failures)
            report_sources: set[str] = set()
            for path in reports:
                row = json.loads(path.read_text())
                report_sources.add(str(row.get("source")))
                require(row.get("training_seed") == seed, f"wrong seed in {path}", failures)
                require(row.get("representation") == method,
                        f"wrong representation in {path}", failures)
                require(row.get("videos") == 16, f"wrong interval video count in {path}", failures)
                require("operating_point" in row and "frontier_hypervolume" in row,
                        f"stale report schema: {path}", failures)
                operating = row.get("operating_point", {})
                for key in ("specified_hl_accuracy", "specified_angular_deg",
                            "localized_diversity_deg", "detection_rate"):
                    number = operating.get(key)
                    require(isinstance(number, (int, float)) and math.isfinite(number),
                            f"invalid {key} in {path}", failures)
                hypervolume = row.get("frontier_hypervolume")
                require(isinstance(hypervolume, (int, float)) and
                        math.isfinite(hypervolume) and hypervolume >= 0,
                        f"invalid hypervolume in {path}", failures)
            require(report_sources == expected_sources,
                    f"{method}/{seed}: interval source set differs from lock", failures)

    for method in ("sethl", "continuous"):
        for seed in SEEDS:
            root = HERE / "results" / "generated" / "full" / f"{method}_seed{seed}"
            generations = list(root.glob("sample_*/generation.json"))
            reports = list(root.glob("sample_*.json"))
            require(len(generations) == SOURCES,
                    f"{method}/{seed}: incomplete full-control generation", failures)
            require(len(reports) == SOURCES,
                    f"{method}/{seed}: incomplete full-control reports", failures)
            require({path.stem.removeprefix("sample_") for path in reports} == expected_indices,
                    f"{method}/{seed}: full-control sample-index set differs from lock", failures)
            full_sources: set[str] = set()
            for path in reports:
                row = json.loads(path.read_text())
                full_sources.add(str(row.get("source")))
                require(row.get("representation") == method and row.get("training_seed") == seed,
                        f"wrong full-control provenance in {path}", failures)
                require(row.get("videos") == 4, f"wrong full-control video count in {path}", failures)
            require(full_sources == expected_sources,
                    f"{method}/{seed}: full-control source set differs from lock", failures)

    for seed in SEEDS:
        root = HERE / "results" / "generated" / "interval" / f"sethl_center_seed{seed}"
        reports = list(root.glob("sample_*.json"))
        require(len(reports) == SOURCES,
                f"sethl_center/{seed}: incomplete inference ablation", failures)
        require({path.stem.removeprefix("sample_") for path in reports} == expected_indices,
                f"sethl_center/{seed}: sample-index set differs from lock", failures)
        for path in reports:
            row = json.loads(path.read_text())
            require(row.get("representation") == "sethl_center" and row.get("training_seed") == seed,
                    f"wrong center-ablation provenance in {path}", failures)

    require(len(list((HERE / "results" / "final_completion").glob("*.json"))) == 45,
            "final completion grid is not 45 reports", failures)
    require((HERE / "results" / "final_generation_summary.json").is_file(),
            "final generated-video summary is missing", failures)
    require((HERE / "results" / "final_denoising_sethl_vs_continuous_interval.json").is_file(),
            "final denoising comparison is missing", failures)
    denoising = HERE / "results" / "denoising"
    for policy in ("interval", "full"):
        for method in MAIN_METHODS:
            for seed in SEEDS:
                path = denoising / policy / f"{method}_seed{seed}.json"
                require(path.is_file(), f"missing denoising report: {path}", failures)
                if path.is_file():
                    text = path.read_text()
                    require("NaN" not in text and "Infinity" not in text,
                            f"non-standard numeric value in {path}", failures)
                    report = json.loads(text)
                    require(report.get("representation") == method and
                            report.get("training_seed") == seed and
                            report.get("mask_policy") == policy,
                            f"wrong denoising provenance in {path}", failures)
                    rows = report.get("per_sample", [])
                    require(report.get("samples") == 269 and len(rows) == 269 and
                            len({row.get("id") for row in rows}) == 269,
                            f"incomplete denoising source set in {path}", failures)
                    require({str(row.get("id")) for row in rows} == expected_test_sources,
                            f"denoising source set differs from manifest in {path}", failures)
                    require(all(isinstance(row.get("flow_mse"), (int, float)) and
                                math.isfinite(row["flow_mse"]) for row in rows),
                            f"invalid denoising MSE in {path}", failures)

    results = (HERE / "paper" / "results.tex").read_text()
    require("pending" not in results.lower() and "{--}" not in results,
            "paper result macros still contain placeholders", failures)
    manuscript = (HERE / "paper" / "main.tex").read_text()
    for required_text in (
            "Haotian Lu", "Xiao-Ping Zhang", "Tsinghua University",
            "haotialu666@gmail.com", "xpzhang@ieee.org",
            "OpenAI Codex assisted"):
        require(required_text in manuscript,
                f"manuscript is missing required text: {required_text}", failures)
    qualitative = HERE / "paper" / "figures" / "qualitative.pdf"
    qualitative_meta = qualitative.with_suffix(".json")
    require(qualitative.is_file(), "qualitative figure is missing", failures)
    require(qualitative_meta.is_file(), "qualitative selection metadata is missing", failures)
    if qualitative_meta.is_file():
        metadata = json.loads(qualitative_meta.read_text())
        require(metadata.get("selection") == "median source by three-seed SetHL hypervolume",
                "qualitative example was not selected by the locked median rule", failures)
    pdf_path = HERE / "paper" / "main.pdf"
    require(pdf_path.is_file(), "paper PDF is missing", failures)
    if pdf_path.is_file():
        pdf_inputs = [HERE / "paper" / name for name in
                      ("main.tex", "results.tex", "refs.bib",
                       "figures/method_overview.pdf", "figures/qualitative.pdf")]
        require(all(pdf_path.stat().st_mtime >= path.stat().st_mtime for path in pdf_inputs),
                "paper PDF is older than one or more source files", failures)
        document = pymupdf.open(pdf_path)
        require(len(document) <= 5, f"paper has {len(document)} pages (maximum 5)", failures)
        for page_number, page in enumerate(document, start=1):
            width, height = page.rect.width, page.rect.height
            require(abs(width - 612) < 1 and abs(height - 792) < 1,
                    f"page {page_number} is not US Letter: {width:.1f} x {height:.1f} pt",
                    failures)
        if len(document) == 5:
            fifth_page_lines = {line.strip().upper()
                                for line in document[4].get_text().splitlines()}
            prohibited = ("ABSTRACT", "INTRODUCTION", "RELATED WORK", "METHOD",
                          "EXPERIMENTS", "CONCLUSION", "ACKNOWLEDGMENT")
            require(not any(heading in fifth_page_lines for heading in prohibited),
                    "page 5 contains technical or acknowledgment content", failures)
        type3 = {font[3] for page in document for font in page.get_fonts(full=True)
                 if font[2] == "Type3"}
        require(not type3, f"PDF contains Type-3 fonts: {sorted(type3)}", failures)
        document.close()
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-only", action="store_true",
                        help="verify the distributable reports without requiring checkpoints")
    args = parser.parse_args()
    failures = audit(evidence_only=args.evidence_only)
    if failures:
        print("RELEASE AUDIT FAILED")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print("RELEASE AUDIT PASSED")


if __name__ == "__main__":
    main()
