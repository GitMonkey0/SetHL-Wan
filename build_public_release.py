#!/usr/bin/env python3
"""Build a clean, data-free SetHL-Wan source and evidence repository."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent

ROOT_FILES = (
    ".gitignore", "CITATION.cff", "DATA.md", "DESIGN_LOCK.md", "ENVIRONMENT.md",
    "LICENSE", "README.md", "experiment_lock.json", "pyproject.toml",
    "requirements.txt", "requirements-eval.txt",
    "build_public_release.py",
    "aggregate_completion_results.py", "aggregate_denoising_results.py",
    "aggregate_generation_results.py", "aggregate_paired_results.py",
    "audit_wan_checkpoint.py", "cache_video_latents.py", "continuous_control_latent_bridge.py",
    "evaluate_codebooks.py", "evaluate_completion.py", "evaluate_denoising.py",
    "evaluate_generated_videos.py", "evaluate_tracker_ceiling.py", "evaluate_video_features.py", "fit_vq_codebook.py",
    "generate_sethl_wan.py",
    "hl_control_latent_bridge.py", "hl_programs.py", "hl_wan_adapter.py",
    "hl_wan_training_module.py", "kinematic_continuous_bridge.py",
    "kinematic_hl_bridge.py", "kinematic_spatializer.py", "masked_continuous_completion.py",
    "masked_hl_completion.py", "merge_latent_shards.py", "prepare_wlasl.py",
    "pretrain_completion.py", "raster_skeleton_bridge.py",
    "release_audit.py", "select_generation_subset.py", "summarize_generation_table.py",
    "train_sethl_wan.py", "validate_manifest.py", "wan_adapter_wrapper.py",
    "wan_training_contract.py", "run_denoising_shard.sh", "run_generation_shard.sh",
    "run_temperature_validation.sh", "run_training_seed.sh", "setup_videox_fun.sh",
)

PAPER_FILES = (
    "paper/main.tex", "paper/results.tex", "paper/refs.bib", "paper/spconf.sty",
    "paper/IEEEbib.bst", "paper/Makefile", "paper/package_overleaf.sh",
    "paper/make_method_figure.py", "paper/make_qualitative_figure.py",
    "paper/figures/method_overview.pdf", "paper/figures/method_overview.png",
    "paper/figures/qualitative.pdf", "paper/figures/qualitative.json", "paper/main.pdf",
    "patches/videox_fun_ascend.patch", "results/README.md",
    "data_sources/wlasl/latents_hand256/manifest.jsonl",
    "data_sources/wlasl/vq26_codebook.pt",
)

RESULT_FILES = (
    "results/generation_subset.json", "results/temperature_validation_subset.json",
    "results/codebook_test.json", "results/tracker_ceiling.json",
    "results/final_completion_sethl_vs_continuous.json",
    "results/final_completion_sethl_vs_vq.json",
    "results/final_completion_sethl_vs_hardhl.json",
    "results/final_completion_sethl_vs_sethl_nohier.json",
    "results/final_generation_summary.json",
    "results/final_generation_sethl_vs_continuous_finger.json",
    "results/policy_validation_selection.json",
    "results/final_denoising_sethl_vs_continuous_interval.json",
    "results/final_video_feature_consistency.json",
    "results/temperature_validation/selection.json",
)


def copy_file(relative: Path, target: Path) -> None:
    source = ROOT / relative
    if not source.is_file():
        raise FileNotFoundError(source)
    destination = target / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    # Generation provenance is retained, but machine-specific prefixes are not.
    if relative.name == "generation.json" and "results/generated" in relative.as_posix():
        payload = json.loads(destination.read_text())
        checkpoint = payload.get("checkpoint")
        if checkpoint:
            checkpoint = Path(checkpoint)
            payload["checkpoint"] = (Path("runs") / checkpoint.parent.name /
                                     checkpoint.name).as_posix()
            destination.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()
    target = args.target.resolve()
    if target.exists():
        raise FileExistsError(f"release target already exists: {target}")

    relative_files = {Path(name) for name in ROOT_FILES + PAPER_FILES + RESULT_FILES}
    excluded_tests = {"test_bitrate_accounting.py", "test_split_hot3d_metadata.py"}
    relative_files.update(path.relative_to(ROOT) for path in ROOT.glob("test_*.py")
                          if path.name not in excluded_tests)
    relative_files.update(path.relative_to(ROOT) for path in
                          (ROOT / "results" / "final_completion").glob("*.json"))
    relative_files.update(path.relative_to(ROOT) for path in
                          (ROOT / "results" / "generated").glob("**/*.json"))
    relative_files.update(path.relative_to(ROOT) for path in
                          (ROOT / "results" / "denoising").glob("**/*.json"))
    for pattern in ("final_generation_*.json", "final_completion_*.json",
                    "final_denoising_*.json"):
        relative_files.update(path.relative_to(ROOT) for path in
                              (ROOT / "results").glob(pattern))
    for pattern in ("wlasl_hand_raster_seed*", "wlasl_hand_pre_continuous_seed*",
                    "wlasl_hand_stable_vq_seed*", "wlasl_hand_pre_hardhl_seed*",
                    "wlasl_hand_cell_no_spherical_seed*", "wlasl_hand_cell_sethl_seed*"):
        relative_files.update(path.relative_to(ROOT) for path in
                              (ROOT / "runs").glob(f"{pattern}/report.json"))

    missing = [relative for relative in sorted(relative_files)
               if not (ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError("release inputs are missing:\n" +
                                "\n".join(str(path) for path in missing))
    target.mkdir(parents=True)
    for relative in sorted(relative_files):
        copy_file(relative, target)

    hashes = []
    for path in sorted(p for p in target.rglob("*") if p.is_file()):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes.append(f"{digest}  {path.relative_to(target).as_posix()}")
    (target / "SHA256SUMS").write_text("\n".join(hashes) + "\n")
    print(f"wrote {len(relative_files)} files plus SHA256SUMS to {target}")


if __name__ == "__main__":
    main()
