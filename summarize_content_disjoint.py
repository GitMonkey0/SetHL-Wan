#!/usr/bin/env python3
"""Summarize the locked content-disjoint generation experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from summarize_generation_table import METHODS, bootstrap_mean, collect, source_values


METRICS = ("specified_hl_accuracy", "specified_angular_deg",
           "localized_diversity_deg", "frontier_hypervolume",
           "specified_pairwise_symbol_disagreement", "detection_rate",
           "temporal_acceleration_error")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--codebook-results", type=Path, required=True)
    parser.add_argument("--temperature-selection", type=Path, required=True)
    parser.add_argument("--full-root", type=Path)
    parser.add_argument("--denoising-results", type=Path)
    parser.add_argument("--video-feature-results", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-tex", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    sources = [str(row["id"]) for row in json.loads(args.subset.read_text())["selected"]]
    expected = set(sources)
    grouped = {method: collect(args.root, method, expected) for method in METHODS}
    arrays = {method: {metric: source_values(rows, metric, sources)
                       for metric in METRICS} for method, rows in grouped.items()}
    rng = np.random.default_rng(args.seed)
    summary = {method: {metric: {"mean": float(np.nanmean(values)),
                                 "ci95": bootstrap_mean(values, rng, args.bootstrap)}
                        for metric, values in method_arrays.items()}
               for method, method_arrays in arrays.items()}
    comparisons = {}
    for method, baseline in (("sethl", "continuous"), ("sethl", "vq"),
                             ("sethl", "hardhl"), ("sethl", "sethl_nohier"),
                             ("hardhl", "continuous")):
        key = f"{method}_minus_{baseline}"
        comparisons[key] = {}
        for metric in METRICS:
            delta = arrays[method][metric] - arrays[baseline][metric]
            comparisons[key][metric] = {
                "mean": float(np.nanmean(delta)),
                "ci95": bootstrap_mean(delta, rng, args.bootstrap),
            }
    codebook = json.loads(args.codebook_results.read_text())
    calibration = json.loads(args.temperature_selection.read_text())
    output = {"sources": len(sources), "training_seeds": 3,
              "diffusion_seeds": 4, "control_scales": [0, .5, 1, 1.5],
              "summary": summary, "comparisons": comparisons,
              "codebook": codebook, "posterior_temperature": calibration}

    extras: dict[str, dict] = {}
    if args.full_root:
        full_set = collect(args.full_root, "sethl", expected)
        full_cont = collect(args.full_root, "continuous", expected)
        left = source_values(full_set, "specified_hl_accuracy", sources)
        right = source_values(full_cont, "specified_hl_accuracy", sources)
        delta = left - right
        extras["full_control"] = {
            "sethl_mean": float(left.mean()), "continuous_mean": float(right.mean()),
            "paired_delta": float(delta.mean()),
            "ci95": bootstrap_mean(delta, rng, args.bootstrap),
        }
    if args.denoising_results:
        extras["denoising"] = json.loads(args.denoising_results.read_text())
    if args.video_feature_results:
        extras["video_features"] = json.loads(args.video_feature_results.read_text())
    output.update(extras)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n")

    symbolic = ("vq", "hardhl", "sethl_nohier", "sethl")
    best = {
        "specified_hl_accuracy": max(symbolic, key=lambda m: summary[m]["specified_hl_accuracy"]["mean"]),
        "specified_angular_deg": min(symbolic, key=lambda m: summary[m]["specified_angular_deg"]["mean"]),
        "localized_diversity_deg": max(symbolic, key=lambda m: summary[m]["localized_diversity_deg"]["mean"]),
        "frontier_hypervolume": max(symbolic, key=lambda m: summary[m]["frontier_hypervolume"]["mean"]),
    }
    prefixes = {"raster": "Raster", "continuous": "Cont", "vq": "VQ",
                "hardhl": "Hard", "sethl_nohier": "NoHier", "sethl": "Set"}
    lines = ["% Automatically generated from the locked content-disjoint experiment."]
    for method, prefix in prefixes.items():
        for metric, suffix, percent in (
                ("specified_hl_accuracy", "Acc", True),
                ("specified_angular_deg", "Ang", False),
                ("localized_diversity_deg", "Div", False),
                ("frontier_hypervolume", "AUC", False)):
            value = summary[method][metric]["mean"] * (100 if percent else 1)
            rendered = f"{value:.1f}" if percent else f"{value:.2f}"
            if method in symbolic and best[metric] == method:
                rendered = rf"\textbf{{{rendered}}}"
            lines.append(rf"\newcommand{{\{prefix}{suffix}}}{{{rendered}}}")

    def effect(prefix: str, comparison: str, metric: str, scale: float = 1) -> None:
        item = comparisons[comparison][metric]
        lines.append(rf"\newcommand{{\{prefix}Delta}}{{{item['mean'] * scale:.2f}}}")
        lines.append(rf"\newcommand{{\{prefix}CI}}{{[{item['ci95'][0] * scale:.2f}, {item['ci95'][1] * scale:.2f}]}}")

    effect("Primary", "sethl_minus_continuous", "frontier_hypervolume")
    effect("VQ", "sethl_minus_vq", "frontier_hypervolume")
    effect("Spherical", "sethl_minus_sethl_nohier", "frontier_hypervolume")
    effect("Hard", "sethl_minus_hardhl", "frontier_hypervolume")
    effect("ContAcc", "sethl_minus_continuous", "specified_hl_accuracy", 100)
    effect("ContAng", "sethl_minus_continuous", "specified_angular_deg")
    effect("ContLeak", "sethl_minus_continuous", "specified_pairwise_symbol_disagreement", 100)
    effect("ContDetection", "sethl_minus_continuous", "detection_rate", 100)
    effect("ContLocalized", "sethl_minus_continuous", "localized_diversity_deg")
    effect("VQAcc", "sethl_minus_vq", "specified_hl_accuracy", 100)
    effect("VQLeak", "sethl_minus_vq", "specified_pairwise_symbol_disagreement", 100)
    effect("SphericalAcc", "sethl_minus_sethl_nohier", "specified_hl_accuracy", 100)
    effect("SphericalAng", "sethl_minus_sethl_nohier", "specified_angular_deg")
    effect("HardContAcc", "hardhl_minus_continuous", "specified_hl_accuracy", 100)
    effect("HardContAng", "hardhl_minus_continuous", "specified_angular_deg")
    effect("HardContLeak", "hardhl_minus_continuous", "specified_pairwise_symbol_disagreement", 100)
    effect("HardContDetection", "hardhl_minus_continuous", "detection_rate", 100)
    effect("HardContLocalized", "hardhl_minus_continuous", "localized_diversity_deg")
    effect("SetHardLocalized", "sethl_minus_hardhl", "localized_diversity_deg")
    selected = calibration["selected"].removeprefix("temp").replace("p", ".")
    lines += [
        rf"\newcommand{{\HLQuantDeg}}{{{codebook['HL']['mean_quantization_angle_deg']:.2f}$^\circ$}}",
        rf"\newcommand{{\VQQuantDeg}}{{{codebook['VQ']['mean_quantization_angle_deg']:.2f}$^\circ$}}",
        rf"\newcommand{{\HLStability}}{{{100 * codebook['HL']['symbol_stability']['0.005']:.1f}\%}}",
        rf"\newcommand{{\VQStability}}{{{100 * codebook['VQ']['symbol_stability']['0.005']:.1f}\%}}",
        rf"\newcommand{{\PosteriorTemperature}}{{{selected}}}",
    ]
    if "full_control" in extras:
        item = extras["full_control"]
        lines += [
            rf"\newcommand{{\FullSetAcc}}{{{100 * item['sethl_mean']:.1f}}}",
            rf"\newcommand{{\FullContAcc}}{{{100 * item['continuous_mean']:.1f}}}",
            rf"\newcommand{{\FullAccDelta}}{{{100 * item['paired_delta']:.2f}}}",
            rf"\newcommand{{\FullAccCI}}{{[{100 * item['ci95'][0]:.2f}, {100 * item['ci95'][1]:.2f}]}}",
        ]
    if "denoising" in extras:
        item = extras["denoising"]["metrics"]["flow_mse"]
        relative = -100 * item["paired_delta"] / item["baseline_mean"]
        lines += [
            rf"\newcommand{{\DenoiseSet}}{{{item['method_mean']:.4f}}}",
            rf"\newcommand{{\DenoiseCont}}{{{item['baseline_mean']:.4f}}}",
            rf"\newcommand{{\DenoiseDelta}}{{{item['paired_delta']:.5f}}}",
            rf"\newcommand{{\DenoiseCI}}{{[{item['ci95'][0]:.5f}, {item['ci95'][1]:.5f}]}}",
            rf"\newcommand{{\DenoiseRelative}}{{{relative:.2f}}}",
        ]
    if "video_features" in extras:
        item = extras["video_features"]
        delta = item["comparisons"]["sethl_minus_continuous"]
        lines += [
            rf"\newcommand{{\VideoFeatureSet}}{{{item['summary']['sethl']['mean']:.4f}}}",
            rf"\newcommand{{\VideoFeatureCont}}{{{item['summary']['continuous']['mean']:.4f}}}",
            rf"\newcommand{{\VideoFeatureDelta}}{{{delta['mean']:.4f}}}",
            rf"\newcommand{{\VideoFeatureCI}}{{[{delta['ci95'][0]:.4f}, {delta['ci95'][1]:.4f}]}}",
        ]
    args.output_tex.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
