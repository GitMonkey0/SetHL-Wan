#!/usr/bin/env python3
"""Validate a complete generation grid and write paper-ready result macros."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METHODS = {
    "raster": "Raster",
    "continuous": "Cont",
    "vq": "VQ",
    "hardhl": "Hard",
    "sethl_nohier": "NoHier",
    "sethl": "Set",
}


def collect(root: Path, method: str, expected_sources: set[str]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for path in sorted(root.glob(f"{method}_seed*/sample_*.json")):
        row = json.loads(path.read_text())
        grouped.setdefault(row["source"], []).append(row)
    if set(grouped) != expected_sources:
        missing = sorted(expected_sources - set(grouped))
        extra = sorted(set(grouped) - expected_sources)
        raise ValueError(f"{method}: source mismatch; missing={missing}, extra={extra}")
    for source, reports in grouped.items():
        seeds = [r.get("training_seed") for r in reports]
        if len(reports) != 3 or len(set(seeds)) != 3:
            raise ValueError(f"{method}/{source}: expected 3 distinct seeds, got {seeds}")
    return grouped


def value(report: dict, metric: str) -> float:
    if metric == "frontier_hypervolume":
        return float(report[metric])
    return float(report["operating_point"][metric])


def source_values(grouped: dict[str, list[dict]], metric: str,
                  sources: list[str]) -> np.ndarray:
    return np.asarray([
        np.nanmean([value(report, metric) for report in grouped[source]])
        for source in sources], dtype=float)


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator,
                   replicates: int) -> list[float]:
    values = values[np.isfinite(values)]
    draws = rng.integers(0, len(values), size=(replicates, len(values)))
    return np.percentile(values[draws].mean(1), [2.5, 97.5]).tolist()


def fmt(value_: float, percent: bool = False) -> str:
    return f"{100 * value_:.1f}" if percent else f"{value_:.2f}"


def fmt_pp(value_: float) -> str:
    """Percentage-point effect size with enough precision for paired CIs."""
    return f"{100 * value_:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--full-root", type=Path, required=True)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--completion-root", type=Path, required=True)
    parser.add_argument("--codebook-results", type=Path, required=True)
    parser.add_argument("--temperature-selection", type=Path, required=True)
    parser.add_argument("--denoising-results", type=Path, required=True)
    parser.add_argument("--video-feature-results", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-tex", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    sources = [row["id"] for row in json.loads(args.subset.read_text())["selected"]]
    expected = set(sources)
    data = {method: collect(args.root, method, expected) for method in METHODS}
    metrics = ("specified_hl_accuracy", "specified_angular_deg",
               "hidden_pairwise_diversity_deg", "localized_diversity_deg",
               "specified_pairwise_symbol_disagreement",
               "detection_rate", "temporal_acceleration_error", "frontier_hypervolume")
    rng = np.random.default_rng(args.seed)
    summary = {}
    arrays = {}
    for method, grouped in data.items():
        arrays[method] = {metric: source_values(grouped, metric, sources)
                          for metric in metrics}
        summary[method] = {
            metric: {"mean": float(np.nanmean(values)),
                     "ci95": bootstrap_mean(values, rng, args.bootstrap)}
            for metric, values in arrays[method].items()}
    comparisons = {}
    for baseline in ("continuous", "vq", "sethl_nohier", "hardhl", "raster"):
        differences = (arrays["sethl"]["frontier_hypervolume"] -
                       arrays[baseline]["frontier_hypervolume"])
        comparisons[f"sethl_minus_{baseline}"] = {
            "mean": float(differences.mean()),
            "ci95": bootstrap_mean(differences, rng, args.bootstrap)}
    secondary_comparisons = {}
    for baseline in ("continuous", "vq", "sethl_nohier"):
        secondary_comparisons[f"sethl_minus_{baseline}"] = {}
        for metric in ("specified_hl_accuracy", "specified_angular_deg",
                       "localized_diversity_deg",
                       "specified_pairwise_symbol_disagreement", "detection_rate"):
            differences = arrays["sethl"][metric] - arrays[baseline][metric]
            secondary_comparisons[f"sethl_minus_{baseline}"][metric] = {
                "mean": float(differences.mean()),
                "ci95": bootstrap_mean(differences, rng, args.bootstrap),
            }
    center_grouped = collect(args.root, "sethl_center", expected)
    center_hv = source_values(center_grouped, "frontier_hypervolume", sources)
    center_delta = arrays["sethl"]["frontier_hypervolume"] - center_hv
    comparisons["sethl_minus_center"] = {
        "mean": float(center_delta.mean()),
        "ci95": bootstrap_mean(center_delta, rng, args.bootstrap),
        "center_mean": float(center_hv.mean()),
    }
    full_sethl = collect(args.full_root, "sethl", expected)
    full_continuous = collect(args.full_root, "continuous", expected)
    full_sethl_acc = source_values(full_sethl, "specified_hl_accuracy", sources)
    full_cont_acc = source_values(full_continuous, "specified_hl_accuracy", sources)
    full_delta = full_sethl_acc - full_cont_acc
    full_ci = bootstrap_mean(full_delta, rng, args.bootstrap)
    full_control = {
        "sethl_mean": float(full_sethl_acc.mean()),
        "continuous_mean": float(full_cont_acc.mean()),
        "paired_delta": float(full_delta.mean()),
        "ci95": full_ci,
        "noninferiority_margin": -0.01,
        "passes": bool(full_ci[0] > -0.01),
    }
    falsification_gate = {
        "sethl_beats_continuous_hv": bool(
            comparisons["sethl_minus_continuous"]["ci95"][0] > 0),
        "sethl_beats_vq_hv": bool(
            comparisons["sethl_minus_vq"]["ci95"][0] > 0),
        "full_control_noninferior": full_control["passes"],
    }
    falsification_gate["passes"] = all(falsification_gate.values())
    completion_cont = json.loads(
        (args.completion_root / "final_completion_sethl_vs_continuous.json").read_text())
    completion_spherical = json.loads(
        (args.completion_root / "final_completion_sethl_vs_sethl_nohier.json").read_text())
    codebook = json.loads(args.codebook_results.read_text())
    calibration = json.loads(args.temperature_selection.read_text())
    denoising = json.loads(args.denoising_results.read_text())
    video_features = json.loads(args.video_feature_results.read_text())
    selected_tag = calibration["selected"].removeprefix("temp")
    selected_temperature = float(selected_tag.replace("p", "."))
    comp_angle = completion_cont["metrics"]["hidden_angular_deg"]
    comp_accuracy = completion_cont["metrics"]["hidden_symbol_accuracy"]
    comp_spherical_angle = completion_spherical["metrics"]["hidden_angular_deg"]
    output = {"sources": len(sources), "training_seeds": 3,
              "summary": summary, "primary_comparisons": comparisons,
              "secondary_comparisons": secondary_comparisons,
              "full_control": full_control,
              "falsification_gate": falsification_gate,
              "completion": {"sethl_vs_continuous": completion_cont,
                             "sethl_vs_no_spherical": completion_spherical},
              "codebook": codebook, "posterior_temperature": calibration,
              "denoising": denoising, "video_features": video_features}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n")

    lines = ["% Automatically generated from locked experiment reports."]
    symbolic_methods = ("vq", "hardhl", "sethl_nohier", "sethl")
    best = {
        "specified_hl_accuracy": max(
            symbolic_methods, key=lambda m: summary[m]["specified_hl_accuracy"]["mean"]),
        "specified_angular_deg": min(
            symbolic_methods, key=lambda m: summary[m]["specified_angular_deg"]["mean"]),
        "localized_diversity_deg": max(
            symbolic_methods, key=lambda m: summary[m]["localized_diversity_deg"]["mean"]),
        "frontier_hypervolume": max(
            symbolic_methods, key=lambda m: summary[m]["frontier_hypervolume"]["mean"]),
    }

    def table_value(method: str, metric: str, percent: bool = False) -> str:
        rendered = fmt(summary[method][metric]["mean"], percent)
        return f"\\textbf{{{rendered}}}" if method in symbolic_methods and best[metric] == method else rendered

    for method, prefix in METHODS.items():
        lines.extend([
            f"\\newcommand{{\\{prefix}Acc}}{{{table_value(method, 'specified_hl_accuracy', True)}}}",
            f"\\newcommand{{\\{prefix}Ang}}{{{table_value(method, 'specified_angular_deg')}}}",
            f"\\newcommand{{\\{prefix}Div}}{{{table_value(method, 'localized_diversity_deg')}}}",
            f"\\newcommand{{\\{prefix}AUC}}{{{table_value(method, 'frontier_hypervolume')}}}",
        ])
    primary = comparisons["sethl_minus_continuous"]
    vq_primary = comparisons["sethl_minus_vq"]
    spherical = comparisons["sethl_minus_sethl_nohier"]
    hard = comparisons["sethl_minus_hardhl"]
    center = comparisons["sethl_minus_center"]
    denoise_mse = denoising["metrics"]["flow_mse"]
    denoise_relative_reduction = (
        -100.0 * denoise_mse["paired_delta"] / denoise_mse["baseline_mean"])
    feature_sethl = video_features["summary"]["sethl"]["mean"]
    feature_cont = video_features["summary"]["continuous"]["mean"]
    feature_delta = video_features["comparisons"]["sethl_minus_continuous"]
    cont_secondary = secondary_comparisons["sethl_minus_continuous"]
    acc_delta = cont_secondary["specified_hl_accuracy"]
    angle_delta = cont_secondary["specified_angular_deg"]
    leak_delta = cont_secondary["specified_pairwise_symbol_disagreement"]
    detection_delta = cont_secondary["detection_rate"]
    localized_delta = cont_secondary["localized_diversity_deg"]
    vq_secondary = secondary_comparisons["sethl_minus_vq"]
    vq_acc_delta = vq_secondary["specified_hl_accuracy"]
    vq_leak_delta = vq_secondary["specified_pairwise_symbol_disagreement"]
    nohier_secondary = secondary_comparisons["sethl_minus_sethl_nohier"]
    spherical_acc_delta = nohier_secondary["specified_hl_accuracy"]
    spherical_angle_delta = nohier_secondary["specified_angular_deg"]
    lines.extend([
        f"\\newcommand{{\\PrimaryDelta}}{{{fmt(primary['mean'])}}}",
        f"\\newcommand{{\\PrimaryCI}}{{[{fmt(primary['ci95'][0])}, {fmt(primary['ci95'][1])}]}}",
        f"\\newcommand{{\\VQDelta}}{{{fmt(vq_primary['mean'])}}}",
        f"\\newcommand{{\\VQCI}}{{[{fmt(vq_primary['ci95'][0])}, {fmt(vq_primary['ci95'][1])}]}}",
        f"\\newcommand{{\\SphericalDelta}}{{{fmt(spherical['mean'])}}}",
        f"\\newcommand{{\\SphericalCI}}{{[{fmt(spherical['ci95'][0])}, {fmt(spherical['ci95'][1])}]}}",
        f"\\newcommand{{\\HardDelta}}{{{fmt(hard['mean'])}}}",
        f"\\newcommand{{\\HardCI}}{{[{hard['ci95'][0]:.3f}, {hard['ci95'][1]:.3f}]}}",
        f"\\newcommand{{\\CenterAUC}}{{{fmt(center['center_mean'])}}}",
        f"\\newcommand{{\\CenterDelta}}{{{fmt(center['mean'])}}}",
        f"\\newcommand{{\\CenterCI}}{{[{fmt(center['ci95'][0])}, {fmt(center['ci95'][1])}]}}",
        f"\\newcommand{{\\ContAccDelta}}{{{fmt_pp(acc_delta['mean'])}}}",
        f"\\newcommand{{\\ContAccCI}}{{[{fmt_pp(acc_delta['ci95'][0])}, {fmt_pp(acc_delta['ci95'][1])}]}}",
        f"\\newcommand{{\\ContAngDelta}}{{{fmt(angle_delta['mean'])}}}",
        f"\\newcommand{{\\ContAngCI}}{{[{fmt(angle_delta['ci95'][0])}, {fmt(angle_delta['ci95'][1])}]}}",
        f"\\newcommand{{\\ContLeakDelta}}{{{fmt_pp(leak_delta['mean'])}}}",
        f"\\newcommand{{\\ContLeakCI}}{{[{fmt_pp(leak_delta['ci95'][0])}, {fmt_pp(leak_delta['ci95'][1])}]}}",
        f"\\newcommand{{\\ContDetectionDelta}}{{{fmt_pp(detection_delta['mean'])}}}",
        f"\\newcommand{{\\ContDetectionCI}}{{[{fmt_pp(detection_delta['ci95'][0])}, {fmt_pp(detection_delta['ci95'][1])}]}}",
        f"\\newcommand{{\\ContLocalizedDelta}}{{{fmt(localized_delta['mean'])}}}",
        f"\\newcommand{{\\ContLocalizedCI}}{{[{fmt(localized_delta['ci95'][0])}, {fmt(localized_delta['ci95'][1])}]}}",
        f"\\newcommand{{\\VQAccDelta}}{{{fmt_pp(vq_acc_delta['mean'])}}}",
        f"\\newcommand{{\\VQAccCI}}{{[{fmt_pp(vq_acc_delta['ci95'][0])}, {fmt_pp(vq_acc_delta['ci95'][1])}]}}",
        f"\\newcommand{{\\VQLeakDelta}}{{{fmt_pp(vq_leak_delta['mean'])}}}",
        f"\\newcommand{{\\VQLeakCI}}{{[{fmt_pp(vq_leak_delta['ci95'][0])}, {fmt_pp(vq_leak_delta['ci95'][1])}]}}",
        f"\\newcommand{{\\SphericalAccDelta}}{{{fmt_pp(spherical_acc_delta['mean'])}}}",
        f"\\newcommand{{\\SphericalAccCI}}{{[{fmt_pp(spherical_acc_delta['ci95'][0])}, {fmt_pp(spherical_acc_delta['ci95'][1])}]}}",
        f"\\newcommand{{\\SphericalAngDelta}}{{{fmt(spherical_angle_delta['mean'])}}}",
        f"\\newcommand{{\\SphericalAngCI}}{{[{fmt(spherical_angle_delta['ci95'][0])}, {fmt(spherical_angle_delta['ci95'][1])}]}}",
        f"\\newcommand{{\\CompSetAng}}{{{fmt(comp_angle['method_mean'])}}}",
        f"\\newcommand{{\\CompContAng}}{{{fmt(comp_angle['baseline_mean'])}}}",
        f"\\newcommand{{\\CompSetAcc}}{{{fmt(comp_accuracy['method_mean'], True)}}}",
        f"\\newcommand{{\\CompContAcc}}{{{fmt(comp_accuracy['baseline_mean'], True)}}}",
        f"\\newcommand{{\\CompSphericalGain}}{{{fmt(-comp_spherical_angle['paired_delta'])}}}",
        f"\\newcommand{{\\HLQuantDeg}}{{{fmt(codebook['HL']['mean_quantization_angle_deg'])}$^\\circ$}}",
        f"\\newcommand{{\\VQQuantDeg}}{{{fmt(codebook['VQ']['mean_quantization_angle_deg'])}$^\\circ$}}",
        f"\\newcommand{{\\HLStability}}{{{fmt(codebook['HL']['symbol_stability']['0.005'], True)}\\%}}",
        f"\\newcommand{{\\VQStability}}{{{fmt(codebook['VQ']['symbol_stability']['0.005'], True)}\\%}}",
        f"\\newcommand{{\\PosteriorTemperature}}{{{selected_temperature:g}}}",
        f"\\newcommand{{\\DenoiseSet}}{{{denoise_mse['method_mean']:.4f}}}",
        f"\\newcommand{{\\DenoiseCont}}{{{denoise_mse['baseline_mean']:.4f}}}",
        f"\\newcommand{{\\DenoiseDelta}}{{{denoise_mse['paired_delta']:.4f}}}",
        f"\\newcommand{{\\DenoiseCI}}{{[{denoise_mse['ci95'][0]:.4f}, {denoise_mse['ci95'][1]:.4f}]}}",
        f"\\newcommand{{\\DenoiseRelative}}{{{denoise_relative_reduction:.2f}}}",
        f"\\newcommand{{\\VideoFeatureSet}}{{{feature_sethl:.3f}}}",
        f"\\newcommand{{\\VideoFeatureCont}}{{{feature_cont:.3f}}}",
        f"\\newcommand{{\\VideoFeatureDelta}}{{{feature_delta['mean']:.3f}}}",
        f"\\newcommand{{\\VideoFeatureCI}}{{[{feature_delta['ci95'][0]:.3f}, {feature_delta['ci95'][1]:.3f}]}}",
        (f"\\newcommand{{\\ResultSentence}}{{SetHL obtains "
         f"{fmt(summary['sethl']['frontier_hypervolume']['mean'])} hypervolume, versus "
         f"{fmt(summary['continuous']['frontier_hypervolume']['mean'])} for continuous "
         f"and {fmt(summary['vq']['frontier_hypervolume']['mean'])} for VQ-26.}}"),
        (f"\\newcommand{{\\FullControlSentence}}{{Under full articulation control, "
         f"SetHL obtains {fmt(full_control['sethl_mean'], True)}\\% HL accuracy versus "
         f"{fmt(full_control['continuous_mean'], True)}\\% for continuous control "
         f"(paired difference {fmt(full_control['paired_delta'] * 100)} points, "
         f"95\\% CI [{fmt(full_control['ci95'][0] * 100)}, "
         f"{fmt(full_control['ci95'][1] * 100)}]).}}"),
    ])
    args.output_tex.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
