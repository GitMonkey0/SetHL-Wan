import json
import sys

from summarize_generation_table import main


METHODS = ("raster", "continuous", "vq", "hardhl", "sethl_nohier", "sethl")
METRICS = (
    "specified_hl_accuracy", "specified_angular_deg",
    "hidden_pairwise_diversity_deg", "localized_diversity_deg",
    "specified_pairwise_symbol_disagreement", "detection_rate",
    "temporal_acceleration_error",
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def generation_report(source, seed, offset=0.0):
    operating = {
        "specified_hl_accuracy": 0.5 + offset,
        "specified_angular_deg": 20.0 - offset,
        "hidden_pairwise_diversity_deg": 10.0 + offset,
        "localized_diversity_deg": 8.0 + offset,
        "specified_pairwise_symbol_disagreement": 0.1 - offset / 10,
        "detection_rate": 0.9 + offset / 10,
        "temporal_acceleration_error": 1.0,
    }
    return {"source": source, "training_seed": seed,
            "operating_point": operating, "frontier_hypervolume": 4.0 + offset}


def test_complete_grid_generates_paper_macros(tmp_path, monkeypatch):
    interval = tmp_path / "interval"
    full = tmp_path / "full"
    sources = ("a", "b")
    seeds = (2027, 2028, 2029)
    for method_index, method in enumerate(METHODS):
        for seed in seeds:
            for sample_index, source in enumerate(sources):
                write_json(interval / f"{method}_seed{seed}" / f"sample_{sample_index}.json",
                           generation_report(source, seed, method_index / 100))
    for seed in seeds:
        for sample_index, source in enumerate(sources):
            write_json(interval / f"sethl_center_seed{seed}" / f"sample_{sample_index}.json",
                       generation_report(source, seed))
            write_json(full / f"sethl_seed{seed}" / f"sample_{sample_index}.json",
                       generation_report(source, seed, 0.01))
            write_json(full / f"continuous_seed{seed}" / f"sample_{sample_index}.json",
                       generation_report(source, seed))

    subset = tmp_path / "subset.json"
    write_json(subset, {"selected": [{"id": source} for source in sources]})
    completion = {
        "metrics": {
            "hidden_angular_deg": {"method_mean": 20.0, "baseline_mean": 21.0,
                                    "paired_delta": -1.0},
            "hidden_symbol_accuracy": {"method_mean": 0.6, "baseline_mean": 0.5,
                                       "paired_delta": 0.1},
        }
    }
    write_json(tmp_path / "final_completion_sethl_vs_continuous.json", completion)
    write_json(tmp_path / "final_completion_sethl_vs_sethl_nohier.json", completion)
    codebook = {name: {"mean_quantization_angle_deg": 10.0,
                       "symbol_stability": {"0.005": 0.8}}
                for name in ("HL", "VQ")}
    write_json(tmp_path / "codebook.json", codebook)
    write_json(tmp_path / "temperature.json", {"selected": "temp1"})
    write_json(tmp_path / "denoising.json", {"metrics": {"flow_mse": {
        "method_mean": 0.1, "baseline_mean": 0.2, "paired_delta": -0.1,
        "ci95": [-0.2, -0.01]}}})
    output_json, output_tex = tmp_path / "summary.json", tmp_path / "results.tex"
    monkeypatch.setattr(sys, "argv", ["summarize_generation_table.py",
        "--root", str(interval), "--full-root", str(full), "--subset", str(subset),
        "--completion-root", str(tmp_path), "--codebook-results", str(tmp_path / "codebook.json"),
        "--temperature-selection", str(tmp_path / "temperature.json"),
        "--denoising-results", str(tmp_path / "denoising.json"),
        "--output-json", str(output_json), "--output-tex", str(output_tex),
        "--bootstrap", "100"])
    main()
    summary = json.loads(output_json.read_text())
    assert summary["sources"] == 2
    assert summary["training_seeds"] == 3
    assert "passes" in summary["falsification_gate"]
    macros = output_tex.read_text()
    assert "pending" not in macros.lower()
    assert "\\newcommand{\\SetAUC}" in macros
    assert "\\newcommand{\\VQAccDelta}" in macros
    assert "\\newcommand{\\VQLeakCI}" in macros
    assert "\\newcommand{\\SphericalAccDelta}" in macros
    assert "\\newcommand{\\HardDelta}" in macros
    assert "\\newcommand{\\DenoiseCI}{[-0.2000, -0.0100]}" in macros
    assert "\\newcommand{\\DenoiseRelative}{50.00}" in macros
