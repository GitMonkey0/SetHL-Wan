import json

import pytest

from aggregate_generation_results import load, source_metric


def _report(source, seed, value):
    return {"source": source, "training_seed": seed, "representation": "x",
            "frontier_hypervolume": value, "means": {"specified_hl_accuracy": value}}


def test_groups_training_seeds_but_averages_within_source(tmp_path):
    paths = []
    for seed, value in [(1, 2.0), (2, 4.0), (3, 6.0)]:
        path = tmp_path / f"r{seed}.json"
        path.write_text(json.dumps(_report("clip", seed, value)))
        paths.append(path)
    grouped = load(paths, expected_training_seeds=3)
    assert source_metric(grouped["clip"], "frontier_hypervolume") == pytest.approx(4.0)


def test_rejects_missing_or_repeated_training_seed(tmp_path):
    paths = []
    for index in range(3):
        path = tmp_path / f"r{index}.json"
        path.write_text(json.dumps(_report("clip", 1, float(index))))
        paths.append(path)
    with pytest.raises(ValueError, match="distinct training seeds"):
        load(paths, expected_training_seeds=3)
