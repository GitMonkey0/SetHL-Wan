from aggregate_completion_results import load

import json


def _report(path, representation, offset):
    path.write_text(json.dumps({
        "representation": representation,
        "mask_policy": "interval",
        "per_sample": [
            {"id": "a", "hidden_angular_deg": 10 + offset,
             "hidden_symbol_accuracy": .5},
            {"id": "b", "hidden_angular_deg": 20 + offset,
             "hidden_symbol_accuracy": .6},
        ],
    }))


def test_load_keeps_training_replicates_nested_within_source(tmp_path):
    paths = []
    for seed in range(3):
        path = tmp_path / f"seed{seed}.json"
        _report(path, "sethl", seed)
        paths.append(path)
    representation, policies, replicates = load(paths)
    assert representation == "sethl"
    assert replicates == 3
    assert len(policies["interval"]["a"]) == 3

