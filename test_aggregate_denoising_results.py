import numpy as np

from aggregate_denoising_results import paired_summary


def test_paired_summary_averages_training_seeds_before_bootstrap():
    method = {
        "a": [{"flow_mse": 1.0}, {"flow_mse": 3.0}],
        "b": [{"flow_mse": 5.0}, {"flow_mse": 7.0}],
    }
    baseline = {
        "a": [{"flow_mse": 2.0}, {"flow_mse": 4.0}],
        "b": [{"flow_mse": 3.0}, {"flow_mse": 5.0}],
    }
    result = paired_summary(method, baseline, "flow_mse",
                            np.random.default_rng(0), 1000)
    assert result["method_mean"] == 4.0
    assert result["baseline_mean"] == 3.5
    assert result["paired_delta"] == 0.5
    assert result["paired_sources"] == 2
