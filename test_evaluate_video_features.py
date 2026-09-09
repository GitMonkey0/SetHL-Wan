import numpy as np

from evaluate_video_features import summarize


def test_summarize_averages_videos_within_source_before_bootstrap():
    scores = {"a": [0.0, 1.0], "b": [1.0, 1.0, 1.0]}
    result = summarize(scores, np.random.default_rng(2027), bootstrap=200)
    assert result["mean"] == 0.75
    assert result["sources"] == 2
    assert result["videos"] == 5
    assert len(result["ci95"]) == 2
