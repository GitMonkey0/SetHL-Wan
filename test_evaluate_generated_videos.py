import pytest

from evaluate_generated_videos import compliance_diversity_hypervolume


def test_hypervolume_discards_dominated_points_and_uses_fixed_origin():
    frontier = [
        {"specified_hl_accuracy": 0.5, "localized_diversity_deg": 20.0},
        {"specified_hl_accuracy": 0.8, "localized_diversity_deg": 10.0},
        {"specified_hl_accuracy": 0.4, "localized_diversity_deg": 5.0},
    ]
    # [0,.5] is dominated at height 20 and [.5,.8] at height 10.
    assert compliance_diversity_hypervolume(frontier) == pytest.approx(13.0)


def test_hypervolume_is_nan_without_valid_points():
    assert compliance_diversity_hypervolume([]) != compliance_diversity_hypervolume([])
