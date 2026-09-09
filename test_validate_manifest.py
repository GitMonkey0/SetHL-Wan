import pytest

from validate_manifest import audit


def row(clip, split, source, start, end):
    return {"clip_id": clip, "split": split, "source_id": source,
            "start_frame": start, "end_frame": end,
            "rgb_path": "rgb.mp4", "pose_path": "pose.npz"}


def test_disjoint_sources_and_windows_are_valid():
    result = audit([row("a", "train", "s1", 0, 16),
                    row("b", "train", "s1", 17, 33),
                    row("c", "test", "s2", 0, 16)])
    assert result["valid"]


def test_cross_split_overlap_and_forbidden_source_are_rejected():
    result = audit([row("a", "train", "B1Counting", 0, 16),
                    row("b", "train", "B1Counting", 10, 26),
                    row("c", "test", "B1Counting", 30, 46)],
                   {"B1Counting", "B1Random"})
    assert not result["valid"]
    assert result["cross_split_sources"]
    assert result["overlapping_windows"]
    assert result["forbidden_source_hits"] == ["B1Counting"]
