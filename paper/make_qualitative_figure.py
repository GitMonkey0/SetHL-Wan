#!/usr/bin/env python3
"""Create the paper qualitative panel using a deterministic median example."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})


def video_frame(path: Path, index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"cannot read frame {index} from {path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def select_median_source(root: Path) -> str:
    by_source: dict[str, list[float]] = {}
    for path in root.glob("sethl_seed*/sample_*.json"):
        row = json.loads(path.read_text())
        if row.get("frontier_hypervolume") is not None:
            by_source.setdefault(row["source"], []).append(row["frontier_hypervolume"])
    complete = [(float(np.mean(values)), source) for source, values in by_source.items()
                if len(values) == 3]
    if len(complete) != 27:
        raise ValueError(f"expected 27 complete SetHL sources, got {len(complete)}")
    complete.sort()
    return complete[len(complete) // 2][1]


def render_mask(mask: np.ndarray, size: int = 256) -> np.ndarray:
    """Render the actual 17-by-40 partial program, without illustrative data."""
    cells = mask.reshape(mask.shape[0], -1).T
    panel = np.empty((*cells.shape, 3), dtype=np.uint8)
    panel[cells] = np.array([13, 148, 136], dtype=np.uint8)
    panel[~cells] = np.array([226, 232, 238], dtype=np.uint8)
    panel = cv2.resize(panel, (size, size), interpolation=cv2.INTER_NEAREST)
    # Thin white cell boundaries keep the binary program readable after the
    # full-width figure is reduced to the ICASSP page width.
    for x in np.linspace(0, size, mask.shape[0] + 1, dtype=int)[1:-1]:
        panel[:, max(0, x - 1):x + 1] = 255
    for y in np.linspace(0, size, cells.shape[0] + 1, dtype=int)[1:-1]:
        panel[max(0, y - 1):y + 1, :] = 255
    return panel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", help="optional source ID for diagnostic previews")
    args = parser.parse_args()

    source = args.source or select_median_source(args.generation_root)
    set_dir = args.generation_root / "sethl_seed2027"
    cont_dir = args.generation_root / "continuous_seed2027"
    set_report = next(path for path in set_dir.glob("sample_*.json")
                      if json.loads(path.read_text())["source"] == source)
    index = set_report.stem.removeprefix("sample_")
    set_gen = set_dir / f"sample_{index}"
    cont_gen = cont_dir / f"sample_{index}"
    spec = json.loads((set_gen / "generation.json").read_text())
    mask = np.asarray(spec["specified_mask"], dtype=bool)[0]
    # Select the center of the least-specified temporal region.
    fraction = mask.mean(axis=(1, 2))
    frame_index = int(np.flatnonzero(fraction == fraction.min())[len(np.flatnonzero(
        fraction == fraction.min())) // 2])

    window = torch.load(args.windows / f"{source}.pt", weights_only=True)
    rgb = window["rgb"].permute(0, 2, 3, 1).numpy()
    panels = [rgb[0], render_mask(mask), rgb[frame_index]]
    labels = ["Input frame", "Partial HL mask", "Held-out motion"]
    colors = ["#64748b", "#0d9488", "#64748b"]
    for method, directory, color in (
            ("Continuous", cont_gen, "#2563eb"), ("SetHL", set_gen, "#ea8c1b")):
        for seed in (0, 1):
            panels.append(video_frame(directory / f"{source}_scale1p0_seed{seed}.mp4",
                                      frame_index))
            labels.append(f"{method} {seed + 1}")
            colors.append(color)

    figure = plt.figure(figsize=(7.05, 3.45), dpi=180)
    grid = figure.add_gridspec(2, 4, hspace=.23, wspace=.055)
    axes = [figure.add_subplot(grid[0, column]) for column in range(3)]
    legend_axis = figure.add_subplot(grid[0, 3])
    axes.extend(figure.add_subplot(grid[1, column]) for column in range(4))
    for axis, panel, label, color in zip(axes, panels, labels, colors):
        axis.imshow(panel)
        axis.set_title(label, fontsize=9, color="#263746", pad=4, weight="semibold")
        axis.set_xticks([]); axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_linewidth(2.0); spine.set_edgecolor(color)
    legend_axis.axis("off")
    legend_axis.text(.08, .68, "specified", color="#0d9488", fontsize=10,
                     weight="bold", transform=legend_axis.transAxes)
    legend_axis.text(.08, .49, "free", color="#94a3b8", fontsize=10,
                     weight="bold", transform=legend_axis.transAxes)
    legend_axis.text(.08, .24, f"withheld frame  {frame_index}", color="#263746",
                     fontsize=9, transform=legend_axis.transAxes)
    figure.subplots_adjust(left=.005, right=.995, top=.96, bottom=.015)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight", pad_inches=.02)
    plt.close(figure)
    metadata = {"selection": ("explicit diagnostic source" if args.source else
                               "median source by three-seed SetHL hypervolume"),
                "source": source, "sample_index": int(index),
                "frame_index": frame_index, "control_scale": 1.0,
                "diffusion_seeds": [0, 1]}
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
