"""Re-render a cross-fitted hyper-rectangle JSON without model inference.

The headline rendering intentionally uses the same visual grammar as the
synthetic-data cube figures: colour, marker shape, and fill encode the three
binary factors.  The numerical observed/predicted overlay remains available in
the original run output; it is deliberately omitted here so it does not obscure
the eight observed CelebA centroids.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from box_viz import plot_box_3d
import metrics_io as mio


_CELEBA_LEVEL_LABELS = {
    "Smiling": ("not smiling", "smiling"),
    "Heavy_Makeup": ("none", "present"),
    "Black_Hair": ("no", "yes"),
}


def _load_plot_points(json_path: Path, payload: dict, explicit_path: Path | None):
    record = payload.get("plot_points") or {}
    if explicit_path is not None:
        path = explicit_path
    elif record.get("artifact"):
        path = json_path.parent.parent / record["artifact"]
    else:
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.int64)
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Plot-point artifact does not exist: {path}")
    with np.load(path, allow_pickle=False) as archive:
        coords = np.asarray(archive["coords"], dtype=np.float32)
        granular_task = np.asarray(archive["granular_task"], dtype=np.int64)
        point_triple = [str(value) for value in archive["triple_names"].tolist()]
    expected_triple = list(payload["selected_triple"])
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"Plot-point coords must have shape [N,3], got {coords.shape}")
    if granular_task.shape != (coords.shape[0],):
        raise ValueError("Plot-point granular_task does not align with coords")
    if np.any((granular_task < 0) | (granular_task > 7)):
        raise ValueError("Plot-point granular_task must contain only values 0..7")
    if point_triple != expected_triple:
        raise ValueError(
            f"Plot-point triple {point_triple} does not match selected {expected_triple}"
        )
    return coords, granular_task


def render_crossfit_json(
    json_path: Path,
    output_dir: Path,
    plot_points_path: Path | None = None,
) -> list[Path]:
    with json_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    protocol = payload.get("protocol") or {}
    if protocol.get("selection_split") != "train" or protocol.get("evaluation_split") != "test":
        raise ValueError("Expected a train-selection/test-evaluation cross-fit record")
    if not protocol.get("triple_frozen_before_test_label_analysis"):
        raise ValueError("The record does not certify that the triple was frozen before test")
    test = payload["test_evaluation"]
    triple = list(payload["selected_triple"])
    if test.get("triple_names") != triple:
        raise ValueError("Selected and evaluated triples do not match")
    if test.get("box") is None or test.get("predicted_box") is None:
        raise ValueError("Cross-fit record does not contain both observed and predicted boxes")

    method = mio.slug(payload["method"])
    epoch = payload["epoch"]
    stem = f"celeba_balanced_cube_{method}_epoch_{epoch}"
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [output_dir / f"{stem}.png", output_dir / f"{stem}.pdf"]
    coords, granular_task = _load_plot_points(json_path, payload, plot_points_path)
    plot_box_3d(
        coords,
        test["box"],
        granular_task,
        triple,
        [str(path) for path in outputs],
        predicted_box=None,
        per_task=160,
        axis_labels=[name.replace("_", " ") for name in triple],
        level_labels=[_CELEBA_LEVEL_LABELS.get(name, ("absent", "present")) for name in triple],
        show_samples=coords.shape[0] > 0,
        show_centroid_se=False,
        publication_compact=False,
        axis_label_positions=[(1.14, 0.0, 0.0), (0.0, 1.14, 0.0), (0.22, 0.0, 0.82)],
        sample_size=8,
        sample_alpha=0.20,
    )
    return outputs


def main(args: argparse.Namespace) -> None:
    json_path = Path(args.json).expanduser().resolve()
    if not json_path.is_file():
        raise SystemExit(f"Metrics JSON does not exist: {json_path}")
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else json_path.parent.parent / "paper_figures"
    )
    plot_points_path = (
        Path(args.plot_points).expanduser().resolve()
        if args.plot_points
        else None
    )
    for path in render_crossfit_json(json_path, output_dir, plot_points_path):
        print(f"Saved: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="Cross-fit metrics JSON")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument(
        "--plot_points",
        default=None,
        help="Optional NPZ override containing genuine held-out 3D coordinates",
    )
    main(parser.parse_args())
