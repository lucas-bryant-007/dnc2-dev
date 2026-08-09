"""Re-render a cross-fitted hyper-rectangle JSON without model inference.

The headline rendering intentionally uses the same visual grammar as the
synthetic-data cube figures: colour, marker shape, and fill encode the three
binary factors.  The numerical observed/predicted overlay remains available in
the original run output; it is deliberately omitted here so it does not obscure
the eight held-out centroids.  When point coordinates are available, the
headline cloud consists of deterministic mini-batch centroids rather than raw
individual embeddings; this shows centroid stability without implying neural
collapse of the within-cell distributions.
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


def _centroid_batch_cloud(
    coords: np.ndarray,
    granular_task: np.ndarray,
    batches_per_cell: int = 24,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return means of disjoint deterministic mini-batches within each cell."""
    if batches_per_cell <= 0:
        raise ValueError("batches_per_cell must be positive")
    rng = np.random.default_rng(seed)
    cloud = []
    cloud_task = []
    for cell in range(8):
        cell_coords = coords[granular_task == cell]
        if cell_coords.shape[0] == 0:
            continue
        shuffled = cell_coords[rng.permutation(cell_coords.shape[0])]
        for batch in np.array_split(shuffled, min(batches_per_cell, len(shuffled))):
            cloud.append(batch.mean(axis=0))
            cloud_task.append(cell)
    if not cloud:
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.asarray(cloud, dtype=np.float32), np.asarray(cloud_task, dtype=np.int64)


def render_crossfit_json(
    json_path: Path,
    output_dir: Path,
    plot_points_path: Path | None = None,
    cloud_mode: str = "centroid_batches",
    clouds_per_cell: int = 24,
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
    dataset = mio.slug(payload.get("dataset") or "celeba")
    epoch = payload["epoch"]
    epoch_suffix = f"_epoch_{epoch}" if isinstance(epoch, int) else ""
    stem = f"{dataset}_balanced_cube_{method}{epoch_suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)
    cloud_outputs = [output_dir / f"{stem}.png", output_dir / f"{stem}.pdf"]
    prediction_stem = f"{dataset}_train_predicted_box_{method}{epoch_suffix}"
    prediction_outputs = [
        output_dir / f"{prediction_stem}.png",
        output_dir / f"{prediction_stem}.pdf",
    ]
    raw_coords, raw_task = _load_plot_points(json_path, payload, plot_points_path)
    if cloud_mode == "centroid_batches":
        coords, granular_task = _centroid_batch_cloud(
            raw_coords,
            raw_task,
            batches_per_cell=clouds_per_cell,
        )
        sample_size, sample_alpha = 18, 0.38
    elif cloud_mode == "individual":
        coords, granular_task = raw_coords, raw_task
        sample_size, sample_alpha = 8, 0.20
    elif cloud_mode == "none":
        coords = np.empty((0, 3), dtype=np.float32)
        granular_task = np.empty((0,), dtype=np.int64)
        sample_size, sample_alpha = 8, 0.20
    else:
        raise ValueError(f"Unknown cloud_mode: {cloud_mode}")
    plot_box_3d(
        coords,
        test["box"],
        granular_task,
        triple,
        [str(path) for path in cloud_outputs],
        predicted_box=None,
        per_task=clouds_per_cell if cloud_mode == "centroid_batches" else 160,
        axis_labels=[name.replace("_", " ") for name in triple],
        level_labels=[
            _CELEBA_LEVEL_LABELS.get(name, ("absent", "present"))
            for name in triple
        ],
        show_samples=coords.shape[0] > 0,
        show_centroid_se=False,
        publication_compact=False,
        axis_label_positions=[(1.14, 0.0, 0.0), (0.0, 1.14, 0.0), (0.22, 0.0, 0.82)],
        sample_size=sample_size,
        sample_alpha=sample_alpha,
    )
    # The second view is the actual held-out theory check: black is the observed
    # test box, while red dashed edges/diamonds are axes and corners fitted only
    # on the balanced training population. Keep clouds out of this panel so the
    # train-to-test discrepancy remains immediately legible.
    plot_box_3d(
        np.empty((0, 3), dtype=np.float32),
        test["box"],
        np.empty((0,), dtype=np.int64),
        triple,
        [str(path) for path in prediction_outputs],
        predicted_box=test["predicted_box"],
        axis_labels=[name.replace("_", " ") for name in triple],
        level_labels=[_CELEBA_LEVEL_LABELS.get(name, ("absent", "present")) for name in triple],
        show_samples=False,
        show_centroid_se=False,
        publication_compact=True,
    )
    return cloud_outputs + prediction_outputs


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
    for path in render_crossfit_json(
        json_path,
        output_dir,
        plot_points_path,
        cloud_mode=args.cloud_mode,
        clouds_per_cell=args.clouds_per_cell,
    ):
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
    parser.add_argument(
        "--cloud_mode",
        choices=("centroid_batches", "individual", "none"),
        default="centroid_batches",
        help="Render mini-batch centroid clouds, raw embeddings, or no cloud",
    )
    parser.add_argument(
        "--clouds_per_cell",
        type=int,
        default=24,
        help="Number of deterministic mini-batch centroids per cell",
    )
    main(parser.parse_args())
