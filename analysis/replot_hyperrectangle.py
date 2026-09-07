"""Regenerate a hyperrectangle figure from validated JSON and point sidecar."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from analysis.hyperrectangle import plot_hyperrectangle
except ModuleNotFoundError:  # Direct execution: python analysis/replot_hyperrectangle.py
    from hyperrectangle import plot_hyperrectangle


SUBTITLES = {
    "vicreg_celeba": "VICReg, pretrained on CelebA",
    "vicreg_imagenet": "VICReg, pretrained on ImageNet-1K",
    "ijepa_imagenet": "I-JEPA, pretrained on ImageNet-1K",
}


def bootstrap_centroid_cloud(coordinates, cells, *, points_per_cell=20,
                             batch_size=8, seed=7):
    """Return deterministic bootstrap mini-batch means for each joint-label cell."""
    if points_per_cell <= 0:
        raise ValueError("points_per_cell must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    rng = np.random.default_rng(seed)
    means = []
    mean_cells = []
    for cell in range(8):
        cell_coordinates = coordinates[cells == cell]
        if len(cell_coordinates) == 0:
            raise ValueError(f"cell {cell} has no saved coordinates")
        selections = rng.integers(
            0, len(cell_coordinates), size=(points_per_cell, batch_size))
        means.append(cell_coordinates[selections].mean(axis=1))
        mean_cells.append(np.full(points_per_cell, cell, dtype=np.int64))
    return np.concatenate(means), np.concatenate(mean_cells)


def replot_result(json_path, output_path, *, cloud_mode="minibatch",
                  cloud_points_per_cell=20, cloud_batch_size=8, cloud_seed=7,
                  sample_size=8, sample_alpha=0.24, cube_only=False):
    json_path = Path(json_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    record = payload.get("plot_points") or {}
    points_path = (json_path.parent / record["artifact"]).resolve()
    if points_path.parent != json_path.parent:
        raise ValueError("plot-point sidecar must be beside its result JSON")
    with np.load(points_path, allow_pickle=False) as archive:
        coordinates = np.asarray(archive["coords"], dtype=np.float32)
        cells = np.asarray(archive["granular_task"], dtype=np.int64)
        triple = [str(value) for value in archive["triple_names"].tolist()]
    if triple != payload["selected_triple"]:
        raise ValueError("point-sidecar triple does not match the result JSON")
    if cloud_mode == "minibatch":
        coordinates, cells = bootstrap_centroid_cloud(
            coordinates,
            cells,
            points_per_cell=cloud_points_per_cell,
            batch_size=cloud_batch_size,
            seed=cloud_seed,
        )
        samples_per_cell = cloud_points_per_cell
        sample_description = (
            f"bootstrap means: {cloud_points_per_cell}/cell, "
            f"n={cloud_batch_size} images/mean"
        )
    elif cloud_mode == "individual":
        samples_per_cell = record["samples_per_cell"]
        sample_description = f"{samples_per_cell} individual samples per cell"
    elif cloud_mode == "none":
        coordinates = np.empty((0, 3), dtype=np.float32)
        cells = np.empty((0,), dtype=np.int64)
        samples_per_cell = 0
        sample_description = None
    else:
        raise ValueError(f"unknown cloud mode: {cloud_mode}")
    model_name = payload["model"]["name"]
    geometry = payload["test_evaluation"]["crossfit_probe_geometry"]
    plot_hyperrectangle(
        output_path,
        payload["selected_triple"],
        payload["test_evaluation"]["box"],
        payload["test_evaluation"]["predicted_box"],
        subtitle=SUBTITLES.get(model_name, model_name),
        diagnostics=payload["test_box_diagnostics"],
        maximum_cosine=geometry["max_abs_cos"],
        side_lengths=payload["test_side_length_diagnostics"],
        passed=payload["headline_criteria_passed"],
        sample_coordinates=coordinates,
        sample_cells=cells,
        samples_per_cell=samples_per_cell,
        sample_size=sample_size,
        sample_alpha=sample_alpha,
        sample_description=sample_description,
        cube_only=cube_only,
    )
    return output_path.with_suffix(".png"), output_path.with_suffix(".pdf")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-size", type=float, default=8)
    parser.add_argument("--sample-alpha", type=float, default=0.24)
    parser.add_argument(
        "--cloud-mode", choices=("minibatch", "individual", "none"),
        default="minibatch")
    parser.add_argument("--cloud-points-per-cell", type=int, default=20)
    parser.add_argument("--cloud-batch-size", type=int, default=8)
    parser.add_argument("--cloud-seed", type=int, default=7)
    parser.add_argument(
        "--cube-only", action="store_true",
        help="Render only the cube and point cloud with zero outer padding")
    args = parser.parse_args()
    for path in replot_result(
            args.json, args.output,
            cloud_mode=args.cloud_mode,
            cloud_points_per_cell=args.cloud_points_per_cell,
            cloud_batch_size=args.cloud_batch_size,
            cloud_seed=args.cloud_seed,
            sample_size=args.sample_size, sample_alpha=args.sample_alpha,
            cube_only=args.cube_only):
        print(path)


if __name__ == "__main__":
    main()
