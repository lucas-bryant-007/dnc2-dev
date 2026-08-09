"""Held-out label-randomization test for frozen hyperrectangle predictions.

This analysis consumes the genuine held-out coordinates exported by the strict
cross-fit pipeline.  It keeps the selected held-out population, train-fitted
projection, and train-predicted corners fixed, then independently permutes the
three factor labels within that population.  The resulting null tests whether
the observed corner fidelity could arise without held-out feature/label
association.  It does not repeat train-time factor selection and is therefore
reported as a conditional held-out randomization test, not a full-pipeline
selection null.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def decode_cell_labels(cell_ids: np.ndarray) -> np.ndarray:
    """Decode cell ids 0..7 into the three binary factor columns."""
    cell_ids = np.asarray(cell_ids)
    if cell_ids.ndim != 1 or not np.all((0 <= cell_ids) & (cell_ids <= 7)):
        raise ValueError("cell_ids must be a one-dimensional array in 0..7")
    cell_ids = cell_ids.astype(np.int64, copy=False)
    return np.column_stack(
        ((cell_ids >> 2) & 1, (cell_ids >> 1) & 1, cell_ids & 1)
    )


def _cell_ids(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim != 2 or labels.shape[1] != 3:
        raise ValueError("labels must have shape [N, 3]")
    if not np.all((labels == 0) | (labels == 1)):
        raise ValueError("labels must be binary")
    return labels[:, 0] * 4 + labels[:, 1] * 2 + labels[:, 2]


def predicted_centers(payload: dict[str, Any]) -> np.ndarray:
    """Return frozen train-predicted centers ordered by cell id."""
    entries = payload["test_evaluation"]["predicted_box"]
    by_cell = {
        int(combo[0]) * 4 + int(combo[1]) * 2 + int(combo[2]): center
        for entry in entries
        if (center := entry.get("center")) is not None
        for combo in [entry["combo"]]
    }
    if set(by_cell) != set(range(8)):
        raise ValueError("predicted_box must contain one center for every cell")
    centers = np.asarray([by_cell[cell] for cell in range(8)], dtype=np.float64)
    if centers.shape != (8, 3) or not np.isfinite(centers).all():
        raise ValueError("predicted centers must be finite with shape [8, 3]")
    return centers


def normalized_corner_rmse(
    coords: np.ndarray,
    labels: np.ndarray,
    frozen_centers: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Compare label-cell centroids to the frozen predicted corners."""
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape [N, 3]")
    if labels.shape[0] != coords.shape[0]:
        raise ValueError("coords and labels must have the same number of rows")
    cell_ids = _cell_ids(labels)
    counts = np.bincount(cell_ids, minlength=8)
    if np.any(counts == 0):
        raise ValueError("all eight permuted cells must be nonempty")
    sums = np.zeros((8, 3), dtype=np.float64)
    np.add.at(sums, cell_ids, coords)
    observed_centers = sums / counts[:, None]
    rmse = float(
        np.sqrt(np.mean(np.sum((observed_centers - frozen_centers) ** 2, axis=1)))
    )
    predicted_radius = float(
        np.sqrt(np.mean(np.sum(frozen_centers * frozen_centers, axis=1)))
    )
    if predicted_radius <= 0:
        raise ValueError("frozen predicted corners have zero RMS radius")
    return rmse / predicted_radius, counts


def run_permutation_null(
    coords: np.ndarray,
    labels: np.ndarray,
    frozen_centers: np.ndarray,
    *,
    n_permutations: int,
    seed: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute the observed statistic and independent-column permutation null."""
    if n_permutations <= 0:
        raise ValueError("n_permutations must be positive")
    observed, observed_counts = normalized_corner_rmse(
        coords,
        labels,
        frozen_centers,
    )
    generator = np.random.default_rng(seed)
    null = np.empty(n_permutations, dtype=np.float64)
    for index in range(n_permutations):
        permuted = np.column_stack(
            [generator.permutation(labels[:, column]) for column in range(3)]
        )
        null[index], _ = normalized_corner_rmse(
            coords,
            permuted,
            frozen_centers,
        )
    return observed, null, observed_counts


def _load_inputs(json_path: Path, npz_path: Path | None):
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if npz_path is None:
        artifact = payload.get("plot_points", {}).get("artifact")
        if not artifact:
            raise ValueError("JSON does not identify a plot-point artifact")
        npz_path = json_path.parent.parent / artifact
    npz_path = npz_path.resolve()
    if not npz_path.is_file():
        raise FileNotFoundError(f"Plot-point artifact does not exist: {npz_path}")
    with np.load(npz_path) as data:
        coords = np.asarray(data["coords"], dtype=np.float64)
        labels = decode_cell_labels(data["granular_task"])
        exported_names = [str(value) for value in data["triple_names"]]
    selected_names = [str(value) for value in payload["selected_triple"]]
    if exported_names != selected_names:
        raise ValueError("NPZ triple names do not match the metrics JSON")
    return payload, coords, labels, predicted_centers(payload), npz_path


def _quantiles(values: np.ndarray) -> dict[str, float]:
    levels = (0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0)
    return {
        f"q{int(level * 100):02d}": float(value)
        for level, value in zip(levels, np.quantile(values, levels), strict=True)
    }


def _portable_path(path: Path) -> str:
    """Prefer a repository-relative artifact path over machine-specific paths."""
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.name


def _plot_null(
    null: np.ndarray,
    observed: float,
    title: str,
    output_stem: Path,
) -> list[Path]:
    figure, axis = plt.subplots(figsize=(5.6, 3.5))
    axis.hist(null, bins=45, color="#9aa0a6", alpha=0.85, edgecolor="white")
    axis.axvline(observed, color="#b3261e", linewidth=2.2, label="Observed")
    axis.axvline(
        float(np.median(null)),
        color="#3c4043",
        linewidth=1.5,
        linestyle="--",
        label="Permutation median",
    )
    axis.set_xlabel("Normalized frozen-corner RMSE (lower is better)")
    axis.set_ylabel("Label permutations")
    axis.set_title(title)
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    outputs = [output_stem.with_suffix(".png"), output_stem.with_suffix(".pdf")]
    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return outputs


def main(args) -> None:
    json_path = Path(args.json).expanduser().resolve()
    npz_path = Path(args.npz).expanduser().resolve() if args.npz else None
    payload, coords, labels, centers, resolved_npz = _load_inputs(json_path, npz_path)
    observed, null, observed_counts = run_permutation_null(
        coords,
        labels,
        centers,
        n_permutations=args.n_permutations,
        seed=args.seed,
    )
    saved_observed = float(payload["test_box_diagnostics"]["normalized_centroid_rmse"])
    if not np.isclose(observed, saved_observed, atol=1e-5, rtol=1e-5):
        raise ValueError(
            f"Recomputed observed RMSE {observed} does not match JSON {saved_observed}"
        )
    empirical_p = float((1 + np.count_nonzero(null <= observed)) / (len(null) + 1))
    result = {
        "dataset": payload["dataset"],
        "method": payload["method"],
        "selected_triple": payload["selected_triple"],
        "source_metrics_json": _portable_path(json_path),
        "source_plot_points": _portable_path(resolved_npz),
        "protocol": {
            "name": "conditional_heldout_independent_column_label_randomization",
            "selection_and_projection": "frozen from the strict train-only pipeline",
            "heldout_population": "fixed jointly balanced sample exported by the strict run",
            "permutation": "each of the three held-out factor columns independently permuted",
            "statistic": "normalized RMSE between permuted-label centroids and frozen train-predicted corners",
            "scope": "held-out association null; does not repeat train-time triple selection",
        },
        "seed": args.seed,
        "n_permutations": args.n_permutations,
        "observed_normalized_centroid_rmse": observed,
        "observed_cell_counts": [int(value) for value in observed_counts],
        "null_mean": float(np.mean(null)),
        "null_std": float(np.std(null, ddof=1)),
        "null_quantiles": _quantiles(null),
        "empirical_lower_tail_p": empirical_p,
        "null_statistics": [float(value) for value in null],
    }
    method = str(payload["method"]).lower().replace("-", "_")
    dataset = str(payload["dataset"]).lower().replace("-", "_")
    output_dir = Path(args.out_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"heldout_permutation_null_{method}_{dataset}"
    json_output = stem.with_suffix(".json")
    json_output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    title = f"{payload['method'].upper()} on {payload['dataset']}: held-out label null"
    figure_outputs = _plot_null(null, observed, title, stem)
    print(f"Observed normalized RMSE: {observed:.4f}")
    print(
        f"Permutation null: mean={np.mean(null):.4f}, "
        f"min={np.min(null):.4f}, p={empirical_p:.6g}"
    )
    print(f"Saved: {json_output}")
    for output in figure_outputs:
        print(f"Saved: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="Strict metrics JSON")
    parser.add_argument("--npz", default=None, help="Optional plot-point NPZ override")
    parser.add_argument("--n_permutations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--out_dir", required=True)
    main(parser.parse_args())
