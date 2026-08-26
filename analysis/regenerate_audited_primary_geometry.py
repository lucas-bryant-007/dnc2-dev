"""Repair archived strict-run corners from recoverable primary-split exports.

The July 2026 JSON files serialized predicted corners with cross-task Gram
terms.  The theorem predicts axis coordinates directly as
``(2*y_t - 1) * sqrt(B_t)``.  The compact archive contains enough information
to repair the primary balanced split, but not the other 19 resamples: only the
primary projected coordinates were exported.  This command therefore repairs
the primary result and explicitly invalidates the obsolete resampling corner
statistics while retaining unaffected capture/cosine diagnostics.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any, Sequence


CORNER_METRIC_KEYS = (
    "centroid_rmse",
    "normalized_centroid_rmse",
    "max_centroid_error",
)


def _portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.name


def _feasible_candidate_count(run_dir: Path) -> int | None:
    pattern = re.compile(r"Balanced proxy candidates meeting train feasibility:\s*(\d+)")
    for log_path in sorted((run_dir / "logs").glob("*.log")):
        match = pattern.search(log_path.read_text(encoding="utf-8", errors="replace"))
        if match:
            return int(match.group(1))
    return None


def predicted_corners_from_capture(
    triple_names: Sequence[str], capture_by_name: dict[str, Any]
) -> list[dict[str, Any]]:
    """Construct the theorem's eight axis-aligned corners in binary order."""
    if len(triple_names) != 3:
        raise ValueError("A strict hyperrectangle run must have three tasks")
    capture = [float(capture_by_name[name]) for name in triple_names]
    if any(not math.isfinite(value) or value < 0 for value in capture):
        raise ValueError("All capture estimates must be finite and nonnegative")
    half_sides = [math.sqrt(value) for value in capture]
    corners: list[dict[str, Any]] = []
    for cell in range(8):
        combo = [(cell >> 2) & 1, (cell >> 1) & 1, cell & 1]
        corners.append(
            {
                "combo": combo,
                "center": [
                    (1.0 if bit else -1.0) * half_sides[index] for index, bit in enumerate(combo)
                ],
            }
        )
    return corners


def box_prediction_diagnostics(
    observed_box: Sequence[dict[str, Any]],
    predicted_box: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Recompute the serialized primary corner diagnostics without torch."""
    observed = {
        tuple(entry["combo"]): entry for entry in observed_box if entry.get("center") is not None
    }
    predicted = {
        tuple(entry["combo"]): entry for entry in predicted_box if entry.get("center") is not None
    }
    shared = sorted(set(observed) & set(predicted))
    if not shared:
        raise ValueError("Observed and predicted boxes share no complete corners")

    errors: list[float] = []
    squared_radii: list[float] = []
    counts: list[int] = []
    for combo in shared:
        obs = [float(value) for value in observed[combo]["center"]]
        pred = [float(value) for value in predicted[combo]["center"]]
        if len(obs) != 3 or len(pred) != 3:
            raise ValueError("Every corner center must have three coordinates")
        errors.append(
            math.sqrt(
                sum(
                    (left - right) ** 2
                    for left, right in zip(obs, pred, strict=True)
                )
            )
        )
        squared_radii.append(sum(value * value for value in pred))
        counts.append(int(observed[combo]["count"]))

    centroid_rmse = math.sqrt(sum(value * value for value in errors) / len(errors))
    predicted_rms_radius = math.sqrt(sum(squared_radii) / len(squared_radii))
    if predicted_rms_radius <= 0:
        raise ValueError("The repaired predicted box has zero RMS radius")
    return {
        "n_corners": len(shared),
        "centroid_rmse": centroid_rmse,
        "predicted_rms_radius": predicted_rms_radius,
        "normalized_centroid_rmse": centroid_rmse / predicted_rms_radius,
        "max_centroid_error": max(errors),
        "min_cell_count": min(counts),
        "per_corner_error": [
            {"combo": list(combo), "l2_error": error}
            for combo, error in zip(shared, errors, strict=True)
        ],
    }


def _invalidate_obsolete_resample_corners(stability: dict[str, Any]) -> None:
    records = stability.get("records") or []
    historical_records = []
    for record in records:
        historical_records.append(
            {
                "test_balance_seed": record.get("test_balance_seed"),
                **{key: record.get(key) for key in CORNER_METRIC_KEYS if key in record},
                "headline_criteria_passed": record.get("headline_criteria_passed"),
            }
        )
        for key in CORNER_METRIC_KEYS:
            record.pop(key, None)
        criteria = record.get("headline_criteria")
        if isinstance(criteria, dict):
            criteria.pop("normalized_centroid_rmse", None)
        record["headline_criteria_passed"] = None
        record["corner_fidelity_status"] = "invalidated_obsolete_geometry"

    statistics = stability.get("statistics") or {}
    historical_statistics = {
        key: statistics.pop(key) for key in CORNER_METRIC_KEYS if key in statistics
    }
    stability["historical_superseded_corner_fidelity"] = {
        "status": "invalid_do_not_report",
        "reason": (
            "These values used predicted corners containing cross-task Gram "
            "terms rather than coordinate_t=(2*y_t-1)*sqrt(B_t)."
        ),
        "statistics": historical_statistics,
        "records": historical_records,
        "pass_count": stability.get("pass_count"),
        "pass_rate": stability.get("pass_rate"),
        "all_resamples_passed": stability.get("all_resamples_passed"),
    }
    stability["pass_count"] = None
    stability["pass_rate"] = None
    stability["all_resamples_passed"] = None
    stability["corner_fidelity_status"] = {
        "status": "requires_full_feature_rerun",
        "predicted_corner_formula": "coordinate_t = (2*y_t-1)*sqrt(B_t)",
        "cross_task_gram_terms_used": False,
        "recoverable_primary_split_repaired": True,
        "resample_corner_statistics_available": False,
        "reason": (
            "The compact archive exports primary projected coordinates only; "
            "the full held-out features and sampled indices for the other "
            "balance seeds are unavailable."
        ),
    }


def repair_payload(
    payload: dict[str, Any], *, source_metrics_json: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a repaired deep copy and a compact regeneration summary."""
    repaired = copy.deepcopy(payload)
    triple = [str(value) for value in repaired["selected_triple"]]
    geometry = repaired["train_selection"]["crossfit_probe_geometry"]
    if not geometry.get("valid_positive_diagonal"):
        raise ValueError("Training split-half capture has a nonpositive diagonal")
    capture_by_name = geometry["capture_B"]
    predicted_box = predicted_corners_from_capture(triple, capture_by_name)

    box_reference = repaired["train_balance"]["box_reference"]
    box_reference["predicted_box"] = copy.deepcopy(predicted_box)
    box_reference["capture_B_used"] = {name: float(capture_by_name[name]) for name in triple}
    box_reference["predicted_box_construction"] = {
        "formula": "coordinate_t = (2*y_t-1)*sqrt(B_t)",
        "capture_source": "train_selection.crossfit_probe_geometry.capture_B",
        "cross_task_gram_terms_used": False,
    }
    repaired["train_selection"]["predicted_box"] = copy.deepcopy(predicted_box)
    repaired["test_evaluation"]["predicted_box"] = copy.deepcopy(predicted_box)

    diagnostics = box_prediction_diagnostics(repaired["test_evaluation"]["box"], predicted_box)
    repaired["test_box_diagnostics"] = diagnostics
    criterion = repaired["headline_criteria"]["normalized_centroid_rmse"]
    criterion["observed"] = diagnostics["normalized_centroid_rmse"]
    criterion["passed"] = diagnostics["normalized_centroid_rmse"] <= float(criterion["target"])
    repaired["headline_criteria_passed"] = all(
        bool(item["passed"]) for item in repaired["headline_criteria"].values()
    )

    _invalidate_obsolete_resample_corners(repaired["test_stability"])
    repaired["post_audit_repair"] = {
        "status": "primary_geometry_repaired_resampling_pending",
        "repair_date": "2026-08-10",
        "source_metrics_json": source_metrics_json,
        "predicted_corner_formula": "coordinate_t = (2*y_t-1)*sqrt(B_t)",
        "capture_source": "train_selection.crossfit_probe_geometry.capture_B",
        "serialized_geometry_consistent": True,
        "primary_plot_coordinates_reused_without_refitting": True,
        "full_resampling_rerun_required": True,
    }
    summary = {
        "dataset": repaired["dataset"],
        "method": repaired["method"],
        "selected_triple": triple,
        "capture_B_used": {name: float(capture_by_name[name]) for name in triple},
        "primary_normalized_centroid_rmse": diagnostics["normalized_centroid_rmse"],
        "headline_criteria_passed": repaired["headline_criteria_passed"],
    }
    return repaired, summary


def regenerate(run_jsons: Sequence[Path], output_dir: Path) -> list[dict[str, Any]]:
    metrics_dir = output_dir / "metrics"
    plot_dir = output_dir / "plot_data"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []

    for source_json in run_jsons:
        source_json = source_json.expanduser().resolve()
        payload = json.loads(source_json.read_text(encoding="utf-8"))
        repaired, summary = repair_payload(payload, source_metrics_json=_portable_path(source_json))
        feasible_candidates = _feasible_candidate_count(source_json.parent.parent)
        repaired["post_audit_repair"]["train_feasible_candidates"] = feasible_candidates
        summary["train_feasible_candidates"] = feasible_candidates

        source_artifact = source_json.parent.parent / payload["plot_points"]["artifact"]
        source_artifact = source_artifact.resolve()
        if not source_artifact.is_file():
            raise FileNotFoundError(f"Missing primary plot artifact: {source_artifact}")
        destination_artifact = plot_dir / source_artifact.name
        shutil.copy2(source_artifact, destination_artifact)
        repaired["plot_points"]["artifact"] = (
            Path("plot_data") / destination_artifact.name
        ).as_posix()

        destination_json = metrics_dir / source_json.name
        destination_json.write_text(json.dumps(repaired, indent=2) + "\n", encoding="utf-8")
        summary["metrics_json"] = destination_json.relative_to(output_dir).as_posix()
        summary["plot_points"] = destination_artifact.relative_to(output_dir).as_posix()
        summaries.append(summary)
        print(
            f"Repaired {summary['method']} / {summary['dataset']}: "
            f"primary normalized RMSE="
            f"{summary['primary_normalized_centroid_rmse']:.6f}"
        )

    status = {
        "status": "primary_geometry_repaired_resampling_pending",
        "repair_date": "2026-08-10",
        "runs": summaries,
        "regenerated": [
            "axis-aligned serialized predicted boxes",
            "primary-split corner diagnostics",
            "primary headline criteria",
        ],
        "pending_full_rerun": [
            "20-seed corrected corner-fidelity stability summaries",
            "few-shot bound curves requiring raw pairwise moments/features",
        ],
    }
    (output_dir / "STATUS.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    return summaries


def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.out_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}; pass --overwrite "
            "only to replace previously generated audit artifacts"
        )
    regenerate([Path(value) for value in args.run_json], output_dir)
    print(f"Saved repaired package data: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_json", nargs="+", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    main(parser.parse_args())
