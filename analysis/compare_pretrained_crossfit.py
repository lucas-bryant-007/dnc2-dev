"""Compare fresh pretrained cross-fit results with frozen reference artifacts.

The comparison deliberately distinguishes exact-protocol reproduction from a
different estimator or sampling design.  A lower point estimate is not called
a "reproduction" when the whitening/capture protocol, selected triple, or
per-cell sampling cap changed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


CURRENT_CORNER_STATUS = "valid_current_geometry"


@dataclass(frozen=True)
class Snapshot:
    dataset: str
    method: str
    source: str
    selection_succeeded: bool
    triple: tuple[str, ...] | None
    primary_seed: int | None
    max_test_cell_samples: int | None
    sampling_cap_source: str
    analysis_protocol: dict[str, Any]
    samples_per_cell: int | None
    primary_rmse: float | None
    aggregate_capture: dict[str, float]
    aggregate_max_abs_cos: float | None
    stability_status: str | None
    stability_rmse: dict[int, float]
    stability_pass_rate: float | None

    @property
    def key(self) -> tuple[str, str]:
        return self.dataset, self.method


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Expected a finite value, got {value!r}")
    return result


_PROTOCOL_IDENTITY_KEYS = (
    "analysis_protocol_version",
    "population",
    "population_estimand",
    "selection_split",
    "evaluation_split",
    "selection_objective",
    "min_class_frac",
    "min_capture",
    "cos_ceiling",
    "crop_to_official_bounding_box",
    "attribute_source",
    "attribute_family_constraint",
    "rewhitening",
    "analysis_whitening_option",
    "test_statistics_used_to_fit_rewhitening",
    "capture_and_cosine_estimator",
    "training_capture_interpretation",
    "headline_inference_source",
    "box_axes_and_predicted_corners",
    "fixed_test_criteria",
)


def _present_keys(source: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: source[key] for key in keys if key in source}


def _analysis_protocol_signature(payload: dict[str, Any]) -> dict[str, Any]:
    """Return estimator-defining metadata, excluding seeds and sample caps.

    The old comparator treated a matching triple, seed, and test sampling cap as
    a matching protocol.  That is insufficient: the archived package fitted a
    regularized ZCA map on the whole selected training population, whereas the
    audited pipeline fits exact rank-truncated whitening on an independent
    third fold.  Those procedures estimate different quantities even when the
    selected triple happens to be unchanged.
    """

    protocol = payload.get("protocol") or {}
    train_balance = payload.get("train_balance") or {}
    train_rewhitener = train_balance.get("rewhitener") or {}
    first_stage = payload.get("first_stage_ssl_whitener")
    signature: dict[str, Any] = {
        "protocol": _present_keys(protocol, _PROTOCOL_IDENTITY_KEYS),
        "train_estimator": _present_keys(
            train_balance,
            (
                "whitening_fit_samples_per_cell",
                "crossfit_samples_per_cell_a",
                "crossfit_samples_per_cell_b",
                "probe_estimator",
                "training_capture_statistical_interpretation",
                "predicted_box_capture_estimator",
                "predicted_box_capture_statistical_interpretation",
            ),
        ),
        "train_rewhitener": _present_keys(
            train_rewhitener,
            (
                "kind",
                "ridge_rel",
                "requested_rel_eig_threshold",
                "numerical_rel_eig_floor",
                "fit_split",
                "fit_population",
                "independent_of_split_half_probe_folds",
                "frozen_for_test",
            ),
        ),
    }
    if isinstance(first_stage, dict):
        signature["first_stage_ssl_whitener"] = _present_keys(
            first_stage,
            (
                "stage",
                "fit_split",
                "fit_population",
                "view_marginal",
                "requested_rank_cap",
                "relative_rank_cutoff",
                "transform_frozen_after_fit",
                "frozen_for_downstream_evaluation",
                "frozen_for_test",
            ),
        )
        fit_loader = first_stage.get("fit_loader")
        if isinstance(fit_loader, dict):
            signature["first_stage_fit_loader"] = _present_keys(
                fit_loader,
                (
                    "role",
                    "dataset_repository",
                    "dataset_split",
                    "dataset_name",
                    "augmentation_method",
                    "image_size",
                    "num_augmented_views_per_instance",
                    "collate_function",
                    "drop_last",
                    "sampler_class",
                    "sampler_replacement",
                    "distributed_sharding",
                    "covers_full_dataset_exactly_once_per_pass",
                ),
            )
    return signature


def load_snapshot(path: str | Path) -> Snapshot:
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    dataset = str(payload["dataset"]).lower()
    method = str(payload["method"]).lower()
    succeeded = bool(payload.get("selection_succeeded", True))
    protocol = payload.get("protocol") or {}
    analysis_protocol = _analysis_protocol_signature(payload)
    test_balance = payload.get("test_balance") or {}
    primary_seed = protocol.get("primary_test_balance_seed")
    if primary_seed is None and test_balance.get("seed") is not None:
        primary_seed = test_balance["seed"]
    if "max_test_cell_samples" in protocol:
        cap = protocol["max_test_cell_samples"]
        cap_source = "serialized_protocol"
    elif dataset == "cub200" and test_balance.get("samples_per_cell") is not None:
        # The archived CUB launcher froze a 350-example cap, but the legacy
        # result schema omitted the CLI value. Its primary balance record and
        # retained launcher jointly identify the effective cap.
        cap = test_balance["samples_per_cell"]
        cap_source = "legacy_cub_effective_cap_from_primary_balance"
    else:
        cap = None
        cap_source = "missing_from_legacy_protocol"
    if not succeeded:
        return Snapshot(
            dataset=dataset,
            method=method,
            source=str(source),
            selection_succeeded=False,
            triple=None,
            primary_seed=int(primary_seed) if primary_seed is not None else None,
            max_test_cell_samples=int(cap) if cap is not None else None,
            sampling_cap_source=cap_source,
            analysis_protocol=analysis_protocol,
            samples_per_cell=None,
            primary_rmse=None,
            aggregate_capture={},
            aggregate_max_abs_cos=None,
            stability_status=None,
            stability_rmse={},
            stability_pass_rate=None,
        )

    triple = tuple(str(value) for value in payload["selected_triple"])
    if len(triple) != 3:
        raise ValueError(f"Expected three selected factors in {source}")
    stability = payload.get("test_stability") or {}
    corner_status = str(
        (stability.get("corner_fidelity_status") or {}).get(
            "status",
            "legacy_unverified_geometry",
        )
    )
    aggregate = stability.get("aggregate_crossfit_probe_geometry") or {}
    capture = {
        str(name): float(value)
        for name, value in (aggregate.get("capture_B") or {}).items()
    }
    records = stability.get("records") or []
    stability_rmse: dict[int, float] = {}
    if corner_status == CURRENT_CORNER_STATUS:
        for record in records:
            seed = int(record["test_balance_seed"])
            stability_rmse[seed] = float(record["normalized_centroid_rmse"])
    diagnostics = payload.get("test_box_diagnostics") or {}
    return Snapshot(
        dataset=dataset,
        method=method,
        source=str(source),
        selection_succeeded=True,
        triple=triple,
        primary_seed=int(primary_seed) if primary_seed is not None else None,
        max_test_cell_samples=int(cap) if cap is not None else None,
        sampling_cap_source=cap_source,
        analysis_protocol=analysis_protocol,
        samples_per_cell=(
            int(test_balance["samples_per_cell"])
            if test_balance.get("samples_per_cell") is not None
            else None
        ),
        primary_rmse=_optional_float(diagnostics.get("normalized_centroid_rmse")),
        aggregate_capture=capture,
        aggregate_max_abs_cos=_optional_float(aggregate.get("max_abs_cos")),
        stability_status=corner_status,
        stability_rmse=stability_rmse,
        stability_pass_rate=_optional_float(stability.get("pass_rate")),
    )


def _index(paths: list[str], label: str) -> dict[tuple[str, str], Snapshot]:
    indexed: dict[tuple[str, str], Snapshot] = {}
    for path in paths:
        snapshot = load_snapshot(path)
        if snapshot.key in indexed:
            raise ValueError(f"Duplicate {label} result for {snapshot.key}")
        indexed[snapshot.key] = snapshot
    return indexed


def _metric_direction(delta: float | None, *, lower_is_better: bool) -> str | None:
    if delta is None:
        return None
    if delta == 0.0:
        return "unchanged"
    improved = delta < 0.0 if lower_is_better else delta > 0.0
    return "better" if improved else "worse"


def compare_snapshots(
    reference: Snapshot,
    fresh: Snapshot,
    *,
    reproduction_atol: float,
) -> dict[str, Any]:
    if reference.key != fresh.key:
        raise ValueError(f"Cannot compare {reference.key} with {fresh.key}")
    result: dict[str, Any] = {
        "dataset": fresh.dataset,
        "method": fresh.method,
        "reference_source": reference.source,
        "fresh_source": fresh.source,
        "reference_selection_succeeded": reference.selection_succeeded,
        "fresh_selection_succeeded": fresh.selection_succeeded,
        "reference_triple": reference.triple,
        "fresh_triple": fresh.triple,
        "same_selected_triple": reference.triple == fresh.triple,
        "reference_max_test_cell_samples": reference.max_test_cell_samples,
        "fresh_max_test_cell_samples": fresh.max_test_cell_samples,
        "reference_sampling_cap_source": reference.sampling_cap_source,
        "fresh_sampling_cap_source": fresh.sampling_cap_source,
        "reference_analysis_protocol": reference.analysis_protocol,
        "fresh_analysis_protocol": fresh.analysis_protocol,
        "same_analysis_protocol": (
            reference.analysis_protocol == fresh.analysis_protocol
        ),
        "same_sampling_cap": (
            reference.max_test_cell_samples == fresh.max_test_cell_samples
        ),
        "reproduction_atol": reproduction_atol,
    }
    if not fresh.selection_succeeded:
        result["verdict"] = "fresh_fixed-constraint_selection_failed"
        return result
    if not reference.selection_succeeded:
        result["verdict"] = "reference_selection_failed_no_numeric_comparison"
        return result
    if reference.primary_rmse is None or fresh.primary_rmse is None:
        raise ValueError("Successful runs must contain primary corner diagnostics")

    rmse_delta = fresh.primary_rmse - reference.primary_rmse
    result.update(
        {
            "reference_primary_rmse": reference.primary_rmse,
            "fresh_primary_rmse": fresh.primary_rmse,
            "primary_rmse_delta": rmse_delta,
            "primary_rmse_relative_delta": (
                rmse_delta / reference.primary_rmse
                if reference.primary_rmse != 0.0
                else None
            ),
            "primary_rmse_direction": _metric_direction(
                rmse_delta,
                lower_is_better=True,
            ),
            "reference_aggregate_max_abs_cos": reference.aggregate_max_abs_cos,
            "fresh_aggregate_max_abs_cos": fresh.aggregate_max_abs_cos,
            "fresh_stability_status": fresh.stability_status,
            "fresh_stability_pass_rate": fresh.stability_pass_rate,
        }
    )
    if (
        reference.aggregate_max_abs_cos is not None
        and fresh.aggregate_max_abs_cos is not None
    ):
        cosine_delta = (
            fresh.aggregate_max_abs_cos - reference.aggregate_max_abs_cos
        )
        result["aggregate_max_abs_cos_delta"] = cosine_delta
        result["aggregate_max_abs_cos_direction"] = _metric_direction(
            cosine_delta,
            lower_is_better=True,
        )

    capture_rows: dict[str, dict[str, Any]] = {}
    for name in sorted(set(reference.aggregate_capture) | set(fresh.aggregate_capture)):
        old = reference.aggregate_capture.get(name)
        new = fresh.aggregate_capture.get(name)
        delta = new - old if old is not None and new is not None else None
        capture_rows[name] = {
            "reference": old,
            "fresh": new,
            "delta": delta,
            "direction": _metric_direction(delta, lower_is_better=False),
        }
    result["aggregate_capture"] = capture_rows

    shared_seeds = sorted(
        set(reference.stability_rmse) & set(fresh.stability_rmse)
    )
    if shared_seeds:
        deltas = [
            fresh.stability_rmse[seed] - reference.stability_rmse[seed]
            for seed in shared_seeds
        ]
        result["paired_stability"] = {
            "n_shared_seeds": len(shared_seeds),
            "mean_rmse_delta": sum(deltas) / len(deltas),
            "fresh_lower_count": sum(delta < 0.0 for delta in deltas),
            "equal_count": sum(delta == 0.0 for delta in deltas),
            "fresh_higher_count": sum(delta > 0.0 for delta in deltas),
        }

    same_protocol = bool(
        result["same_analysis_protocol"]
        and result["same_selected_triple"]
        and result["same_sampling_cap"]
        and reference.primary_seed == fresh.primary_seed
    )
    result["same_primary_protocol"] = same_protocol
    result["reproduced_within_tolerance"] = bool(
        same_protocol and abs(rmse_delta) <= reproduction_atol
    )
    if not result["same_analysis_protocol"]:
        result["verdict"] = "different_analysis_protocol_not_a_reproduction"
    elif not result["same_selected_triple"]:
        result["verdict"] = "selected_triple_changed_not_directly_comparable"
    elif not result["same_sampling_cap"]:
        result["verdict"] = "different_sampling_design_not_a_reproduction"
    elif result["reproduced_within_tolerance"]:
        result["verdict"] = "reproduced_within_tolerance"
    elif rmse_delta < 0.0:
        result["verdict"] = "lower_primary_mismatch_than_reference"
    else:
        result["verdict"] = "higher_primary_mismatch_than_reference"
    return result


def _write_markdown(comparisons: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Pretrained cross-fit comparison",
        "",
        "Lower normalized corner RMSE and lower task-direction overlap are better; "
        "higher capture is better. Different estimators, selected triples, or "
        "sampling caps are reported as different estimands rather than ranked as "
        "reproductions.",
        "",
        (
            "| Dataset | Method | Protocol match | Triple match | Reference RMSE | "
            "Fresh RMSE | Delta | Verdict |"
        ),
        "|---|---|---|---|---:|---:|---:|---|",
    ]
    for item in comparisons:
        reference = item.get("reference_primary_rmse")
        fresh = item.get("fresh_primary_rmse")
        delta = item.get("primary_rmse_delta")
        lines.append(
            f"| {item['dataset']} | {item['method']} | "
            f"{item.get('same_analysis_protocol', False)} | "
            f"{item.get('same_selected_triple', False)} | "
            f"{reference:.6f} | {fresh:.6f} | {delta:+.6f} | "
            f"{item['verdict']} |"
            if reference is not None and fresh is not None and delta is not None
            else (
                f"| {item['dataset']} | {item['method']} | "
                f"{item.get('same_analysis_protocol', False)} | "
                f"{item.get('same_selected_triple', False)} |  |  |  | "
                f"{item['verdict']} |"
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_comparison(
    reference_paths: list[str],
    fresh_paths: list[str],
    output_dir: str | Path,
    *,
    reproduction_atol: float = 1e-4,
) -> list[dict[str, Any]]:
    if reproduction_atol < 0.0:
        raise ValueError("reproduction_atol must be nonnegative")
    references = _index(reference_paths, "reference")
    fresh = _index(fresh_paths, "fresh")
    missing = sorted(set(references) - set(fresh))
    extra = sorted(set(fresh) - set(references))
    if missing or extra:
        raise ValueError(f"Result-key mismatch: missing={missing}, extra={extra}")
    comparisons = [
        compare_snapshots(
            references[key],
            fresh[key],
            reproduction_atol=reproduction_atol,
        )
        for key in sorted(references)
    ]
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    all_reproduced = all(
        item.get("reproduced_within_tolerance", False) for item in comparisons
    )
    (destination / "comparison.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "reproduction_atol": reproduction_atol,
                "all_reproduced_within_tolerance": all_reproduced,
                "comparisons": comparisons,
                "reference_snapshots": [
                    asdict(references[key]) for key in sorted(references)
                ],
                "fresh_snapshots": [asdict(fresh[key]) for key in sorted(fresh)],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    with (destination / "comparison.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        fieldnames = [
            "dataset",
            "method",
            "same_analysis_protocol",
            "reference_primary_rmse",
            "fresh_primary_rmse",
            "primary_rmse_delta",
            "primary_rmse_relative_delta",
            "same_selected_triple",
            "same_sampling_cap",
            "reproduced_within_tolerance",
            "verdict",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for item in comparisons:
            writer.writerow({name: item.get(name) for name in fieldnames})
    _write_markdown(comparisons, destination / "COMPARISON.md")
    return comparisons


def main(args: argparse.Namespace) -> None:
    comparisons = write_comparison(
        args.reference_json,
        args.fresh_json,
        args.out_dir,
        reproduction_atol=args.reproduction_atol,
    )
    for item in comparisons:
        print(f"{item['method']}/{item['dataset']}: {item['verdict']}")
    print(f"Saved comparison to {Path(args.out_dir).expanduser().resolve()}")
    if args.require_reproduction:
        failures = [
            f"{item['method']}/{item['dataset']}={item['verdict']}"
            for item in comparisons
            if not item.get("reproduced_within_tolerance", False)
        ]
        if failures:
            raise SystemExit(
                "Fresh full-support results did not reproduce the frozen "
                f"reference: {', '.join(failures)}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_json", nargs="+", required=True)
    parser.add_argument("--fresh_json", nargs="+", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--reproduction_atol", type=float, default=1e-4)
    parser.add_argument("--require_reproduction", action="store_true")
    main(parser.parse_args())
