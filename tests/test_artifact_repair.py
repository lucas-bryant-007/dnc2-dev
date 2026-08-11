import copy
import json

import pytest

from analysis.regenerate_audited_primary_geometry import (
    predicted_corners_from_capture,
    repair_payload,
)
from analysis.plot_strict_pretrained_paper import load_full_pipeline_null


TRIPLE = ["first", "second", "third"]
CAPTURE = {"first": 0.25, "second": 0.36, "third": 0.49}


def _payload():
    correct_box = predicted_corners_from_capture(TRIPLE, CAPTURE)
    observed_box = [{**copy.deepcopy(entry), "count": 100} for entry in correct_box]
    criteria = {
        "max_pairwise_abs_cos": {"target": 0.15, "observed": 0.0, "passed": True},
        "min_capture_B": {"target": 0.1, "observed": 0.25, "passed": True},
        "normalized_centroid_rmse": {
            "target": 0.25,
            "observed": 9.0,
            "passed": False,
        },
        "min_cell_count": {"target": 50, "observed": 100, "passed": True},
    }
    return {
        "dataset": "fixture",
        "method": "fixture",
        "selected_triple": TRIPLE,
        "train_balance": {
            "box_reference": {"predicted_box": [{"combo": [0, 0, 0], "center": [99.0, 99.0, 99.0]}]}
        },
        "train_selection": {
            "predicted_box": [],
            "crossfit_probe_geometry": {
                "valid_positive_diagonal": True,
                "capture_B": CAPTURE,
            },
        },
        "test_evaluation": {
            "box": observed_box,
            "predicted_box": [],
        },
        "test_box_diagnostics": {"normalized_centroid_rmse": 9.0},
        "headline_criteria": criteria,
        "headline_criteria_passed": False,
        "test_stability": {
            "n_resamples": 1,
            "pass_count": 1,
            "pass_rate": 1.0,
            "all_resamples_passed": True,
            "statistics": {
                "triple_max_abs_cos": {"mean": 0.0},
                "centroid_rmse": {"mean": 9.0},
                "normalized_centroid_rmse": {"mean": 9.0},
                "max_centroid_error": {"mean": 9.0},
            },
            "records": [
                {
                    "test_balance_seed": 7,
                    "centroid_rmse": 9.0,
                    "normalized_centroid_rmse": 9.0,
                    "max_centroid_error": 9.0,
                    "headline_criteria": copy.deepcopy(criteria),
                    "headline_criteria_passed": True,
                }
            ],
        },
    }


def test_repair_uses_axis_aligned_sqrt_capture_corners_everywhere():
    repaired, summary = repair_payload(_payload(), source_metrics_json="fixture/source.json")

    expected = predicted_corners_from_capture(TRIPLE, CAPTURE)
    assert repaired["train_balance"]["box_reference"]["predicted_box"] == expected
    assert repaired["train_selection"]["predicted_box"] == expected
    assert repaired["test_evaluation"]["predicted_box"] == expected
    assert repaired["test_box_diagnostics"]["normalized_centroid_rmse"] == pytest.approx(0)
    assert summary["primary_normalized_centroid_rmse"] == pytest.approx(0)
    assert repaired["headline_criteria_passed"] is True


def test_repair_invalidates_unrecoverable_resample_corner_values():
    repaired, _ = repair_payload(_payload(), source_metrics_json="fixture/source.json")
    stability = repaired["test_stability"]
    record = stability["records"][0]

    assert stability["corner_fidelity_status"]["status"] == "requires_full_feature_rerun"
    assert stability["pass_count"] is None
    assert "normalized_centroid_rmse" not in stability["statistics"]
    assert "normalized_centroid_rmse" not in record
    assert record["headline_criteria_passed"] is None
    assert stability["historical_superseded_corner_fidelity"]["status"] == "invalid_do_not_report"


def test_current_full_pipeline_null_schema_is_loadable_without_log(tmp_path):
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    path = metrics_dir / "null.json"
    path.write_text(
        json.dumps(
            {
                "method": "vicreg",
                "selection_succeeded": False,
                "protocol": {
                    "name": "full_pipeline_independent_column_label_permutation",
                    "label_randomization": {"train_seed": 3101},
                },
                "train_balance": {"feasible_proxy_candidate_count": 0},
            }
        ),
        encoding="utf-8",
    )
    loaded = load_full_pipeline_null(path)
    assert loaded.seed == 3101
    assert loaded.feasible_train_candidates == 0
    assert loaded.selection_succeeded is False
