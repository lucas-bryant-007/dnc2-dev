import csv
import json
from types import SimpleNamespace

import pytest
import torch

from analysis.celeba_hyperrect_crossfit import (
    _evaluate_balanced_test_seed,
    _summarize_stability,
    _write_stability_csv,
)
from analysis.plot_crossfit_stability import load_stability, plot_stability


TRIPLE = ["Smiling", "Heavy_Makeup", "Black_Hair"]


def _record(seed, offset=0.0, passed=True):
    return {
        "test_balance_seed": seed,
        "samples_per_cell": 500,
        "total_balanced_samples": 4000,
        "triple_max_abs_cos": 0.03 + offset,
        "capture_B": {
            "Smiling": 0.50 + offset,
            "Heavy_Makeup": 0.45 + offset,
            "Black_Hair": 0.40 + offset,
        },
        "min_capture_B": 0.40 + offset,
        "centroid_rmse": 0.05 + offset,
        "normalized_centroid_rmse": 0.04 + offset,
        "max_centroid_error": 0.08 + offset,
        "min_cell_count": 500,
        "headline_criteria": {
            "max_pairwise_abs_cos": {"target": 0.15},
            "min_capture_B": {"target": 0.10},
            "normalized_centroid_rmse": {"target": 0.25},
        },
        "headline_criteria_passed": passed,
    }


def test_stability_summary_records_spread_and_pass_rate():
    records = [_record(7), _record(8, 0.01), _record(9, 0.02, passed=False)]

    summary = _summarize_stability(records, TRIPLE)

    assert summary["n_resamples"] == 3
    assert summary["pass_count"] == 2
    assert summary["pass_rate"] == pytest.approx(2 / 3)
    assert not summary["all_resamples_passed"]
    assert summary["statistics"]["triple_max_abs_cos"]["mean"] == pytest.approx(0.04)
    assert summary["capture_B"]["Black_Hair"]["min"] == pytest.approx(0.40)
    assert summary["test_balance_seeds"] == [7, 8, 9]


def test_stability_csv_has_one_row_per_seed_and_named_capture_columns(tmp_path):
    path = tmp_path / "stability.csv"
    records = [_record(7), _record(8, 0.01)]

    _write_stability_csv(path, records, TRIPLE)

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["test_balance_seed"]) for row in rows] == [7, 8]
    assert float(rows[1]["capture_B_Black_Hair"]) == pytest.approx(0.41)
    assert rows[0]["headline_criteria_passed"] == "True"


def test_stability_plot_writes_png_and_pdf(tmp_path):
    records = [_record(7), _record(8, 0.01), _record(9, 0.02)]
    payload = {
        "method": "vicreg",
        "selected_triple": TRIPLE,
        "test_stability": {
            "records": records,
        },
    }
    json_path = tmp_path / "metrics.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_stability(json_path)
    outputs = plot_stability(loaded, tmp_path / "figures" / "stability")

    assert {path.suffix for path in outputs} == {".png", ".pdf"}
    assert all(path.stat().st_size > 0 for path in outputs)


def test_balanced_test_seed_uses_fixed_triple_and_per_cell_cap():
    generator = torch.Generator().manual_seed(4)
    combos = torch.tensor(
        [[(cell >> bit) & 1 for bit in (2, 1, 0)] for cell in range(8)]
    )
    attrs = combos.repeat_interleave(20, dim=0)
    signed = 2.0 * attrs.float() - 1.0
    features = torch.cat(
        [
            signed + 0.15 * torch.randn(160, 3, generator=generator),
            0.25 * torch.randn(160, 3, generator=generator),
        ],
        dim=1,
    )
    args = SimpleNamespace(
        max_test_cell_samples=10,
        min_class_frac=0.20,
        min_capture=0.10,
        cos_ceiling=0.12,
    )

    result, balance = _evaluate_balanced_test_seed(
        features,
        attrs,
        [0, 1, 2],
        TRIPLE,
        test_seed=17,
        args=args,
    )

    assert result["triple_names"] == TRIPLE
    assert [row["count"] for row in result["box"]] == [10] * 8
    assert balance["seed"] == 17
    assert balance["samples_per_cell"] == 10
    assert balance["total_balanced_samples"] == 80
