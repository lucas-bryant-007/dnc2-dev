import hashlib
import json
import math
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "paper_outputs" / "pretrained_crossfit_postaudit_20260810"

RUNS = (
    (
        "hyperrect_crossfit_vicreg_celeba_epoch_1000_full_support_20x_v1.json",
        0.14330599343599543,
        "vicreg/heldout_permutation_null_vicreg_celeba.json",
    ),
    (
        "hyperrect_crossfit_ijepa_celeba_epoch_1000_full_support_20x_v1.json",
        0.25580642845632523,
        "ijepa/heldout_permutation_null_ijepa_celeba.json",
    ),
    (
        "hyperrect_crossfit_vicreg_official_imagenet1k_cub200_bbox_distinct_"
        "families_full_support_v3.json",
        0.29592962903412556,
        "cub200_vicreg/heldout_permutation_null_vicreg_official_imagenet1k_"
        "cub200.json",
    ),
)


@pytest.mark.parametrize("filename,expected_rmse,null_relative", RUNS)
def test_checked_postaudit_artifacts_use_one_consistent_axis_aligned_box(
    filename, expected_rmse, null_relative
):
    payload = json.loads((PACKAGE / "metrics" / filename).read_text(encoding="utf-8"))
    triple = payload["selected_triple"]
    capture = payload["train_selection"]["crossfit_probe_geometry"]["capture_B"]
    expected_half_sides = [math.sqrt(float(capture[name])) for name in triple]

    boxes = (
        payload["train_balance"]["box_reference"]["predicted_box"],
        payload["train_selection"]["predicted_box"],
        payload["test_evaluation"]["predicted_box"],
    )
    assert boxes[0] == boxes[1] == boxes[2]
    for entry in boxes[0]:
        for coordinate, expected in zip(
            entry["center"], expected_half_sides, strict=True
        ):
            assert abs(float(coordinate)) == pytest.approx(expected)

    observed_rmse = payload["test_box_diagnostics"]["normalized_centroid_rmse"]
    assert observed_rmse == pytest.approx(expected_rmse)
    assert (
        payload["test_stability"]["corner_fidelity_status"]["status"]
        == "requires_full_feature_rerun"
    )
    assert payload["test_stability"]["pass_count"] is None
    assert (
        payload["test_stability"]["historical_superseded_corner_fidelity"][
            "status"
        ]
        == "invalid_do_not_report"
    )

    null_payload = json.loads(
        (
            PACKAGE
            / "controls"
            / "heldout_label_permutation"
            / null_relative
        ).read_text(encoding="utf-8")
    )
    assert null_payload["observed_normalized_centroid_rmse"] == pytest.approx(
        expected_rmse, abs=2e-8
    )
    assert null_payload["n_permutations"] == 5000
    assert null_payload["empirical_lower_tail_p"] == pytest.approx(1 / 5001)


def test_historical_package_has_machine_readable_tombstone():
    status = json.loads(
        (
            ROOT
            / "paper_outputs"
            / "pretrained_crossfit_20260723"
            / "STATUS.json"
        ).read_text(encoding="utf-8")
    )
    assert status["status"] == "superseded_do_not_cite"
    assert "corner-fidelity values" in status["invalid"][0]


def test_postaudit_package_checksum_manifest():
    checksum_path = PACKAGE / "provenance" / "SHA256SUMS"
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", maxsplit=1)
        digest = hashlib.sha256((PACKAGE / relative).read_bytes()).hexdigest()
        assert digest == expected, relative
