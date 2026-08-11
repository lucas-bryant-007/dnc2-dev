import json
import tempfile
import unittest
from pathlib import Path

from analysis.compare_pretrained_crossfit import load_snapshot, write_comparison


ROOT = Path(__file__).resolve().parents[1]
ARCHIVED_VICREG = (
    ROOT
    / "paper_outputs"
    / "pretrained_crossfit_postaudit_20260810"
    / "metrics"
    / "hyperrect_crossfit_vicreg_celeba_epoch_1000_full_support_20x_v1.json"
)


def _payload(*, rmse=0.2, cap=None, triple=None, values=None):
    triple = triple or ["a", "b", "c"]
    values = values or [0.20, 0.21]
    records = [
        {
            "test_balance_seed": seed,
            "normalized_centroid_rmse": value,
        }
        for seed, value in zip((7, 8), values, strict=True)
    ]
    return {
        "dataset": "celeba",
        "method": "vicreg",
        "selection_succeeded": True,
        "selected_triple": triple,
        "protocol": {
            "analysis_protocol_version": "synthetic_independent_third_fold_v1",
            "primary_test_balance_seed": 7,
            "max_test_cell_samples": cap,
            "rewhitening": "independent third-fold exact whitening",
            "capture_and_cosine_estimator": "split-half cross-Gram",
        },
        "test_balance": {"samples_per_cell": 500 if cap else 788},
        "test_box_diagnostics": {"normalized_centroid_rmse": rmse},
        "test_stability": {
            "corner_fidelity_status": {"status": "valid_current_geometry"},
            "pass_rate": 1.0,
            "aggregate_crossfit_probe_geometry": {
                "capture_B": {"a": 0.2, "b": 0.3, "c": 0.4},
                "max_abs_cos": 0.1,
            },
            "records": records,
        },
    }


class ComparePretrainedCrossfitTest(unittest.TestCase):
    def _write(self, root, name, payload):
        path = root / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return str(path)

    def test_exact_protocol_reproduction_and_durable_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = self._write(root, "reference.json", _payload())
            fresh = self._write(
                root,
                "fresh.json",
                _payload(rmse=0.20005, values=[0.19, 0.20]),
            )
            output = root / "comparison"
            comparisons = write_comparison([reference], [fresh], output)
            item = comparisons[0]
            self.assertEqual(item["verdict"], "reproduced_within_tolerance")
            self.assertEqual(item["paired_stability"]["fresh_lower_count"], 2)
            self.assertTrue((output / "comparison.json").is_file())
            self.assertTrue((output / "comparison.csv").is_file())
            self.assertTrue((output / "COMPARISON.md").is_file())
            serialized = json.loads((output / "comparison.json").read_text())
            self.assertTrue(serialized["all_reproduced_within_tolerance"])

    def test_archived_reference_is_identified_as_legacy_whole_population_zca(
        self,
    ):
        snapshot = load_snapshot(ARCHIVED_VICREG)
        protocol = snapshot.analysis_protocol["protocol"]
        rewhitener = snapshot.analysis_protocol["train_rewhitener"]
        self.assertNotIn("analysis_protocol_version", protocol)
        self.assertIn(
            "selected jointly balanced training population",
            protocol["rewhitening"],
        )
        self.assertEqual(rewhitener["kind"], "zca")
        self.assertEqual(
            rewhitener["fit_population"],
            "uniform_over_selected_eight_label_cells",
        )

    def test_estimator_change_is_not_mislabeled_as_same_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference_payload = _payload()
            reference_payload["protocol"]["rewhitening"] = (
                "whole-population regularized ZCA"
            )
            reference_payload["protocol"].pop("analysis_protocol_version")
            reference = self._write(root, "reference.json", reference_payload)
            fresh = self._write(root, "fresh.json", _payload(rmse=0.1))
            output = root / "comparison"
            item = write_comparison([reference], [fresh], output)[0]
            self.assertFalse(item["same_analysis_protocol"])
            self.assertFalse(item["same_primary_protocol"])
            self.assertEqual(
                item["verdict"],
                "different_analysis_protocol_not_a_reproduction",
            )
            serialized = json.loads((output / "comparison.json").read_text())
            self.assertFalse(serialized["all_reproduced_within_tolerance"])

    def test_sampling_change_is_not_mislabeled_as_reproduction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = self._write(root, "reference.json", _payload())
            fresh = self._write(root, "fresh.json", _payload(cap=500, rmse=0.1))
            item = write_comparison(
                [reference],
                [fresh],
                root / "comparison",
            )[0]
            self.assertEqual(
                item["verdict"],
                "different_sampling_design_not_a_reproduction",
            )
            self.assertEqual(item["primary_rmse_direction"], "better")

    def test_selection_failure_is_retained_as_negative_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = self._write(root, "reference.json", _payload())
            failed_payload = {
                "dataset": "celeba",
                "method": "vicreg",
                "selection_succeeded": False,
                "protocol": {},
            }
            fresh = self._write(root, "fresh.json", failed_payload)
            snapshot = load_snapshot(fresh)
            self.assertFalse(snapshot.selection_succeeded)
            item = write_comparison(
                [reference],
                [fresh],
                root / "comparison",
            )[0]
            self.assertEqual(
                item["verdict"],
                "fresh_fixed-constraint_selection_failed",
            )

    def test_legacy_cub_effective_cap_is_recovered_from_primary_balance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = _payload(cap=350)
            payload["dataset"] = "cub200"
            payload["method"] = "vicreg_official_imagenet1k"
            del payload["protocol"]["max_test_cell_samples"]
            payload["test_balance"] = {"seed": 7, "samples_per_cell": 350}
            path = self._write(root, "legacy_cub.json", payload)
            snapshot = load_snapshot(path)
            self.assertEqual(snapshot.max_test_cell_samples, 350)
            self.assertEqual(
                snapshot.sampling_cap_source,
                "legacy_cub_effective_cap_from_primary_balance",
            )


if __name__ == "__main__":
    unittest.main()
