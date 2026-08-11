import argparse
import csv
from pathlib import Path

import numpy as np
import pytest
import torch

from analysis.compositional_transfer import (
    PROTOCOL_VERSION,
    _cross_validated_linear_score,
    _load_cub_training_labels,
    _write_manifest,
    array_sha256,
    build_train_manifest,
    conditional_geometry,
    deterministic_sample_matrix,
    evaluate_command,
    evaluate_pair_transfer,
    select_cub_family_representatives,
    summarize_command,
)


def _complete_binary_design(n_attributes=3, repeats=20):
    rows = []
    for value in range(2**n_attributes):
        row = [(value >> shift) & 1 for shift in range(n_attributes)]
        rows.extend([row] * repeats)
    return np.asarray(rows, dtype=np.uint8)


def test_manifest_is_deterministic_and_freezes_only_train_design_inputs():
    attributes = _complete_binary_design()
    names = ["alpha", "beta", "gamma"]
    arguments = dict(
        dataset="celeba",
        attributes=attributes,
        attribute_names=names,
        row_ids=np.arange(attributes.shape[0]),
        fold_seed=17,
        shot_seeds=[101, 102],
        primary_shot=4,
        dataset_source={"split": "train"},
    )
    first = build_train_manifest(**arguments)
    second = build_train_manifest(**arguments)

    assert first == second
    assert first["protocol_version"] == PROTOCOL_VERSION
    assert first["selection_split"] == "train"
    assert first["heldout_data_accessed"] is False
    assert first["train_attribute_sha256"] == array_sha256(attributes)
    assert first["eligible_pair_count"] == 6
    assert len(first["fold_assignment"]) == attributes.shape[0]
    assert all(
        min(value for fold_counts in pair["train_cell_counts_by_fold"] for value in fold_counts)
        >= 4
        for pair in first["pairs"]
    )


def test_cub_manifest_loader_does_not_parse_heldout_attribute_values(tmp_path):
    root = tmp_path / "CUB_200_2011"
    (root / "attributes").mkdir(parents=True)
    (root / "train_test_split.txt").write_text("1 1\n2 0\n", encoding="utf-8")
    (root / "attributes.txt").write_text(
        "1 has_color::red\n2 has_shape::round\n", encoding="utf-8"
    )
    (root / "attributes" / "image_attribute_labels.txt").write_text(
        "1 1 1 4 0.0\n1 2 0 4 0.0\n2\n",
        encoding="utf-8",
    )

    attributes, names, row_ids = _load_cub_training_labels(root)

    assert names == ["color=red", "shape=round"]
    assert row_ids.tolist() == [1]
    assert attributes.tolist() == [[1, 0]]


def test_cub_selection_uses_one_supported_value_per_family():
    base = _complete_binary_design(n_attributes=2, repeats=30)
    attributes = np.column_stack(
        (
            base[:, 0],
            np.zeros(base.shape[0], dtype=np.uint8),
            base[:, 1],
            np.ones(base.shape[0], dtype=np.uint8),
        )
    )
    names = [
        "wing_color=red",
        "wing_color=rare",
        "bill_shape=hooked",
        "bill_shape=constant",
    ]
    folds = np.tile(np.asarray([0, 1], dtype=np.uint8), base.shape[0] // 2)

    selected, records = select_cub_family_representatives(
        attributes,
        names,
        folds,
        minimum_per_class_per_fold=4,
    )

    assert selected == [2, 0]
    assert [record["family"] for record in records] == ["bill_shape", "wing_color"]
    assert [record["chosen"] for record in records] == [
        "bill_shape=hooked",
        "wing_color=red",
    ]


def _four_cell_features(flipped_destination=False, repeats=8):
    features = []
    target = []
    context = []
    for context_value in (0, 1):
        for target_value in (0, 1):
            target_sign = 2 * target_value - 1
            if flipped_destination and context_value == 1:
                target_sign *= -1
            point = [float(target_sign), float(2 * context_value - 1)]
            features.extend([point] * repeats)
            target.extend([target_value] * repeats)
            context.extend([context_value] * repeats)
    return (
        torch.tensor(features, dtype=torch.float32),
        np.asarray(target, dtype=np.uint8),
        np.asarray(context, dtype=np.uint8),
    )


def test_additive_geometry_has_unit_alignment_and_transported_margin():
    features, target, context = _four_cell_features()

    metrics = conditional_geometry(features, target, context, source_context=0)

    assert metrics["valid"] is True
    assert metrics["conditional_axis_cosine"] == pytest.approx(1.0)
    assert metrics["interaction_defect_normalized"] == pytest.approx(0.0)
    assert metrics["midpoint_drift_abs"] == pytest.approx(0.0)
    assert metrics["transported_margin"] == pytest.approx(1.0)
    assert metrics["target_context_abs_cosine"] == pytest.approx(0.0)
    assert metrics["target_capture_balanced"] == pytest.approx(1.0)


def test_conditional_axis_reversal_predicts_complete_transfer_failure():
    features, target, context = _four_cell_features(flipped_destination=True)
    geometry = conditional_geometry(features, target, context, source_context=0)

    rows = evaluate_pair_transfer(
        head_features=features,
        head_target=target,
        head_context=context,
        test_features=features,
        test_target=target,
        test_context=context,
        source_context=0,
        shot=4,
        shot_seeds=[7, 8],
        sampling_key=("synthetic", "axis-reversal", 0),
    )

    assert geometry["conditional_axis_cosine"] == pytest.approx(-1.0)
    assert geometry["transported_margin"] == pytest.approx(-1.0)
    assert all(row["source_id_balanced_accuracy"] == pytest.approx(1.0) for row in rows)
    assert all(row["source_ood_balanced_accuracy"] == pytest.approx(0.0) for row in rows)
    assert all(row["source_transfer_gap"] == pytest.approx(1.0) for row in rows)
    assert all(row["source_id_auroc"] == pytest.approx(1.0) for row in rows)
    assert all(row["source_ood_auroc"] == pytest.approx(0.0) for row in rows)


def test_sampling_is_keyed_and_reproducible_across_representations():
    pool = np.arange(100, dtype=np.int64)
    first = deterministic_sample_matrix(pool, 8, [1, 2, 3], "pair", 0, "positive")
    second = deterministic_sample_matrix(pool, 8, [1, 2, 3], "pair", 0, "positive")
    changed = deterministic_sample_matrix(pool, 8, [1, 2, 3], "pair", 1, "positive")

    assert np.array_equal(first, second)
    assert not np.array_equal(first, changed)
    assert all(len(np.unique(row)) == 8 for row in first)


def test_target_clustered_prediction_detects_geometry_beyond_capture():
    rows = []
    for index in range(30):
        geometry = (index - 14.5) / 14.5
        rows.append(
            {
                "target": f"target_{index:02d}",
                "source_ood_balanced_accuracy": 0.7 + 0.1 * geometry,
                "target_capture_balanced": float((index * 7) % 11) / 10.0,
                "conditional_axis_cosine": geometry,
                "interaction_defect_normalized": 0.0,
                "midpoint_drift_abs": 0.0,
                "transported_margin": geometry,
                "abs_train_phi": 0.0,
            }
        )

    capture = _cross_validated_linear_score(
        rows, ("target_capture_balanced",), seed=5
    )
    augmented = _cross_validated_linear_score(
        rows, ("target_capture_balanced", "transported_margin"), seed=5
    )

    assert augmented["r2"] > 0.99
    assert augmented["r2"] > capture["r2"]


def test_small_end_to_end_evaluation_checks_manifest_and_writes_crossfit_rows(tmp_path):
    train_attributes = _complete_binary_design(n_attributes=3, repeats=12)
    test_attributes = _complete_binary_design(n_attributes=3, repeats=4)
    names = ["alpha", "beta", "gamma"]
    train_ids = np.arange(train_attributes.shape[0], dtype=np.int64)
    test_ids = np.arange(test_attributes.shape[0], dtype=np.int64) + 1000
    manifest = build_train_manifest(
        dataset="celeba",
        attributes=train_attributes,
        attribute_names=names,
        row_ids=train_ids,
        fold_seed=23,
        shot_seeds=[11, 12],
        primary_shot=2,
        minimum_heldout_cell_count=2,
    )
    manifest_path, manifest_hash = _write_manifest(tmp_path / "manifest.json", manifest)

    def save_cache(path, split, attributes, row_ids):
        signs = 2.0 * attributes.astype(np.float32) - 1.0
        features = torch.from_numpy(signs).contiguous()
        metadata = {
            "protocol_version": PROTOCOL_VERSION,
            "dataset": "celeba",
            "split": split,
            "encoder_id": "synthetic_additive",
            "attribute_names": names,
        }
        torch.save(
            {
                "features": features,
                "attributes": torch.from_numpy(attributes.copy()),
                "row_ids": torch.from_numpy(row_ids.copy()),
                "metadata": metadata,
            },
            path,
        )

    train_cache = tmp_path / "train.pt"
    test_cache = tmp_path / "test.pt"
    save_cache(train_cache, "train", train_attributes, train_ids)
    save_cache(test_cache, "test", test_attributes, test_ids)
    output = tmp_path / "evaluation"
    evaluate_command(
        argparse.Namespace(
            manifest=str(manifest_path),
            manifest_sha256=manifest_hash,
            train_cache=str(train_cache),
            test_cache=str(test_cache),
            output_dir=str(output),
            device="cpu",
            shots=[2],
            whiten_rel_eig_threshold=1e-3,
        )
    )

    with (output / "geometry.csv").open(newline="", encoding="utf-8") as handle:
        geometry_rows = list(csv.DictReader(handle))
    with (output / "transfer.csv").open(newline="", encoding="utf-8") as handle:
        transfer_rows = list(csv.DictReader(handle))
    assert len(geometry_rows) == 6 * 2 * 2
    assert len(transfer_rows) == 6 * 2 * 2 * 2
    assert all(row["valid"] == "True" for row in transfer_rows)
    assert all(
        float(row["source_ood_balanced_accuracy"]) >= 0.75
        for row in transfer_rows
    )
    assert (output / "metadata.json").is_file()
    assert (output / "SHA256SUMS").is_file()

    summary_output = tmp_path / "summary"
    summarize_command(
        argparse.Namespace(
            evaluations=[str(output)],
            output_dir=str(summary_output),
            primary_shot=2,
            bootstrap_repetitions=10,
            bootstrap_seed=31,
        )
    )
    assert (summary_output / "model_summary.csv").is_file()
    assert (summary_output / "predictive_increment.csv").is_file()
    assert (summary_output / "paired_model_comparisons.csv").is_file()
    assert (summary_output / "context_heldout_accuracy.pdf").is_file()


def test_s2_launcher_requires_reviewed_manifests_and_runs_the_fixed_model_matrix():
    launcher = (
        Path(__file__).resolve().parents[1]
        / "analysis"
        / "run_compositional_transfer_s2.sh"
    ).read_text(encoding="utf-8")

    assert "--prepare" in launcher
    assert 'CELEBA_MANIFEST_SHA256 must be set' in launcher
    assert 'CUB_MANIFEST_SHA256 must be set' in launcher
    assert "heldout_evaluation_started=false" in launcher
    assert "vicreg_celeba_epoch1000" in launcher
    assert "ijepa_celeba_epoch1000" in launcher
    assert "vicreg_imagenet1k_resnet50" in launcher
    assert "supervised_imagenet1k_resnet50" in launcher
    assert 'GPU_LOCAL_VICREG="${GPU_LOCAL_VICREG:-0}"' in launcher
    assert 'GPU_LOCAL_IJEPA="${GPU_LOCAL_IJEPA:-1}"' in launcher
    assert 'GPU_OFFICIAL_VICREG="${GPU_OFFICIAL_VICREG:-2}"' in launcher
    assert 'GPU_SUPERVISED="${GPU_SUPERVISED:-3}"' in launcher
