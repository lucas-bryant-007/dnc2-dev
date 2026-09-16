import itertools
import math

import numpy as np
import torch

from analysis.hyperrectangle import (
    CELLS,
    box_shape_diagnostics,
    corner_diagnostics,
    evaluate_all_triples,
    fit_ssl_map,
    select_train_triple,
    write_features,
)


def _box_cells(centers):
    return [{"combo": [(s + 1) // 2 for s in signs], "signs": list(signs), "count": 10,
             "center": list(map(float, center))}
            for signs, center in zip(CELLS, centers)]


def _predicted(capture):
    return [{"combo": [(s + 1) // 2 for s in signs], "signs": list(signs),
             "center": [signs[i] * math.sqrt(capture[i]) for i in range(3)]}
            for signs in CELLS]


def test_shape_decomposition_is_exact_and_zero_for_any_axis_aligned_box():
    capture = [0.5, 0.6, 0.7]
    signs = np.array(CELLS, float)
    # A shifted, rescaled but axis-aligned box: no non-box energy at all.
    centers = signs * np.array([0.9, 0.4, 0.8]) + np.array([0.1, -0.2, 0.05])
    shape = box_shape_diagnostics(_box_cells(centers), _predicted(capture))
    assert shape["non_box_share"] < 1e-12
    assert abs(shape["additive_share"] - 1) < 1e-12
    parts = shape["squared_error_parts"]
    assert abs(sum(parts.values()) - shape["mean_squared_corner_error"]) < 1e-12
    assert parts["tilted_edges"] < 1e-24 and parts["pair_interactions"] < 1e-24
    assert abs(parts["shift"] - (0.1 ** 2 + 0.2 ** 2 + 0.05 ** 2)) < 1e-12
    assert abs(shape["mean_squared_corner_error"]
               - corner_diagnostics(_box_cells(centers), _predicted(capture))["centroid_rmse"] ** 2) < 1e-12


def test_shape_decomposition_attributes_shear_and_interaction():
    signs = np.array(CELLS, float)
    shear = np.zeros((3, 3)); shear[0, 1] = 0.3  # attribute 0 leaks onto axis 1
    centers = signs @ (np.eye(3) * 0.8 + shear)
    shape = box_shape_diagnostics(_box_cells(centers), _predicted([0.64] * 3))
    parts = shape["squared_error_parts"]
    assert abs(parts["tilted_edges"] - 0.09) < 1e-12
    assert parts["pair_interactions"] < 1e-24
    assert abs(shape["non_box_share"] - 0.09 / (3 * 0.64 + 0.09)) < 1e-12
    assert shape["edge_direction_max_abs_cosine"] > 0.3

    interacting = signs * 0.8
    interacting[:, 2] += 0.2 * signs[:, 0] * signs[:, 1]
    shape = box_shape_diagnostics(_box_cells(interacting), _predicted([0.64] * 3))
    parts = shape["squared_error_parts"]
    assert abs(parts["pair_interactions"] - 0.04) < 1e-12
    assert parts["tilted_edges"] < 1e-24
    assert shape["edge_direction_max_abs_cosine"] < 1e-12


def _synthetic_features(n, seed, capture=(0.6, 0.5, 0.7), extra_attributes=2, dim=24):
    generator = torch.Generator().manual_seed(seed)
    labels = (torch.rand(n, 3 + extra_attributes, generator=generator) < 0.5).float()
    pm = 2 * labels - 1
    features = torch.randn(n, dim, generator=generator)
    for t in range(3):
        features[:, t] = pm[:, t] * math.sqrt(capture[t]) + math.sqrt(1 - capture[t]) * features[:, t]
    return features, labels


def test_all_triples_scores_every_supported_triple_and_recovers_planted_box():
    torch.manual_seed(0)
    names = ["a", "b", "c", "d", "e"]
    train_features, train_labels = _synthetic_features(24000, 1)
    test_features, test_labels = _synthetic_features(6000, 2)
    result = evaluate_all_triples(train_features, train_labels, test_features, test_labels, names)
    summary, records = result["summary"], result["triples"]
    assert summary["n_candidate_triples"] == len(list(itertools.combinations(range(5), 3)))
    # Every candidate is either scored or skipped for a recorded reason.
    assert summary["n_scored"] + sum(summary["skipped"].values()) == summary["n_candidate_triples"]
    assert summary["skipped"]["train_support"] == summary["skipped"]["test_support"] == 0
    planted = next(row for row in records if row["triple"] == ["a", "b", "c"])
    assert planted["normalized_centroid_rmse"] < 0.1
    assert planted["non_box_share"] < 0.02
    assert planted["train_criteria_passed"] and planted["test_criteria_passed"]
    # Label-free attributes have non-positive cross-fit capture (no box exists, skipped)
    # or spurious tiny capture (scored, but cannot pass the capture threshold).
    for row in records:
        if "d" in row["triple"] or "e" in row["triple"]:
            assert not row["train_criteria_passed"]
    assert summary["skipped"]["invalid_capture"] + sum(
        ("d" in r["triple"] or "e" in r["triple"]) for r in records) == 9
    assert summary["non_box_share"]["median"] is not None
    assert summary["fraction_passing_test_criteria"] == 1 / summary["n_scored"]


def test_fixed_attributes_bypasses_search_but_reports_criteria():
    names = ["a", "b", "c", "d", "e"]
    features, labels = _synthetic_features(24000, 3)
    # Give 'd' a weak but real signal: capture ~0.02, below the 0.10 train threshold.
    features[:, 3] += 0.15 * (2 * labels[:, 3] - 1)
    selection = select_train_triple(features, labels, names, fixed_attributes=["a", "b", "d"])
    assert selection["names"] == ["a", "b", "d"]
    assert selection["exact_attempts"][0]["mode"] == "fixed_attributes"
    assert selection["exact_attempts"][0]["passed"] is False
    assert 0 < selection["box"]["capture_B"][2] < 0.05


def test_ssl_map_fixed_dimension_and_keff():
    generator = torch.Generator().manual_seed(0)
    scale = torch.tensor([10.0, 3.0, 1.0] + [0.01] * 9)
    base = torch.randn(4000, 12, generator=generator) * scale
    first = base + 0.1 * torch.randn(4000, 12, generator=generator)
    second = base + 0.1 * torch.randn(4000, 12, generator=generator)
    _, default = fit_ssl_map(first, second)
    _, fixed = fit_ssl_map(first, second, covariance_dimension=5)
    _, keff = fit_ssl_map(first, second, covariance_dimension="keff")
    assert fixed["covariance_retained_dimension"] == 5
    assert fixed["covariance_dimension_rule"] == "fixed_dimension"
    assert default["covariance_dimension_rule"] == "relative_eigenvalue_cutoff"
    pr = default["effective_dimension_participation_ratio"]
    assert 1 < pr < 3  # dominated by the single 10x direction
    assert keff["covariance_retained_dimension"] == math.ceil(pr)
    assert default["dimension_for_99pct_variance"] >= 2


def test_write_features_round_trips_as_float16(tmp_path):
    features = torch.randn(10, 4)
    labels = torch.ones(10, 2)
    record = write_features(tmp_path / "f.npz", features=features, labels=labels,
                            attribute_names=np.asarray(["x", "y"]))
    loaded = np.load(tmp_path / "f.npz")
    assert loaded["features"].dtype == np.float16
    assert np.allclose(loaded["features"], features.numpy(), atol=2e-3)
    assert list(loaded["attribute_names"]) == ["x", "y"]
    assert record["arrays"]["features"] == [10, 4]
