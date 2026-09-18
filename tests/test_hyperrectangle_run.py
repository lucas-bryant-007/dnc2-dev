"""End-to-end smoke of ``run_experiment`` with synthetic features (no GPU, no dataset)."""
import argparse
import json
import math

import numpy as np
import torch

from analysis import hyperrectangle as hr

# Order must follow hyperrectangle.ATTRIBUTES, which fixes the label column order.
NAMES = ["Arched_Eyebrows", "Bangs", "Black_Hair", "Smiling", "Young"]


class FakeSplit:
    column_names = ["image", *NAMES]
    _fingerprint = "fake"

    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n


def _features(n, seed, capture=(0.5, 0.3, 0.7, 0.6, 0.02), dim=32):
    generator = torch.Generator().manual_seed(seed)
    labels = (torch.rand(n, len(NAMES), generator=generator) < 0.5).float()
    pm = 2 * labels - 1
    features = torch.randn(n, dim, generator=generator)
    for t, b in enumerate(capture):
        features[:, t] = pm[:, t] * math.sqrt(b) + math.sqrt(1 - b) * features[:, t]
    return torch.nn.functional.normalize(features, dim=1), labels


def _patch(monkeypatch):
    train_features, train_labels = _features(24000, 1)
    test_features, test_labels = _features(6000, 2)
    view_a, _ = _features(24000, 1)
    view_b = view_a + 0.05 * torch.randn_like(view_a)

    monkeypatch.setattr(hr, "load_celeba_splits", lambda cache: (FakeSplit(24000), FakeSplit(6000)))
    monkeypatch.setattr(hr, "load_encoder", lambda *a, **k: hr.EncoderAdapter(
        torch.nn.Identity(), False, {"name": "vicreg_celeba", "method": "vicreg",
                                     "pretraining_dataset": "celeba", "weights": "fake"}))
    monkeypatch.setattr(hr, "extract_paired_features", lambda *a, **k: (view_a, view_b))

    def extract(dataset, *a, **k):
        return (train_features, train_labels) if dataset.n == 24000 else (test_features, test_labels)

    monkeypatch.setattr(hr, "extract_dataset_features", extract)


def _args(out_dir, **overrides):
    base = dict(model="vicreg_celeba", weights=None, model_cache_dir=None, out_dir=str(out_dir),
                cache_dir=None, device="cpu", batch_size=None, transform_batch_size=4096,
                max_samples=None, ssl_dim=None, fixed_attributes=None, all_triples=False,
                save_features=False, features_dir=None)
    base.update(overrides)
    return argparse.Namespace(**base)


def test_run_with_all_triples_keff_and_saved_features(tmp_path, monkeypatch):
    _patch(monkeypatch)
    json_path, figure_path = hr.run_experiment(
        _args(tmp_path, ssl_dim="keff", all_triples=True, save_features=True))
    payload = json.loads(json_path.read_text())
    assert payload["selection_succeeded"] and figure_path.is_file()
    assert "Young" not in payload["selected_triple"]  # the zero-signal attribute
    assert payload["ssl_subspace"]["covariance_dimension_rule"] == "effective_dimension_99pct_variance"
    assert payload["ssl_subspace"]["covariance_retained_dimension"] == payload["ssl_subspace"]["dimension_for_99pct_variance"]
    assert payload["protocol"]["ssl_covariance_dimension_override"] == "keff"
    shape = payload["test_box_shape_diagnostics"]
    assert shape["non_box_share"] < 0.05
    assert abs(sum(shape["squared_error_parts"].values()) - shape["mean_squared_corner_error"]) < 1e-9

    summary = payload["all_triples_summary"]
    assert summary["n_candidate_triples"] == 10
    assert summary["n_scored"] + sum(summary["skipped"].values()) == 10
    assert (tmp_path / "hyperrectangle_vicreg_celeba_all_triples.csv").is_file()
    triples = json.loads((tmp_path / "hyperrectangle_vicreg_celeba_all_triples.json").read_text())
    best = min(triples["triples"], key=lambda row: row["normalized_centroid_rmse"])
    assert "Young" not in best["triple"]
    # Every scored triple carries its eight held-out centroids and predicted corners.
    assert all(len(row["centroids"]) == 8 for row in triples["triples"])
    import csv as _csv
    with (tmp_path / "hyperrectangle_vicreg_celeba_all_triples_centroids.csv").open() as handle:
        centroid_rows = list(_csv.DictReader(handle))
    assert len(centroid_rows) == 8 * summary["n_scored"]
    first = centroid_rows[0]
    assert {first["label_1"], first["label_2"], first["label_3"]} <= {"-1", "1"}
    recomputed = math.sqrt(sum(
        sum((float(r[f"observed_{i}"]) - float(r[f"predicted_{i}"])) ** 2 for i in (1, 2, 3))
        for r in centroid_rows[:8]) / sum(
        sum(float(r[f"predicted_{i}"]) ** 2 for i in (1, 2, 3)) for r in centroid_rows[:8]))
    assert abs(recomputed - float(first["normalized_centroid_rmse"])) < 1e-6

    for key in ("train_paired_views", "train", "test"):
        assert (tmp_path / payload["feature_artifacts"][key]["artifact"]).is_file()
    with np.load(tmp_path / "features_vicreg_celeba_test.npz") as archive:
        assert archive["features"].shape == (6000, 32) and archive["features"].dtype == np.float16
        assert archive["labels"].shape == (6000, 5)
        assert archive["attribute_names"].tolist() == NAMES


def test_run_with_fixed_attributes_records_mode(tmp_path, monkeypatch):
    _patch(monkeypatch)
    json_path, _ = hr.run_experiment(
        _args(tmp_path, fixed_attributes=["Smiling", "Black_Hair", "Bangs"], ssl_dim="16"))
    payload = json.loads(json_path.read_text())
    assert payload["selected_triple"] == ["Smiling", "Black_Hair", "Bangs"]
    assert payload["protocol"]["selection_mode"] == "fixed_attributes"
    assert payload["ssl_subspace"]["covariance_retained_dimension"] == 16
    assert payload["train_selection"]["exact_train_candidate_attempts"][0]["mode"] == "fixed_attributes"


def test_saved_features_reproduce_the_encoded_run_without_an_encoder(tmp_path, monkeypatch):
    _patch(monkeypatch)
    first_json, _ = hr.run_experiment(_args(tmp_path / "gpu", ssl_dim="16", save_features=True))

    def forbidden(*a, **k):
        raise AssertionError("encoder or dataset touched in saved-feature mode")

    for name in ("load_encoder", "load_celeba_splits", "extract_paired_features",
                 "extract_dataset_features"):
        monkeypatch.setattr(hr, name, forbidden)
    second_json, _ = hr.run_experiment(
        _args(tmp_path / "cpu", ssl_dim="16", features_dir=str(tmp_path / "gpu"), all_triples=True))
    first, second = json.loads(first_json.read_text()), json.loads(second_json.read_text())
    assert second["selected_triple"] == first["selected_triple"]
    # float16 storage perturbs features slightly; geometry must agree closely.
    assert abs(second["test_box_diagnostics"]["normalized_centroid_rmse"]
               - first["test_box_diagnostics"]["normalized_centroid_rmse"]) < 5e-3
    assert second["model"]["features_loaded_from"].endswith("gpu")
    assert second["samples"]["train"] == 24000 and second["samples"]["test"] == 6000
    assert second["all_triples_summary"]["n_scored"] >= 1
