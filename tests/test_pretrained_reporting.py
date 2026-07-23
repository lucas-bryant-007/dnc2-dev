import json

import numpy as np

from analysis.build_results_manifest import build_manifest
from analysis.plot_crossfit_hyperrect import _centroid_batch_cloud, render_crossfit_json
from analysis.plot_pretrained_summary import (
    load_completed_runs,
    plot_interference,
    plot_summary,
)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _central(method):
    return {
        "method": method,
        "attribute": "Male",
        "epoch": 1000,
        "config": f"configs/eval/{method}_celeba_hf.yaml",
        "ckpt_path": f"/checkpoints/{method}.ckpt",
        "results": {
            "adaptive": {
                "k_eff": 8,
                "r_values": [2, 4],
                "B_r": {"2": 0.2, "4": 0.35},
                "tilde_V_pred": {"2": 2.0, "4": 1.0},
                "tilde_V_obs": {"2": 1.9, "4": 1.05},
            }
        },
    }


def _hyper(method):
    values = [0.2, 0.3, 0.25]
    return {
        "method": method,
        "epoch": 1000,
        "split": "test",
        "whitened": True,
        "tag": "test",
        "mean_abs_offdiag_cosine": 0.1,
        "metrics": [
            {
                "name": name,
                "capture_B": capture,
                "pos_frac": prevalence,
                "usable": True,
            }
            for name, capture, prevalence in zip(
                ("Attractive", "Male", "Smiling"),
                values,
                (0.5, 0.4, 0.5),
                strict=True,
            )
        ],
        "cosine_matrix": [[1.0, 0.1, 0.05], [0.1, 1.0, -0.15], [0.05, -0.15, 1.0]],
    }


def _bundle(tmp_path):
    for method in ("ijepa", "vicreg"):
        metrics = tmp_path / f"celeba_{method}_hf" / "metrics"
        _write_json(metrics / f"metrics_{method}_male_epoch_1000_test.json", _central(method))
        _write_json(
            metrics / f"hyperrect_{method}_celeba_epoch_1000_test.json",
            _hyper(method),
        )
    return tmp_path


def test_matched_report_writes_png_and_pdf(tmp_path):
    root = _bundle(tmp_path / "results")
    runs = load_completed_runs(root)

    summary = plot_summary(runs, root / "paper" / "summary")
    heatmap = plot_interference(runs, root / "paper" / "interference")

    assert {path.suffix for path in summary} == {".png", ".pdf"}
    assert {path.suffix for path in heatmap} == {".png", ".pdf"}
    assert all(path.stat().st_size > 0 for path in summary + heatmap)


def test_manifest_hashes_artifacts_and_records_run_commits(tmp_path):
    root = _bundle(tmp_path / "results")
    output = root / "reproducibility_manifest.json"

    manifest = build_manifest(
        root,
        output,
        run_commits={"celeba_ijepa_hf": "abc123"},
    )

    assert manifest["run_commits"]["celeba_ijepa_hf"] == "abc123"
    assert len(manifest["artifacts"]) == 4
    assert all(len(row["sha256"]) == 64 for row in manifest["artifacts"])
    assert {row["method"] for row in manifest["metric_records"]} == {"ijepa", "vicreg"}


def test_crossfit_json_renders_proposal_cube_with_genuine_point_sidecar(tmp_path):
    box = []
    predicted = []
    for cell in range(8):
        combo = [(cell >> 2) & 1, (cell >> 1) & 1, cell & 1]
        center = [2 * value - 1 for value in combo]
        box.append({"combo": combo, "count": 100, "center": center, "center_se": [0.01] * 3})
        predicted.append({"combo": combo, "center": center})
    payload = {
        "method": "vicreg",
        "epoch": 1000,
        "selected_triple": ["Smiling", "Heavy_Makeup", "Black_Hair"],
        "protocol": {
            "selection_split": "train",
            "evaluation_split": "test",
            "triple_frozen_before_test_label_analysis": True,
        },
        "test_evaluation": {
            "triple_names": ["Smiling", "Heavy_Makeup", "Black_Hair"],
            "box": box,
            "predicted_box": predicted,
        },
        "plot_points": {"artifact": "plot_data/points.npz"},
    }
    point_path = tmp_path / "plot_data" / "points.npz"
    point_path.parent.mkdir(parents=True)
    rng = np.random.default_rng(7)
    coords = np.concatenate(
        [rng.normal(center, 0.05, size=(12, 3)) for center in [row["center"] for row in box]]
    ).astype(np.float32)
    np.savez_compressed(
        point_path,
        coords=coords,
        granular_task=np.repeat(np.arange(8, dtype=np.int8), 12),
        triple_names=np.asarray(["Smiling", "Heavy_Makeup", "Black_Hair"]),
    )
    json_path = tmp_path / "metrics" / "crossfit.json"
    _write_json(json_path, payload)

    outputs = render_crossfit_json(json_path, tmp_path / "paper")

    assert {path.suffix for path in outputs} == {".png", ".pdf"}
    assert len(outputs) == 4
    assert any("train_predicted_box" in path.name for path in outputs)
    assert all(path.stat().st_size > 0 for path in outputs)


def test_centroid_batch_cloud_is_balanced_deterministic_and_uses_real_means():
    rng = np.random.default_rng(11)
    tasks = np.repeat(np.arange(8), 40)
    centers = np.asarray(
        [[2 * ((cell >> bit) & 1) - 1 for bit in (2, 1, 0)] for cell in range(8)],
        dtype=np.float32,
    )
    coords = np.concatenate(
        [rng.normal(centers[cell], 0.5, size=(40, 3)) for cell in range(8)]
    ).astype(np.float32)

    first, first_tasks = _centroid_batch_cloud(coords, tasks, batches_per_cell=5, seed=7)
    second, second_tasks = _centroid_batch_cloud(coords, tasks, batches_per_cell=5, seed=7)

    assert first.shape == (40, 3)
    assert np.array_equal(first, second)
    assert np.array_equal(first_tasks, second_tasks)
    assert np.array_equal(np.bincount(first_tasks, minlength=8), np.full(8, 5))
    for cell in range(8):
        assert np.allclose(first[first_tasks == cell].mean(axis=0), coords[tasks == cell].mean(axis=0))
