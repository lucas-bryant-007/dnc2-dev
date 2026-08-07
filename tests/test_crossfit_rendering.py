import json

import numpy as np

from analysis.plot_crossfit_hyperrect import (
    _centroid_batch_cloud,
    render_crossfit_json,
)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_crossfit_json_renders_points_and_frozen_prediction(tmp_path):
    box = []
    predicted = []
    for cell in range(8):
        combo = [(cell >> 2) & 1, (cell >> 1) & 1, cell & 1]
        center = [2 * value - 1 for value in combo]
        box.append(
            {
                "combo": combo,
                "count": 100,
                "center": center,
                "center_se": [0.01] * 3,
            }
        )
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
        [
            rng.normal(center, 0.05, size=(12, 3))
            for center in [row["center"] for row in box]
        ]
    ).astype(np.float32)
    np.savez_compressed(
        point_path,
        coords=coords,
        granular_task=np.repeat(np.arange(8, dtype=np.int8), 12),
        triple_names=np.asarray(["Smiling", "Heavy_Makeup", "Black_Hair"]),
    )
    json_path = tmp_path / "metrics" / "crossfit.json"
    _write_json(json_path, payload)

    outputs = render_crossfit_json(json_path, tmp_path / "rendered")

    assert {path.suffix for path in outputs} == {".png", ".pdf"}
    assert len(outputs) == 4
    assert any("train_predicted_box" in path.name for path in outputs)
    assert all(path.stat().st_size > 0 for path in outputs)


def test_centroid_batch_cloud_is_balanced_and_deterministic():
    rng = np.random.default_rng(11)
    tasks = np.repeat(np.arange(8), 40)
    centers = np.asarray(
        [
            [2 * ((cell >> bit) & 1) - 1 for bit in (2, 1, 0)]
            for cell in range(8)
        ],
        dtype=np.float32,
    )
    coords = np.concatenate(
        [rng.normal(centers[cell], 0.5, size=(40, 3)) for cell in range(8)]
    ).astype(np.float32)

    first, first_tasks = _centroid_batch_cloud(
        coords, tasks, batches_per_cell=5, seed=7
    )
    second, second_tasks = _centroid_batch_cloud(
        coords, tasks, batches_per_cell=5, seed=7
    )

    assert first.shape == (40, 3)
    assert np.array_equal(first, second)
    assert np.array_equal(first_tasks, second_tasks)
    assert np.array_equal(np.bincount(first_tasks, minlength=8), np.full(8, 5))
    for cell in range(8):
        observed = first[first_tasks == cell].mean(axis=0)
        expected = coords[tasks == cell].mean(axis=0)
        assert np.allclose(observed, expected)
