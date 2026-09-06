import itertools
import json

import numpy as np
import torch

from analysis.hyperrectangle import (
    CELLS,
    plot_hyperrectangle,
    select_plot_points,
    write_plot_points,
)
from analysis.replot_hyperrectangle import replot_result


def synthetic_points():
    signs = torch.tensor(CELLS, dtype=torch.float32).repeat_interleave(30, dim=0)
    generator = torch.Generator().manual_seed(41)
    features = signs + 0.15 * torch.randn((len(signs), 3), generator=generator)
    return features, signs


def test_plot_points_are_balanced_genuine_and_deterministic(tmp_path):
    features, labels = synthetic_points()
    first = select_plot_points(features, labels, torch.eye(3), seed=7)
    second = select_plot_points(features, labels, torch.eye(3), seed=7)

    assert first["coordinates"].shape == (160, 3)
    assert torch.equal(first["coordinates"], second["coordinates"])
    assert torch.equal(first["test_row_indices"], second["test_row_indices"])
    assert torch.equal(
        torch.bincount(first["cell_indices"], minlength=8),
        torch.full((8,), 20),
    )
    assert len(torch.unique(first["test_row_indices"])) == 160
    assert torch.equal(first["coordinates"], features[first["test_row_indices"]])

    path = tmp_path / "points.npz"
    record = write_plot_points(path, first, ["a", "b", "c"])
    assert record["n_points"] == 160
    assert record["samples_per_cell"] == 20
    with np.load(path, allow_pickle=False) as archive:
        assert archive["coords"].shape == (160, 3)
        assert archive["granular_task"].shape == (160,)
        assert archive["signs"].shape == (160, 3)
        assert archive["test_row_indices"].shape == (160,)
        assert archive["triple_names"].tolist() == ["a", "b", "c"]


def test_plot_renders_samples_centroids_and_box(tmp_path):
    features, labels = synthetic_points()
    points = select_plot_points(features, labels, torch.eye(3), seed=7)
    observed = [
        {"signs": list(cell), "center": list(map(float, cell))}
        for cell in itertools.product((-1, 1), repeat=3)
    ]
    predicted = [
        {"signs": list(cell), "center": list(map(float, cell))}
        for cell in itertools.product((-1, 1), repeat=3)
    ]
    output = tmp_path / "figure.png"
    plot_hyperrectangle(
        output,
        ["a", "b", "c"],
        observed,
        predicted,
        sample_coordinates=points["coordinates"],
        sample_cells=points["cell_indices"],
    )
    assert output.is_file() and output.stat().st_size > 0
    assert output.with_suffix(".pdf").is_file()


def test_saved_sidecar_supports_cpu_only_replot(tmp_path):
    features, labels = synthetic_points()
    points = select_plot_points(features, labels, torch.eye(3), seed=7)
    points_path = tmp_path / "result_points.npz"
    record = write_plot_points(points_path, points, ["a", "b", "c"])
    box = [
        {"signs": list(cell), "center": list(map(float, cell))}
        for cell in itertools.product((-1, 1), repeat=3)
    ]
    payload = {
        "model": {"name": "vicreg_imagenet"},
        "selected_triple": ["a", "b", "c"],
        "plot_points": record,
        "test_evaluation": {
            "box": box,
            "predicted_box": box,
            "crossfit_probe_geometry": {"max_abs_cos": 0.05},
        },
        "test_box_diagnostics": {"normalized_centroid_rmse": 0.1},
        "test_side_length_diagnostics": {
            "mean_empirical_edge_length": 2.0,
            "mean_predicted_edge_length": 2.0,
        },
        "headline_criteria_passed": True,
    }
    json_path = tmp_path / "result.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")
    png, pdf = replot_result(json_path, tmp_path / "replot.png")
    assert png.is_file() and png.stat().st_size > 0
    assert pdf.is_file() and pdf.stat().st_size > 0
