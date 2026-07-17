import math

import torch

from analysis.hyperrect import (
    balanced_joint_indices,
    balanced_triple_proxy,
    box_prediction_diagnostics,
    orthonormal_basis,
    subclass_box,
    task_axis_basis,
)


def test_task_axis_basis_normalizes_without_qr_rotation():
    directions = torch.tensor(
        [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]]
    )
    task_axes = task_axis_basis(directions)
    qr_axes = orthonormal_basis(directions)
    assert torch.allclose(task_axes.norm(dim=0), torch.ones(3))
    assert torch.allclose(task_axes[:, 1], directions[:, 1] / directions[:, 1].norm())
    assert not torch.allclose(task_axes, qr_axes)


def test_task_axis_basis_rejects_zero_direction():
    directions = torch.eye(3)
    directions[:, 1] = 0
    try:
        task_axis_basis(directions)
    except ValueError as error:
        assert "nonzero" in str(error)
    else:
        raise AssertionError("zero task direction should fail")


def test_subclass_box_records_centroid_standard_errors():
    features = torch.tensor(
        [[-1.0, -1.0, -1.0], [-0.8, -1.2, -1.0], [1.0, 1.0, 1.0], [1.2, 0.8, 1.0]]
    )
    labels = torch.tensor([[0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1]])
    _coords, box, _tasks = subclass_box(features, labels, torch.eye(3))

    populated = [entry for entry in box if entry["count"]]
    assert len(populated) == 2
    assert all(entry["center_se"] is not None for entry in populated)
    assert math.isclose(populated[0]["center_se"][0], 0.1, rel_tol=1e-5)


def test_box_prediction_diagnostics_reports_corner_error_and_counts():
    observed = [
        {"combo": [0, 0, 0], "count": 20, "center": [0.0, 0.0, 0.0]},
        {"combo": [1, 1, 1], "count": 10, "center": [1.0, 1.0, 1.0]},
    ]
    predicted = [
        {"combo": [0, 0, 0], "center": [0.0, 0.0, 0.0]},
        {"combo": [1, 1, 1], "center": [2.0, 1.0, 1.0]},
    ]

    diagnostics = box_prediction_diagnostics(observed, predicted)

    assert diagnostics["n_corners"] == 2
    assert math.isclose(diagnostics["centroid_rmse"], math.sqrt(0.5))
    assert math.isclose(diagnostics["predicted_rms_radius"], math.sqrt(3.0))
    assert math.isclose(diagnostics["normalized_centroid_rmse"], math.sqrt(1.0 / 6.0))
    assert diagnostics["max_centroid_error"] == 1.0
    assert diagnostics["min_cell_count"] == 10


def test_balanced_joint_indices_equalizes_all_eight_cells_reproducibly():
    cells = []
    for cell in range(8):
        combo = [(cell >> 2) & 1, (cell >> 1) & 1, cell & 1]
        cells.extend([combo] * (cell + 2))
    attrs = torch.tensor(cells)

    first, counts, per_cell = balanced_joint_indices(attrs, seed=7)
    second, _, _ = balanced_joint_indices(attrs, seed=7)
    selected = attrs[first]
    groups = selected[:, 0] * 4 + selected[:, 1] * 2 + selected[:, 2]

    assert counts == list(range(2, 10))
    assert per_cell == 2
    assert torch.equal(first, second)
    assert torch.equal(torch.bincount(groups, minlength=8), torch.full((8,), 2))


def test_balanced_triple_proxy_removes_label_prior_from_cell_weighting():
    centers = []
    attrs = []
    for cell in range(8):
        signs = torch.tensor(
            [2 * ((cell >> 2) & 1) - 1,
             2 * ((cell >> 1) & 1) - 1,
             2 * (cell & 1) - 1],
            dtype=torch.float32,
        )
        count = cell + 1
        centers.extend([signs] * count)
        attrs.extend([((signs + 1) / 2).long()] * count)

    proxy = balanced_triple_proxy(torch.stack(centers), torch.stack(attrs))

    assert proxy["min_cell_count"] == 1
    assert torch.allclose(torch.tensor(proxy["capture_proxy"]), torch.ones(3))
    assert math.isclose(proxy["max_abs_cos"], 0.0, abs_tol=1e-7)
