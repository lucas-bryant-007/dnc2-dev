import math

import torch

from analysis.hyperrect import (
    apply_rewhitener,
    balanced_joint_indices,
    balanced_triple_proxy,
    box_prediction_diagnostics,
    crossfit_probe_geometry,
    fit_box_reference,
    fit_rewhitener,
    orthonormal_basis,
    subclass_box,
    task_axis_basis,
)


def test_train_fitted_rewhitener_applies_without_recentering_test_data():
    generator = torch.Generator().manual_seed(19)
    train = torch.randn(2000, 4, generator=generator) * torch.tensor(
        [0.5, 1.0, 2.0, 4.0]
    )
    test = (
        torch.randn(500, 4, generator=generator)
        * torch.tensor([1.0, 2.0, 4.0, 8.0])
        + 3.0
    )

    transform = fit_rewhitener(train)
    transformed_train = apply_rewhitener(train, transform)
    transformed_test = apply_rewhitener(test, transform)

    assert torch.linalg.vector_norm(transformed_train.mean(dim=0)) < 1e-4
    assert torch.linalg.vector_norm(transformed_test.mean(dim=0)) > 1.0
    assert transform.n_fit == train.shape[0]
    assert transform.feature_dim == train.shape[1]
    assert transform.metadata()["kind"] == "zca"


def test_apply_rewhitener_rejects_wrong_feature_dimension():
    transform = fit_rewhitener(torch.randn(20, 3))
    try:
        apply_rewhitener(torch.randn(10, 4), transform)
    except ValueError as error:
        assert "does not match" in str(error)
    else:
        raise AssertionError("mismatched feature dimension should fail")


def test_crossfit_probe_geometry_avoids_same_sample_dimension_noise_floor():
    generator = torch.Generator().manual_seed(31)
    n = 4000
    dimension = 256
    attrs_a = torch.randint(0, 2, (n, 3), generator=generator)
    attrs_b = torch.randint(0, 2, (n, 3), generator=generator)
    noise_a = torch.randn(n, dimension, generator=generator)
    noise_b = torch.randn(n, dimension, generator=generator)

    geometry = crossfit_probe_geometry(
        noise_a,
        attrs_a,
        noise_b,
        attrs_b,
        ["a", "b", "c"],
    )
    same_sample_capture = sum(
        (
            (
                (2 * attrs_a[:, index].float() - 1).unsqueeze(1)
                * noise_a
            ).mean(dim=0).square().sum().item()
        )
        for index in range(3)
    ) / 3
    crossfit_capture = sum(geometry["capture_B"].values()) / 3

    assert same_sample_capture > 0.04
    assert abs(crossfit_capture) < 0.02


def test_box_reference_freezes_train_axes_and_corners():
    attrs = torch.tensor(
        [[(cell >> bit) & 1 for bit in (2, 1, 0)] for cell in range(8)]
    ).repeat_interleave(5, dim=0)
    signs = 2.0 * attrs.float() - 1.0
    features = torch.cat([signs, torch.zeros(signs.shape[0], 2)], dim=1)

    reference = fit_box_reference(features, attrs, ["a", "b", "c"])

    assert reference.triple_names == ["a", "b", "c"]
    assert reference.n_fit == 40
    assert torch.allclose(reference.basis[:3], torch.eye(3))
    assert len(reference.predicted_box) == 8


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
