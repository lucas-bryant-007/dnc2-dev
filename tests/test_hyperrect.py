import torch

from analysis.hyperrect import orthonormal_basis, task_axis_basis


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
