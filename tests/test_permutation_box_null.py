import numpy as np
import pytest

from analysis.permutation_box_null import (
    decode_cell_labels,
    normalized_corner_rmse,
    run_permutation_null,
)


def _factorial_box(samples_per_cell=80, seed=4):
    generator = np.random.default_rng(seed)
    cell_ids = np.repeat(np.arange(8), samples_per_cell)
    labels = decode_cell_labels(cell_ids)
    signs = 2.0 * labels - 1.0
    coords = signs + 0.10 * generator.normal(size=signs.shape)
    centers = np.asarray(
        [
            [2 * ((cell >> 2) & 1) - 1, 2 * ((cell >> 1) & 1) - 1, 2 * (cell & 1) - 1]
            for cell in range(8)
        ],
        dtype=np.float64,
    )
    return coords, labels, centers


def test_decode_cell_labels_uses_hyperrect_bit_order():
    decoded = decode_cell_labels(np.arange(8))

    assert decoded.tolist() == [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ]


def test_permutation_null_separates_real_box_from_shuffled_labels():
    coords, labels, centers = _factorial_box()

    observed, null, counts = run_permutation_null(
        coords,
        labels,
        centers,
        n_permutations=199,
        seed=17,
    )

    assert counts.tolist() == [80] * 8
    assert observed < 0.02
    assert np.min(null) > 0.85
    assert (1 + np.count_nonzero(null <= observed)) / 200 == pytest.approx(0.005)


def test_corner_rmse_rejects_missing_cells():
    coords, labels, centers = _factorial_box(samples_per_cell=2)
    keep = np.any(labels != np.asarray([1, 1, 1]), axis=1)

    with pytest.raises(ValueError, match="all eight"):
        normalized_corner_rmse(coords[keep], labels[keep], centers)
