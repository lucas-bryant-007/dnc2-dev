import math

import pytest

from analysis.bounds import (
    cdnv_from_B,
    directional_cdnv_from_B,
    hyperrectangle_half_side_lengths,
    nccc_error_bound,
    nccc_error_bound_from_tilde_v,
)


def test_equivalent_fewshot_parameterizations_match():
    captured, rank, shots = 0.63, 7, 11
    directional = directional_cdnv_from_B(captured)
    assert nccc_error_bound(captured, rank, shots, clamp=False) == pytest.approx(
        nccc_error_bound_from_tilde_v(directional, rank, shots, clamp=False)
    )


def test_bounds_validate_theoretical_domains():
    with pytest.raises(ValueError):
        directional_cdnv_from_B(1.2)
    with pytest.raises(ValueError):
        cdnv_from_B(0.5, 0)
    with pytest.raises(ValueError):
        nccc_error_bound(0.5, 2, 0)
    with pytest.raises(ValueError):
        hyperrectangle_half_side_lengths([-0.1])


def test_hyperrectangle_values_are_half_sides():
    assert hyperrectangle_half_side_lengths([0.25, 1.0]) == [0.5, 1.0]
    assert math.isinf(directional_cdnv_from_B(0.0))
