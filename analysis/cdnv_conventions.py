"""Canonical CDNV normalizations and explicit conversion helpers.

Three related quantities occur in the papers and reference implementations:

``ordered_single_class``
    ``v_i / d_ij^2`` for an ordered pair ``(i, j)``.  The 2025 directional-CDNV
    paper defines its aggregate CDNV by averaging this quantity over ``i != j``.

``original_half_symmetric``
    ``(v_i + v_j) / (2 d_ij^2)``, the ICLR 2022 CDNV definition.  Averaging the
    ordered quantity over both directions gives this normalization.

``unhalved_symmetric``
    ``(v_i + v_j) / d_ij^2``.  This is the convention used by the current
    captured-energy identities and by the 2026 pairwise-metric schema.

The repository's theorem-facing B geometry is canonically *unhalved symmetric*.
Every interface to a paper or baseline using another convention must convert
explicitly through this module.
"""

import math


ORDERED_SINGLE_CLASS = "ordered_single_class"
ORIGINAL_HALF_SYMMETRIC = "original_half_symmetric"
UNHALVED_SYMMETRIC = "unhalved_symmetric"

CANONICAL_CDNV_NORMALIZATION = UNHALVED_SYMMETRIC

SYMMETRIC_NORMALIZATIONS = {
    ORIGINAL_HALF_SYMMETRIC,
    UNHALVED_SYMMETRIC,
}


def _nonnegative_finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative, got {value}")
    return value


def convert_symmetric_cdnv(
    value: float,
    *,
    source: str,
    target: str,
) -> float:
    """Convert a symmetric CDNV value between original and unhalved forms."""
    value = _nonnegative_finite(value, "CDNV")
    if source not in SYMMETRIC_NORMALIZATIONS:
        raise ValueError(f"unsupported source CDNV normalization: {source!r}")
    if target not in SYMMETRIC_NORMALIZATIONS:
        raise ValueError(f"unsupported target CDNV normalization: {target!r}")
    if source == target:
        return value
    if source == ORIGINAL_HALF_SYMMETRIC:
        return 2.0 * value
    return 0.5 * value


def symmetric_cdnv_from_class_variances(
    vi: float,
    vj: float,
    dij2: float,
    *,
    normalization: str = CANONICAL_CDNV_NORMALIZATION,
) -> float:
    """Compute symmetric CDNV from class trace variances and squared gap."""
    vi = _nonnegative_finite(vi, "vi")
    vj = _nonnegative_finite(vj, "vj")
    dij2 = float(dij2)
    if not math.isfinite(dij2) or dij2 <= 0.0:
        raise ValueError(f"dij2 must be finite and positive, got {dij2}")
    unhalved = (vi + vj) / dij2
    return convert_symmetric_cdnv(
        unhalved,
        source=UNHALVED_SYMMETRIC,
        target=normalization,
    )


def ordered_cdnv_from_class_variance(vi: float, dij2: float) -> float:
    """Return the ordered 2025 quantity ``v_i / d_ij^2``."""
    vi = _nonnegative_finite(vi, "vi")
    dij2 = float(dij2)
    if not math.isfinite(dij2) or dij2 <= 0.0:
        raise ValueError(f"dij2 must be finite and positive, got {dij2}")
    return vi / dij2


def cdnv_from_captured_energy(
    B: float,
    r: int,
    *,
    normalization: str = CANONICAL_CDNV_NORMALIZATION,
) -> float:
    """Captured-energy identity in an explicitly selected normalization."""
    if normalization not in SYMMETRIC_NORMALIZATIONS:
        raise ValueError(f"unsupported CDNV normalization: {normalization!r}")
    B = float(B)
    if not math.isfinite(B) or B < 0.0 or B > 1.0:
        raise ValueError(f"B must lie in [0, 1], got {B}")
    if isinstance(r, bool):
        raise ValueError(f"r must be a positive integer, got {r}")
    try:
        integer_r = int(r)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"r must be a positive integer, got {r}") from error
    if integer_r != r or integer_r < 1:
        raise ValueError(f"r must be a positive integer, got {r}")
    if B == 0.0:
        return float("inf")
    unhalved = (float(integer_r) - B) / (2.0 * B)
    return convert_symmetric_cdnv(
        unhalved,
        source=UNHALVED_SYMMETRIC,
        target=normalization,
    )


def directional_cdnv_from_captured_energy(
    B: float,
    *,
    normalization: str = CANONICAL_CDNV_NORMALIZATION,
) -> float:
    """Directional captured-energy identity in a symmetric normalization."""
    if normalization not in SYMMETRIC_NORMALIZATIONS:
        raise ValueError(f"unsupported CDNV normalization: {normalization!r}")
    B = float(B)
    if not math.isfinite(B) or B < 0.0 or B > 1.0:
        raise ValueError(f"B must lie in [0, 1], got {B}")
    if B == 0.0:
        return float("inf")
    unhalved = (1.0 - B) / (2.0 * B)
    return convert_symmetric_cdnv(
        unhalved,
        source=UNHALVED_SYMMETRIC,
        target=normalization,
    )


CDNV_CONVENTION_PROVENANCE = {
    "canonical_internal": CANONICAL_CDNV_NORMALIZATION,
    "original_2022": ORIGINAL_HALF_SYMMETRIC,
    "luthra2025_aggregate": ORDERED_SINGLE_CLASS,
    "pairwise_Vij_legacy_key": UNHALVED_SYMMETRIC,
    "pairwise_Vtilde_ij_legacy_key": ORDERED_SINGLE_CLASS,
    "conversion": {
        "unhalved_to_original": "multiply_by_1/2",
        "original_to_unhalved": "multiply_by_2",
    },
}
