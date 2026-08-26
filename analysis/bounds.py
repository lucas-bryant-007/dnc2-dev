"""Exact population bounds from "Which Tasks Survive Self-Supervised Learning?".

All formulas are implemented directly from the paper, in terms of the captured
posterior energy

    B = B(F) = || E[Y F(X)] ||_2^2          (centered, whitened F; balanced Y in {+-1})

which for an SSL population optimum equals B_r = sum_{j<=r} <eta, psi_j>^2.

References (paper equation -> function):
  * Prop 4.1  directional-collapse law          -> directional_cdnv_from_B / cdnv_from_B
       V~_F = (1 - B)/(2B),   V_F = (r - B)/(2B)
       (These already exist in br.geometric_estimators.predict_{tilde_,}V_from_B;
        re-exposed here for a single import surface.)
  * Thm 4.5  few-shot NCC error bound           -> nccc_error_bound[_from_tilde_v]
       err_NCC_m(F) <= (1 - B) + (r - B)/m + (1 - B)/(1 - B + 2 m B)
  * Thm 4.4  multitask hyper-rectangle          -> hyperrectangle_half_side_lengths,
       half-side along task t = sqrt(B_t);          centroid_geometry_rhs,
       centroid bound sum_t (1 - B_t);             near_orthogonality_bound
       |u_i.u_j| <= (|rho_ij| + sqrt(eps_i eps_j)
                     + sqrt((||eta_i||^2 - B_i)(||eta_j||^2 - B_j))) / sqrt(B_i B_j)
"""
import math
from typing import List, Sequence

import torch

try:
    from .cdnv_conventions import (
        CANONICAL_CDNV_NORMALIZATION,
        CDNV_CONVENTION_PROVENANCE,
        ORDERED_SINGLE_CLASS,
        ORIGINAL_HALF_SYMMETRIC,
        UNHALVED_SYMMETRIC,
        cdnv_from_captured_energy,
        directional_cdnv_from_captured_energy,
        ordered_cdnv_from_class_variance,
        symmetric_cdnv_from_class_variances,
    )
    from .br.whitening import canonicalize_binary_labels
except ImportError:  # direct script execution from analysis/
    from cdnv_conventions import (
        CANONICAL_CDNV_NORMALIZATION,
        CDNV_CONVENTION_PROVENANCE,
        ORDERED_SINGLE_CLASS,
        ORIGINAL_HALF_SYMMETRIC,
        UNHALVED_SYMMETRIC,
        cdnv_from_captured_energy,
        directional_cdnv_from_captured_energy,
        ordered_cdnv_from_class_variance,
        symmetric_cdnv_from_class_variances,
    )
    from br.whitening import canonicalize_binary_labels


def _validate_B(B: float, eps: float = 1e-12) -> float:
    B = float(B)
    if not math.isfinite(B) or B < 0.0 or B > 1.0 + eps:
        raise ValueError(f"B must lie in [0, 1], got {B}")
    return min(B, 1.0)


def _validate_integer(value: int, *, minimum: int, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value}")
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            f"{name} must be an integer >= {minimum}, got {value}"
        ) from error
    if integer != value or integer < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value}")
    return integer


# ---------------------------------------------------------------------------
# Prop 4.1: directional-collapse law
# ---------------------------------------------------------------------------
def directional_cdnv_from_B(
    B: float,
    eps: float = 1e-12,
    *,
    normalization: str = CANONICAL_CDNV_NORMALIZATION,
) -> float:
    """Directional CDNV from B, with an explicit symmetric normalization.

    The default is the paper's unhalved convention, ``(1-B)/(2B)``.  Under the
    original ICLR 2022 half-normalization the value is ``(1-B)/(4B)``.
    """
    B = _validate_B(B, eps)
    if B <= eps:
        return directional_cdnv_from_captured_energy(
            0.0,
            normalization=normalization,
        )
    return directional_cdnv_from_captured_energy(
        B,
        normalization=normalization,
    )


def cdnv_from_B(
    B: float,
    r: int,
    eps: float = 1e-12,
    *,
    normalization: str = CANONICAL_CDNV_NORMALIZATION,
) -> float:
    """Full CDNV from B, with an explicit symmetric normalization.

    The default is the paper's unhalved convention, ``(r-B)/(2B)``.  Under the
    original ICLR 2022 half-normalization the value is ``(r-B)/(4B)``.
    """
    B = _validate_B(B, eps)
    r = _validate_integer(r, minimum=1, name="r")
    if B <= eps:
        return cdnv_from_captured_energy(
            0.0,
            r,
            normalization=normalization,
        )
    return cdnv_from_captured_energy(B, r, normalization=normalization)


# ---------------------------------------------------------------------------
# Thm 4.5: direct few-shot NCC bound via captured posterior energy
# ---------------------------------------------------------------------------
def nccc_error_bound(B: float, r: int, m: int, clamp: bool = True) -> float:
    """m-shot NCC error bound err_NCC_m(F) <= (1-B) + (r-B)/m + (1-B)/(1-B+2mB).

    The leading (1 - B) is the uncaptured posterior energy (irreducible floor);
    the rest are finite-sample centroid-estimation terms vanishing as m -> inf.
    At B = 0 the displayed right-hand side is the finite but vacuous value
    ``2 + r/m`` (or 1.0 when clamped to the probability range).
    """
    B = _validate_B(B)
    r = _validate_integer(r, minimum=1, name="r")
    m = _validate_integer(m, minimum=1, name="m")
    val = (1.0 - B) + (r - B) / m + (1.0 - B) / (1.0 - B + 2.0 * m * B)
    return min(val, 1.0) if clamp else val


def nccc_error_bound_from_tilde_v(tilde_v: float, r: int, m: int, clamp: bool = True) -> float:
    """Thm 4.5 in directional-CDNV form (algebraically identical to nccc_error_bound):

        2V~/(1+2V~) + ((r-1)+2rV~)/(m(1+2V~)) + V~/(V~+m).
    """
    tv = float(tilde_v)
    r = _validate_integer(r, minimum=1, name="r")
    m = _validate_integer(m, minimum=1, name="m")
    if math.isnan(tv) or tv < 0:
        raise ValueError(
            f"tilde_v must be nonnegative and r,m positive; got {tv}, {r}, {m}"
        )
    if math.isinf(tv):
        val = 2.0 + float(r) / m
        return min(val, 1.0) if clamp else val
    val = (
        (2.0 * tv) / (1.0 + 2.0 * tv)
        + ((r - 1) + 2.0 * r * tv) / (m * (1.0 + 2.0 * tv))
        + tv / (tv + m)
    )
    return min(val, 1.0) if clamp else val


# ---------------------------------------------------------------------------
# Thm 4.4: multitask hyper-rectangle geometry
# ---------------------------------------------------------------------------
def hyperrectangle_half_side_lengths(Bs: Sequence[float]) -> List[float]:
    """Per-task hyper-rectangle half-side length along task t is sqrt(B_t)."""
    return [math.sqrt(_validate_B(b)) for b in Bs]


def hyperrectangle_side_lengths(Bs: Sequence[float]) -> List[float]:
    """Full edge lengths ``2*sqrt(B_t)`` of the predicted hyper-rectangle."""
    return [2.0 * value for value in hyperrectangle_half_side_lengths(Bs)]


def centroid_geometry_rhs(Bs: Sequence[float]) -> float:
    """Thm 4.4 centroid bound RHS: sum_t (1 - B_t)."""
    return float(sum(1.0 - _validate_B(b) for b in Bs))


def near_orthogonality_bound(
    rho_ij: float, B_i: float, B_j: float, eta_i_sq: float, eta_j_sq: float,
    eps: float = 1e-12,
) -> float:
    """Thm 4.4 upper bound on |u_i^T u_j| between two task decision axes.

    eps_t = 1 - ||eta_t||^2 is the irreducible label uncertainty; ||eta_t||^2 - B_t
    is the unrecovered posterior mass for task t.
    """
    denom = math.sqrt(max(B_i, 0.0) * max(B_j, 0.0))
    if denom <= eps:
        return float("inf")
    eps_i = max(1.0 - eta_i_sq, 0.0)
    eps_j = max(1.0 - eta_j_sq, 0.0)
    resid = max(eta_i_sq - B_i, 0.0) * max(eta_j_sq - B_j, 0.0)
    num = abs(rho_ij) + math.sqrt(eps_i * eps_j) + math.sqrt(resid)
    return num / denom


# ---------------------------------------------------------------------------
# Empirical m-shot NCC error (to plot against the Thm 4.5 bound)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Directional-CDNV few-shot bounds from "Directional Neural Collapse Explains
# Few-Shot Transfer in SSL" (Luthra, Salunkhe, Galanti 2026), plus the 2025
# comparison.  The two papers use incompatible CDNV normalizations.  New-bound
# pairwise V is unhalved symmetric, while the 2025 aggregate is ordered
# single-class (equivalently original-half after averaging both directions).
# Baseline adapters therefore derive their inputs from vi, vj, and d2 instead
# of consuming the ambiguous legacy Vij key.
# ---------------------------------------------------------------------------
BOUND_PROVENANCE = {
    "cdnv_conventions": CDNV_CONVENTION_PROVENANCE,
    "reporting_policy": {
        "raw_rhs": "literal theorem or corollary right-hand side",
        "probability_clipped": "min(raw_rhs, 1) for display only",
        "balanced_binary_chance_level": 0.5,
        "interpretation": (
            "raw_rhs >= 1 is probability-vacuous; raw_rhs >= 0.5 does not "
            "guarantee error below chance on the balanced binary task"
        ),
    },
    "luthra2025_fixed_a16": {
        "formula": "NeurIPS 2025 Proposition 1, displayed a=16 corollary",
        "interpretation": "declared symbols; proof later uses unhalved symmetric V_ij",
        "directional_normalization": ORDERED_SINGLE_CLASS,
        "cdnv_normalization": ORDERED_SINGLE_CLASS,
        "sqrt_cdnv": "mean_over_ordered_pairs_of_sqrt(v_i/d_ij^2)",
        "minimum_m": 10,
    },
    "luthra2025_official_optimized": {
        "source_repository": "https://github.com/DLFundamentals/directional-nc",
        "source_commit": "947f1410e12034a5a6097bf2884040110cc1b8c7",
        "source_file": "bound_analysis/old_bound_core.py",
        "directional_normalization": ORDERED_SINGLE_CLASS,
        "cdnv_normalization": ORIGINAL_HALF_SYMMETRIC,
        "sqrt_cdnv": "sqrt(aggregate_cdnv), matching official code",
        "empirical_variance_denominator": "N, matching eval_utils/geometry.py",
        "minimum_m": 10,
    },
    "luthra2026_pairwise": {
        "Vij_normalization": UNHALVED_SYMMETRIC,
        "Vtilde_ij_normalization": ORDERED_SINGLE_CLASS,
        "theorem_4_1_minimum_m": 10,
        "theorem_C_2_minimum_m": 1,
        "additional_required_guard": "d_ij^2 + (v_j-v_i)/m > 0",
    },
}


def _nonnegative_finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative, got {value}")
    return value


def _validate_m(m: int, *, minimum: int, theorem: str) -> int:
    try:
        return _validate_integer(m, minimum=minimum, name=f"{theorem} m")
    except ValueError as error:
        raise ValueError(
            f"{theorem} requires integer m >= {minimum}, got {m}"
        ) from error


def _E_terms(V: float, Theta: float, m: int):
    """E1, E2, E3 from Prop C.1 (the finite-shot leakage / tail terms)."""
    V = _nonnegative_finite(V, "V")
    Theta = _nonnegative_finite(Theta, "Theta")
    m = _validate_m(m, minimum=1, theorem="finite-shot correction")
    E1 = (4.0 / m) * (V * V + 0.25 * V)
    E2 = V / m
    E3 = (Theta + 2.0 * (m - 1) * V * V) / (m ** 3)
    return E1, E2, E3


def _imbalance_denom(vi: float, vj: float, dij2: float, m: int) -> float:
    """Validated squared variance-imbalance factor.

    The Chebyshev step requires the expected pairwise margin
    ``dij2 + (vj-vi)/m`` to be strictly positive.  Squaring a nonpositive
    margin would otherwise turn an inapplicable theorem into a finite curve.
    """
    vi = _nonnegative_finite(vi, "vi")
    vj = _nonnegative_finite(vj, "vj")
    dij2 = float(dij2)
    if not math.isfinite(dij2) or dij2 <= 0.0:
        raise ValueError(f"dij2 must be finite and positive, got {dij2}")
    m = _validate_m(m, minimum=1, theorem="pairwise NCC bound")
    expected_margin = dij2 + (vj - vi) / m
    if not math.isfinite(expected_margin) or expected_margin <= 0.0:
        raise ValueError(
            "pairwise NCC bound requires positive expected margin "
            f"dij2 + (vj-vi)/m > 0, got {expected_margin}"
        )
    return (expected_margin / dij2) ** 2


def nccc_pair_thm41(Vt, V, Theta, vi, vj, dij2, m) -> float:
    """Pairwise NCC bound, Thm 4.1 form: 4*Vt + (sqrt E1 + sqrt E2 + sqrt E3)^2."""
    m = _validate_m(m, minimum=10, theorem="2026 Theorem 4.1")
    Vt = _nonnegative_finite(Vt, "Vtilde")
    E1, E2, E3 = _E_terms(V, Theta, m)
    num = 4.0 * Vt + (math.sqrt(E1) + math.sqrt(E2) + math.sqrt(E3)) ** 2
    return num / _imbalance_denom(vi, vj, dij2, m)


def nccc_pair_c2(Vt, V, Theta, vi, vj, dij2, m) -> float:
    """Pairwise NCC bound, Thm C.2 form (lambda=1): 4*Vt + 3*(E1+E2+E3).

    By Cauchy-Schwarz (sqrt E1+sqrt E2+sqrt E3)^2 <= 3(E1+E2+E3), so the Thm 4.1
    bound is always <= this one.
    """
    m = _validate_m(m, minimum=1, theorem="2026 Theorem C.2")
    Vt = _nonnegative_finite(Vt, "Vtilde")
    E1, E2, E3 = _E_terms(V, Theta, m)
    num = 4.0 * Vt + 3.0 * (E1 + E2 + E3)
    return num / _imbalance_denom(vi, vj, dij2, m)


def _pairwise_classes(pairwise: dict):
    if not pairwise:
        raise ValueError("pairwise metrics must not be empty")
    cs = set()
    for key in pairwise:
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError(f"pairwise key must be an ordered (i, j) tuple: {key!r}")
        i, j = key
        if i == j:
            raise ValueError(f"pairwise metrics must exclude diagonal key {key!r}")
        cs.add(i); cs.add(j)
    classes = sorted(cs)
    if len(classes) < 2:
        raise ValueError("pairwise metrics must contain at least two classes")
    expected = {(i, j) for i in classes for j in classes if i != j}
    missing = expected.difference(pairwise)
    extra = set(pairwise).difference(expected)
    if missing or extra:
        raise ValueError(
            "multiclass bounds require every ordered class pair; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    return classes


def _pairwise_new_bound_values(pair: dict) -> tuple:
    """Return the 2026 metrics with V recomputed in its declared convention."""
    required = ("Vtilde_ij", "Theta_ij", "vi", "vj", "d2")
    missing = [key for key in required if key not in pair]
    if missing:
        raise ValueError(f"pairwise metrics are missing required keys: {missing}")
    V = symmetric_cdnv_from_class_variances(
        pair["vi"],
        pair["vj"],
        pair["d2"],
        normalization=UNHALVED_SYMMETRIC,
    )
    return (
        _nonnegative_finite(pair["Vtilde_ij"], "Vtilde_ij"),
        V,
        _nonnegative_finite(pair["Theta_ij"], "Theta_ij"),
        pair["vi"],
        pair["vj"],
        pair["d2"],
    )


def directional_nccc_bound(pairwise: dict, m: int, kind: str = "thm41") -> float:
    """Multiclass-averaged directional bound: (1/C') sum_i sum_{j!=i} pair(i,j).

    kind in {"thm41", "c2", "lim"} (lim = m->inf limit, 4*Vtilde, m-independent).
    """
    if kind == "thm41":
        _validate_m(m, minimum=10, theorem="2026 Theorem 4.1")
    elif kind in {"c2", "lim"}:
        _validate_m(m, minimum=1, theorem=(
            "2026 Theorem C.2" if kind == "c2" else "directional limit"
        ))
    else:
        raise ValueError(f"kind must be one of 'thm41', 'c2', or 'lim', got {kind!r}")
    classes = _pairwise_classes(pairwise)
    Cp = len(classes)
    if kind == "lim":
        values = [_pairwise_new_bound_values(pair) for pair in pairwise.values()]
        return sum(4.0 * value[0] for value in values) / Cp
    fn = {"thm41": nccc_pair_thm41, "c2": nccc_pair_c2}[kind]
    return sum(
        fn(*_pairwise_new_bound_values(pair), m)
        for pair in pairwise.values()
    ) / Cp


def _luthra2025_aggregates(pairwise: dict) -> tuple:
    """Aggregate the 2025 ordered metrics without using legacy ``Vij``."""
    classes = _pairwise_classes(pairwise)
    ordered_directional = []
    ordered_cdnv = []
    for pair in pairwise.values():
        required = ("Vtilde_ij", "vi", "d2")
        missing = [key for key in required if key not in pair]
        if missing:
            raise ValueError(f"pairwise metrics are missing required keys: {missing}")
        ordered_directional.append(
            _nonnegative_finite(pair["Vtilde_ij"], "Vtilde_ij")
        )
        ordered_cdnv.append(
            ordered_cdnv_from_class_variance(pair["vi"], pair["d2"])
        )
    n = len(ordered_cdnv)
    return (
        len(classes),
        sum(ordered_directional) / n,
        sum(ordered_cdnv) / n,
        sum(math.sqrt(value) for value in ordered_cdnv) / n,
    )


@torch.no_grad()
def luthra2025_aggregates_from_features(
    features: torch.Tensor,
    labels: torch.Tensor,
) -> tuple:
    """Reproduce the official 2025 metric estimators from feature rows.

    The authors' comparison curve is fed by ``compute_cdnv`` and
    ``compute_directional_cdnv``, both of which use population second moments
    (denominator N), unlike the unbiased pairwise covariance used by the 2026
    bound implementation.
    """
    if features.ndim != 2:
        raise ValueError(f"features must be 2D, got shape {features.shape}")
    labels = labels.reshape(-1)
    if labels.numel() != features.shape[0]:
        raise ValueError("features and labels must have the same number of rows")
    if not bool(torch.isfinite(features).all().item()):
        raise ValueError("features must contain only finite values")
    classes = torch.unique(labels, sorted=True)
    if classes.numel() < 2:
        raise ValueError("2025 metrics require at least two classes")
    means = {}
    trace_variances = {}
    centered = {}
    for class_value in classes:
        key = class_value.item()
        class_features = features[labels == class_value]
        if class_features.shape[0] == 0:
            raise ValueError(f"class {key} has no feature rows")
        mean = class_features.mean(dim=0)
        residual = class_features - mean
        means[key] = mean
        centered[key] = residual
        trace_variances[key] = residual.pow(2).sum(dim=1).mean()

    directional_values = []
    cdnv_values = []
    class_keys = [value.item() for value in classes]
    for i in class_keys:
        for j in class_keys:
            if i == j:
                continue
            delta = means[j] - means[i]
            dij2_tensor = torch.dot(delta, delta)
            dij2 = float(dij2_tensor.item())
            if not math.isfinite(dij2) or dij2 <= 0.0:
                raise ValueError(f"class pair ({i}, {j}) has nonpositive mean gap")
            direction = delta / math.sqrt(dij2)
            directional_variance = (centered[i] @ direction).pow(2).mean()
            directional_values.append(float(directional_variance.item()) / dij2)
            cdnv_values.append(float(trace_variances[i].item()) / dij2)
    n = len(cdnv_values)
    return (
        len(class_keys),
        sum(directional_values) / n,
        sum(cdnv_values) / n,
        sum(math.sqrt(value) for value in cdnv_values) / n,
    )


def luthra2025_fixed_a16_from_aggregates(
    directional_cdnv: float,
    cdnv: float,
    sqrt_cdnv_mean: float,
    m: int,
    n_classes: int,
) -> float:
    """Exact displayed 2025 Proposition 1 corollary with ``a=16``."""
    m = _validate_m(m, minimum=10, theorem="2025 Proposition 1")
    n_classes = _validate_integer(
        n_classes, minimum=2, name="n_classes"
    )
    Vt_f = _nonnegative_finite(directional_cdnv, "directional_cdnv")
    V_f = _nonnegative_finite(cdnv, "cdnv")
    Vs_f = _nonnegative_finite(sqrt_cdnv_mean, "sqrt_cdnv_mean")
    inv_sqrt_m = 1.0 / math.sqrt(m)
    return (n_classes - 1) * (
        8.0 * Vt_f
        + 8.0 * inv_sqrt_m * Vs_f
        + (8.0 * inv_sqrt_m + 4.0 / m) * V_f
    )


def luthra2025_fixed_a16_bound(pairwise: dict, m: int) -> float:
    """Published fixed-``a=16`` comparison with explicit 2025 metrics.

    ``V_f`` is the ordered single-class/original-half aggregate, not the
    unhalved ``Vij`` stored for the 2026 bound.
    """
    Cp, Vt_f, V_f, Vs_f = _luthra2025_aggregates(pairwise)
    return luthra2025_fixed_a16_from_aggregates(Vt_f, V_f, Vs_f, m, Cp)


def _real_cuberoot(value: float) -> float:
    return math.copysign(abs(value) ** (1.0 / 3.0), value)


def luthra2025_official_optimized_details(
    directional_cdnv: float,
    cdnv: float,
    m: int,
    n_classes: int,
) -> dict:
    """Port of the authors' optimized-``a`` implementation.

    This intentionally uses ``sqrt(aggregate CDNV)`` because that is what
    official ``bound_analysis/old_bound_core.py`` computes.  The distinction
    from the published pair-average ``V_f^s`` is retained in provenance.
    """
    m = _validate_m(m, minimum=10, theorem="2025 optimized corollary")
    n_classes = _validate_integer(
        n_classes, minimum=2, name="n_classes"
    )
    alpha = _nonnegative_finite(directional_cdnv, "directional_cdnv")
    beta = _nonnegative_finite(cdnv, "cdnv")
    A = 2.0 + (2.0 ** 1.5) / m
    Bcoef = 0.25 * (
        2.0 * math.sqrt(beta / m)
        + 2.0 * beta / math.sqrt(m)
        + beta / m
    )
    if Bcoef == 0.0:
        value = (n_classes - 1) * 4.0 * alpha
        return {
            "value": value,
            "a_opt": float("inf"),
            "A": A,
            "B": Bcoef,
            "F": None,
        }

    F = (2.0 * alpha * A) / Bcoef
    threshold = (8.0 * F) / 27.0
    if A * A >= threshold:
        radical = math.sqrt(max(A * A - threshold, 0.0))
        y_star = _real_cuberoot(8.0 * F * (A + radical)) + _real_cuberoot(
            8.0 * F * (A - radical)
        )
    else:
        acos_arg = 3.0 * A * math.sqrt(3.0 / (8.0 * F))
        acos_arg = min(max(acos_arg, -1.0), 1.0)
        y_star = 4.0 * math.sqrt((2.0 * F) / 3.0) * math.cos(
            math.acos(acos_arg) / 3.0
        )
    a_opt = max(5.0, 2.0 * A + y_star)
    coefficient = 0.5 - 2.0 / a_opt - (2.0 ** 1.5) / (a_opt * m)
    per_competitor = coefficient ** -2 * alpha + Bcoef * a_opt
    return {
        "value": (n_classes - 1) * per_competitor,
        "a_opt": a_opt,
        "A": A,
        "B": Bcoef,
        "F": F,
    }


def luthra2025_official_optimized_from_aggregates(
    directional_cdnv: float,
    cdnv: float,
    m: int,
    n_classes: int,
) -> float:
    return luthra2025_official_optimized_details(
        directional_cdnv,
        cdnv,
        m,
        n_classes,
    )["value"]


def luthra2025_official_optimized_bound(pairwise: dict, m: int) -> float:
    """Authors' optimized comparison curve, using their metric convention."""
    Cp, Vt_f, V_f, _Vs_f = _luthra2025_aggregates(pairwise)
    return luthra2025_official_optimized_from_aggregates(Vt_f, V_f, m, Cp)


def luthra2025_nccc_bound(pairwise: dict, m: int) -> float:
    """Backward-compatible name for the official optimized 2025 baseline."""
    return luthra2025_official_optimized_bound(pairwise, m)


def _bound_status(function, *args) -> dict:
    """Evaluate a theorem curve without turning an invalid point into a number."""
    try:
        return {"value": float(function(*args)), "valid": True, "reason": None}
    except ValueError as error:
        return {"value": None, "valid": False, "reason": str(error)}


def _probability_bound_reporting(value, chance_level: float = 0.5) -> dict:
    """Keep a theorem's literal RHS separate from display-only clipping.

    A probability bound above one is mathematically valid but vacuous.  For a
    balanced binary task, a bound in ``[chance_level, 1)`` is nontrivial only
    relative to the probability ceiling; it still does not guarantee an error
    below chance.  Invalid theorem points retain ``None`` rather than acquiring
    a clipped numerical value.
    """
    chance_level = float(chance_level)
    if not 0.0 < chance_level < 1.0:
        raise ValueError(
            f"chance_level must lie strictly between zero and one, got {chance_level}"
        )
    if value is None:
        return {
            "raw_rhs": None,
            "probability_clipped": None,
            "below_probability_ceiling": None,
            "informative_vs_chance": None,
            "chance_level": chance_level,
        }
    raw_rhs = _nonnegative_finite(value, "probability-bound RHS")
    return {
        "raw_rhs": raw_rhs,
        "probability_clipped": min(raw_rhs, 1.0),
        "below_probability_ceiling": raw_rhs < 1.0,
        "informative_vs_chance": raw_rhs < chance_level,
        "chance_level": chance_level,
    }


@torch.no_grad()
def directional_fewshot_curves(
    features: torch.Tensor, labels: torch.Tensor, pairwise: dict,
    m_values, n_trials: int = 100, seed: int = 0, max_query: int = 5000,
) -> dict:
    """Figure-3 curves: empirical NCC error + Our/Lim/Luthra bounds, per shot m.

    ``pairwise`` is precomputed (e.g. GeometricEvaluator.compute_pairwise_metrics)
    on the same ``features`` so the bound geometry and empirical error match.
    """
    old_Cp, old_Vt, old_V, old_Vs = luthra2025_aggregates_from_features(
        features,
        labels,
    )
    out = {}
    for m in m_values:
        m = int(m)
        thm41 = _bound_status(directional_nccc_bound, pairwise, m, "thm41")
        c2 = _bound_status(directional_nccc_bound, pairwise, m, "c2")
        limit = _bound_status(directional_nccc_bound, pairwise, m, "lim")
        old_a16 = _bound_status(
            luthra2025_fixed_a16_from_aggregates,
            old_Vt,
            old_V,
            old_Vs,
            m,
            old_Cp,
        )
        old_optimized = _bound_status(
            luthra2025_official_optimized_from_aggregates,
            old_Vt,
            old_V,
            m,
            old_Cp,
        )
        bound_reporting = {
            "our_thm41": _probability_bound_reporting(thm41["value"]),
            "our_c2": _probability_bound_reporting(c2["value"]),
            "lim": _probability_bound_reporting(limit["value"]),
            "luthra2025_optimized_official": _probability_bound_reporting(
                old_optimized["value"]
            ),
            "luthra2025_a16_published": _probability_bound_reporting(
                old_a16["value"]
            ),
        }
        out[m] = {
            "empirical": empirical_nccc_error(
                features,
                labels,
                m,
                n_trials,
                seed,
                max_query,
                assume_single_view=True,
            ),
            "our_thm41": thm41["value"],
            "our_c2": c2["value"],
            "lim": limit["value"],
            # The legacy field now means the authors' optimized comparison.
            "luthra2025": old_optimized["value"],
            "luthra2025_optimized_official": old_optimized["value"],
            "luthra2025_a16_published": old_a16["value"],
            "validity": {
                "our_thm41": thm41,
                "our_c2": c2,
                "lim": limit,
                "luthra2025_optimized_official": old_optimized,
                "luthra2025_a16_published": old_a16,
            },
            "bound_reporting": bound_reporting,
            "bound_provenance": BOUND_PROVENANCE,
        }
    return out


@torch.no_grad()
def combined_fewshot_curves(
    psi: torch.Tensor, labels: torch.Tensor, B_by_r: dict, r_values, m_values,
    pairwise_by_r: dict, instance_ids: torch.Tensor = None,
    n_trials: int = 100, seed: int = 0, max_query: int = 5000,
) -> dict:
    """New-vs-old few-shot comparison on the whitened representation psi.

    For each rank r, on psi[:, :r]: empirical NCC error, the NEW bound
    (Thm 4.5 via B = B_by_r[r]), and the OLD bounds (Thm 4.1 directional + Luthra
    2025 + the 4*Vtilde limit) from ``pairwise_by_r[r]`` (the directional metrics
    of psi[:, :r]). ``thm45_B`` is retained as the probability-clipped value;
    ``thm45_B_raw`` and ``bound_reporting`` retain the literal RHS and its
    interpretation. Returns {r: {"B": B, "curves": {m: {...}}}}.
    """
    if instance_ids is None:
        raise ValueError(
            "Theorem-facing combined few-shot evaluation requires instance_ids"
        )
    instance_layout = _build_instance_sampling_layout(labels, instance_ids)
    _require_balanced_instance_layout(
        instance_layout,
        "Theorem-facing combined few-shot evaluation",
    )
    sampling = {
        **instance_layout["diagnostics"],
        "seed": int(seed),
        "n_trials": int(n_trials),
        "max_query_instances_per_class": (
            None if max_query is None else int(max_query)
        ),
    }
    out = {}
    for r in r_values:
        r = int(r)
        psir = psi[:, :r]
        B = float(B_by_r[r])
        pw = pairwise_by_r[r]
        old_Cp, old_Vt, old_V, old_Vs = luthra2025_aggregates_from_features(
            psir,
            labels,
        )
        curves = {}
        for m in m_values:
            m = int(m)
            class_counts = sampling["class_instance_counts"]
            query_negative = class_counts["-1"] - m
            query_positive = class_counts["+1"] - m
            if max_query is not None:
                query_negative = min(query_negative, max_query)
                query_positive = min(query_positive, max_query)
            thm41 = _bound_status(directional_nccc_bound, pw, m, "thm41")
            limit = _bound_status(directional_nccc_bound, pw, m, "lim")
            old_a16 = _bound_status(
                luthra2025_fixed_a16_from_aggregates,
                old_Vt,
                old_V,
                old_Vs,
                m,
                old_Cp,
            )
            old_optimized = _bound_status(
                luthra2025_official_optimized_from_aggregates,
                old_Vt,
                old_V,
                m,
                old_Cp,
            )
            thm45_raw = nccc_error_bound(B, r, m, clamp=False)
            bound_reporting = {
                "thm45_B": _probability_bound_reporting(thm45_raw),
                "thm41_dir": _probability_bound_reporting(thm41["value"]),
                "luthra2025_optimized_official": _probability_bound_reporting(
                    old_optimized["value"]
                ),
                "luthra2025_a16_published": _probability_bound_reporting(
                    old_a16["value"]
                ),
                "lim": _probability_bound_reporting(limit["value"]),
            }
            curves[m] = {
                "empirical": empirical_nccc_error(
                    psir,
                    labels,
                    m,
                    n_trials,
                    seed,
                    max_query,
                    instance_ids=instance_ids,
                    _instance_layout=instance_layout,
                ),
                # Backward-compatible field: probability-clipped display value.
                "thm45_B": bound_reporting["thm45_B"][
                    "probability_clipped"
                ],
                # Literal RHS of draft Theorem 4.5, retained for audit/tables.
                "thm45_B_raw": thm45_raw,
                "thm41_dir": thm41["value"],
                "luthra2025": old_optimized["value"],
                "luthra2025_optimized_official": old_optimized["value"],
                "luthra2025_a16_published": old_a16["value"],
                "lim": limit["value"],
                "validity": {
                    "thm41_dir": thm41,
                    "lim": limit,
                    "luthra2025_optimized_official": old_optimized,
                    "luthra2025_a16_published": old_a16,
                },
                "bound_reporting": bound_reporting,
                "bound_provenance": BOUND_PROVENANCE,
                "empirical_group_counts": {
                    "support_instances_per_class": m,
                    "query_instances": {
                        "-1": query_negative,
                        "+1": query_positive,
                    },
                },
            }
        out[r] = {
            "B": B,
            "curves": curves,
            "empirical_sampling": sampling,
        }
    return out


def _build_instance_sampling_layout(
    labels: torch.Tensor,
    instance_ids: torch.Tensor,
) -> dict:
    """Validate instance groups and build an efficient row-sampling layout."""
    labels = labels.reshape(-1)
    instance_ids = instance_ids.reshape(-1)
    if labels.numel() != instance_ids.numel():
        raise ValueError("labels and instance_ids must have the same number of rows")
    if labels.numel() < 2:
        raise ValueError("instance-aware sampling requires at least two rows")
    if instance_ids.is_floating_point() and not bool(
        torch.isfinite(instance_ids).all().item()
    ):
        raise ValueError("instance_ids must contain only finite values")

    canonical_labels, encoding = canonicalize_binary_labels(
        labels,
        "few-shot labels",
    )
    labels_cpu = canonical_labels.detach().cpu().long()
    ids_cpu = instance_ids.detach().cpu()
    unique_ids, inverse = torch.unique(
        ids_cpu,
        sorted=True,
        return_inverse=True,
    )
    order = torch.argsort(inverse, stable=True)
    sorted_groups = inverse[order]
    sorted_labels = labels_cpu[order]
    same_group = sorted_groups[1:] == sorted_groups[:-1]
    inconsistent = same_group & (sorted_labels[1:] != sorted_labels[:-1])
    if bool(inconsistent.any().item()):
        bad_position = int(torch.where(inconsistent)[0][0].item())
        bad_group = int(sorted_groups[bad_position].item())
        raise ValueError(
            "Every latent instance must have exactly one class label; "
            f"instance_id={unique_ids[bad_group].item()} has conflicting rows"
        )

    counts = torch.bincount(inverse, minlength=unique_ids.numel())
    offsets = torch.zeros(unique_ids.numel(), dtype=torch.long)
    if unique_ids.numel() > 1:
        offsets[1:] = torch.cumsum(counts, dim=0)[:-1]
    group_labels = sorted_labels[offsets]
    positive_groups = torch.where(group_labels == 1)[0]
    negative_groups = torch.where(group_labels == -1)[0]
    diagnostics = {
        "sampling_unit": "latent_instance",
        "view_selection": "one_uniform_random_view_per_selected_instance",
        "support_query_instance_disjointness_enforced": True,
        "n_rows": int(labels.numel()),
        "n_instances": int(unique_ids.numel()),
        "class_instance_counts": {
            "-1": int(negative_groups.numel()),
            "+1": int(positive_groups.numel()),
        },
        "min_rows_per_instance": int(counts.min().item()),
        "max_rows_per_instance": int(counts.max().item()),
        "label_mapping": encoding["canonical_mapping"],
    }
    return {
        "unique_ids": unique_ids,
        "order": order,
        "counts": counts,
        "offsets": offsets,
        "positive_groups": positive_groups,
        "negative_groups": negative_groups,
        "diagnostics": diagnostics,
    }


def instance_group_diagnostics(
    labels: torch.Tensor,
    instance_ids: torch.Tensor,
) -> dict:
    """Return JSON-safe validation and count metadata for NCC instance groups."""
    return _build_instance_sampling_layout(labels, instance_ids)["diagnostics"]


def _require_balanced_instance_layout(layout: dict, context: str) -> None:
    counts = layout["diagnostics"]["class_instance_counts"]
    if counts["-1"] != counts["+1"]:
        raise ValueError(
            f"{context} requires equal class counts at the latent-instance "
            f"level; class_instance_counts={counts}"
        )


def _sample_one_row_per_instance(
    layout: dict,
    groups: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    counts = layout["counts"][groups]
    random_offsets = torch.floor(
        torch.rand(groups.numel(), generator=generator) * counts
    ).long()
    sorted_positions = layout["offsets"][groups] + random_offsets
    return layout["order"][sorted_positions]


def _sample_grouped_binary_trial(
    layout: dict,
    m: int,
    generator: torch.Generator,
    max_query: int = None,
) -> dict:
    """Draw disjoint support/query instances and one view row from each."""
    positive = layout["positive_groups"]
    negative = layout["negative_groups"]
    if positive.numel() <= m or negative.numel() <= m:
        raise ValueError(
            f"need > m={m} instances per class "
            f"(have {positive.numel()}+/{negative.numel()}-)"
        )
    if max_query is not None and max_query < 1:
        raise ValueError(f"max_query must be positive when supplied, got {max_query}")
    positive = positive[torch.randperm(positive.numel(), generator=generator)]
    negative = negative[torch.randperm(negative.numel(), generator=generator)]
    support_positive = positive[:m]
    support_negative = negative[:m]
    query_positive = positive[m:]
    query_negative = negative[m:]
    if max_query is not None:
        query_positive = query_positive[:max_query]
        query_negative = query_negative[:max_query]
    support_groups = torch.cat((support_positive, support_negative))
    query_groups = torch.cat((query_positive, query_negative))
    if bool(torch.isin(support_groups, query_groups).any().item()):
        raise RuntimeError("support and query latent-instance IDs overlap")

    return {
        "support_positive_rows": _sample_one_row_per_instance(
            layout,
            support_positive,
            generator,
        ),
        "support_negative_rows": _sample_one_row_per_instance(
            layout,
            support_negative,
            generator,
        ),
        "query_positive_rows": _sample_one_row_per_instance(
            layout,
            query_positive,
            generator,
        ),
        "query_negative_rows": _sample_one_row_per_instance(
            layout,
            query_negative,
            generator,
        ),
        "support_instance_ids": layout["unique_ids"][support_groups],
        "query_instance_ids": layout["unique_ids"][query_groups],
    }


@torch.no_grad()
def empirical_nccc_error(
    features: torch.Tensor, labels: torch.Tensor, m: int,
    n_trials: int = 200, seed: int = 0, max_query: int = None,
    instance_ids: torch.Tensor = None,
    assume_single_view: bool = False,
    _instance_layout: dict = None,
) -> float:
    """Balanced, latent-instance-aware m-shot nearest-centroid error.

    Each trial samples m support *instances* per class and disjoint query
    instances. Exactly one uniformly random view row is used for every selected
    instance, matching the paper's single-view law. The per-class error rates
    are averaged. Supply ``instance_ids`` for grouped data. For a genuinely
    single-view dataset, explicitly pass ``assume_single_view=True``. Supplying
    both modes or neither is rejected.
    """
    if features.ndim != 2 or features.shape[0] != labels.reshape(-1).numel():
        raise ValueError("features and labels must have compatible [N,*] shapes")
    if m < 1 or n_trials < 1:
        raise ValueError("m and n_trials must be positive")
    if _instance_layout is None:
        if instance_ids is None and not assume_single_view:
            raise ValueError(
                "Specify instance_ids or explicitly set assume_single_view=True"
            )
        if instance_ids is not None and assume_single_view:
            raise ValueError(
                "Specify exactly one of instance_ids or assume_single_view=True"
            )
        if assume_single_view:
            instance_ids = torch.arange(labels.reshape(-1).numel())
        layout = _build_instance_sampling_layout(labels, instance_ids)
    else:
        if assume_single_view:
            raise ValueError("Grouped sampling layout cannot assume single-view data")
        layout = _instance_layout
    gen = torch.Generator(device="cpu").manual_seed(seed)
    errs = []
    for _ in range(n_trials):
        trial = _sample_grouped_binary_trial(layout, m, gen, max_query)
        support_positive = trial["support_positive_rows"].to(features.device)
        support_negative = trial["support_negative_rows"].to(features.device)
        q_pos = trial["query_positive_rows"].to(features.device)
        q_neg = trial["query_negative_rows"].to(features.device)
        c_pos = features[support_positive].mean(dim=0)
        c_neg = features[support_negative].mean(dim=0)
        fp, fn = features[q_pos], features[q_neg]
        # nearest-centroid: predict positive iff strictly closer to c_pos
        err_pos = (((fp - c_pos) ** 2).sum(1) >= ((fp - c_neg) ** 2).sum(1)).float().mean()
        err_neg = (((fn - c_neg) ** 2).sum(1) > ((fn - c_pos) ** 2).sum(1)).float().mean()
        errs.append(0.5 * float(err_pos) + 0.5 * float(err_neg))
    return sum(errs) / len(errs)


@torch.no_grad()
def fewshot_curves(
    psi: torch.Tensor, labels: torch.Tensor, B_by_r: dict,
    r_values, m_values, instance_ids: torch.Tensor = None,
    n_trials: int = 100, seed: int = 0, max_query: int = 5000,
) -> dict:
    """Empirical m-shot NCC error and the Thm 4.5 bound, per rank r and shot m.

    ``psi`` is the whitened representation [N, k]; for each r the support/query
    NCC runs on ``psi[:, :r]`` and the bound uses B = ``B_by_r[r]`` and that r.
    ``bound`` is clipped to the probability range for compatibility, while
    ``bound_raw`` retains the literal theorem RHS.
    """
    if instance_ids is None:
        raise ValueError("Theorem-facing few-shot evaluation requires instance_ids")
    instance_layout = _build_instance_sampling_layout(labels, instance_ids)
    _require_balanced_instance_layout(
        instance_layout,
        "Theorem-facing few-shot evaluation",
    )
    sampling = {
        **instance_layout["diagnostics"],
        "seed": int(seed),
        "n_trials": int(n_trials),
        "max_query_instances_per_class": (
            None if max_query is None else int(max_query)
        ),
    }
    out = {}
    for r in r_values:
        r = int(r)
        psir = psi[:, :r]
        B = float(B_by_r[r])
        emp = {int(m): empirical_nccc_error(
                   psir, labels, int(m), n_trials, seed, max_query,
                   instance_ids=instance_ids,
                   _instance_layout=instance_layout)
               for m in m_values}
        bnd_raw = {
            int(m): nccc_error_bound(B, r, int(m), clamp=False)
            for m in m_values
        }
        bnd = {m: min(value, 1.0) for m, value in bnd_raw.items()}
        group_counts = {}
        for m in m_values:
            m = int(m)
            negative = sampling["class_instance_counts"]["-1"] - m
            positive = sampling["class_instance_counts"]["+1"] - m
            if max_query is not None:
                negative = min(negative, max_query)
                positive = min(positive, max_query)
            group_counts[m] = {
                "support_instances_per_class": m,
                "query_instances": {"-1": negative, "+1": positive},
            }
        out[r] = {
            "B": B,
            "empirical": emp,
            "bound": bnd,
            "bound_raw": bnd_raw,
            "empirical_sampling": sampling,
            "empirical_group_counts": group_counts,
        }
    return out
