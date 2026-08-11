import torch
import math
from .diagnostics import _as_pm_one
try:
    from ..cdnv_conventions import (
        CANONICAL_CDNV_NORMALIZATION,
        UNHALVED_SYMMETRIC,
        cdnv_from_captured_energy,
        convert_symmetric_cdnv,
        directional_cdnv_from_captured_energy,
    )
except ImportError:  # imported as top-level br package by legacy scripts
    from cdnv_conventions import (
        CANONICAL_CDNV_NORMALIZATION,
        UNHALVED_SYMMETRIC,
        cdnv_from_captured_energy,
        convert_symmetric_cdnv,
        directional_cdnv_from_captured_energy,
    )

# -----------------------------------------------------------------------------
# CDNV estimators
# -----------------------------------------------------------------------------

def estimate_tilde_V(
    psi: torch.Tensor,
    y: torch.Tensor,
    r: int,
    *,
    normalization: str = CANONICAL_CDNV_NORMALIZATION,
) -> float:
    """Directional CDNV on the top-r coordinates; unhalved by default."""
    y = _as_pm_one(y)
    f = psi[:, :r]

    pos = y == 1
    neg = y == -1
    if pos.sum() == 0 or neg.sum() == 0:
        raise ValueError("Both classes must be present to estimate tilde_V")

    mu_pos = f[pos].mean(dim=0)
    mu_neg = f[neg].mean(dim=0)
    delta = mu_pos - mu_neg
    delta_norm_sq = float(torch.dot(delta, delta).item())
    if delta_norm_sq <= 1e-12:
        return float("inf")

    u = delta / math.sqrt(delta_norm_sq)

    def directional_variance(x: torch.Tensor) -> torch.Tensor:
        xc = x - x.mean(dim=0, keepdim=True)
        proj = xc @ u
        return (proj ** 2).mean()

    var_pos = directional_variance(f[pos])
    var_neg = directional_variance(f[neg])
    unhalved = float(((var_pos + var_neg) / delta_norm_sq).item())
    return convert_symmetric_cdnv(
        unhalved,
        source=UNHALVED_SYMMETRIC,
        target=normalization,
    )



def estimate_V(
    psi: torch.Tensor,
    y: torch.Tensor,
    r: int,
    *,
    normalization: str = CANONICAL_CDNV_NORMALIZATION,
) -> float:
    """
    Full CDNV on the top-r SSL coordinates:
        V = (tr(Sigma_+) + tr(Sigma_-)) / ||Delta||^2.
    """
    y = _as_pm_one(y)
    f = psi[:, :r]

    pos = y == 1
    neg = y == -1
    if pos.sum() == 0 or neg.sum() == 0:
        raise ValueError("Both classes must be present to estimate V")

    mu_pos = f[pos].mean(dim=0)
    mu_neg = f[neg].mean(dim=0)
    delta = mu_pos - mu_neg
    delta_norm_sq = float(torch.dot(delta, delta).item())
    if delta_norm_sq <= 1e-12:
        return float("inf")

    def trace_cov(x: torch.Tensor) -> torch.Tensor:
        xc = x - x.mean(dim=0, keepdim=True)
        return (xc.pow(2).sum(dim=1)).mean()

    tr_pos = trace_cov(f[pos])
    tr_neg = trace_cov(f[neg])
    unhalved = float(((tr_pos + tr_neg) / delta_norm_sq).item())
    return convert_symmetric_cdnv(
        unhalved,
        source=UNHALVED_SYMMETRIC,
        target=normalization,
    )



def predict_tilde_V_from_B(
    B_r: float,
    eps: float = 1e-12,
    *,
    normalization: str = CANONICAL_CDNV_NORMALIZATION,
) -> float:
    if B_r <= eps:
        return float("inf")
    return directional_cdnv_from_captured_energy(
        B_r,
        normalization=normalization,
    )



def predict_V_from_B(
    B_r: float,
    r: int,
    eps: float = 1e-12,
    *,
    normalization: str = CANONICAL_CDNV_NORMALIZATION,
) -> float:
    if B_r <= eps:
        return float("inf")
    return cdnv_from_captured_energy(B_r, r, normalization=normalization)
