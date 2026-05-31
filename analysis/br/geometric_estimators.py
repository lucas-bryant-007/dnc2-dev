import torch
import math
from .diagnostics import _as_pm_one

# -----------------------------------------------------------------------------
# CDNV estimators
# -----------------------------------------------------------------------------

def estimate_tilde_V(psi: torch.Tensor, y: torch.Tensor, r: int) -> float:
    """Directional CDNV on the top-r SSL coordinates."""
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
    return float(((var_pos + var_neg) / delta_norm_sq).item())



def estimate_V(psi: torch.Tensor, y: torch.Tensor, r: int) -> float:
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
    return float(((tr_pos + tr_neg) / delta_norm_sq).item())



def predict_tilde_V_from_B(B_r: float, eps: float = 1e-12) -> float:
    if B_r <= eps:
        return float("inf")
    return (1.0 - B_r) / (2.0 * B_r)



def predict_V_from_B(B_r: float, r: int, eps: float = 1e-12) -> float:
    if B_r <= eps:
        return float("inf")
    return (float(r) - B_r) / (2.0 * B_r)