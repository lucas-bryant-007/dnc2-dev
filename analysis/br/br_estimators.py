import torch
import math
from typing import Dict, Iterable, Optional
from .diagnostics import _as_pm_one

# -----------------------------------------------------------------------------
# B_r estimators
# -----------------------------------------------------------------------------

def estimate_B_r(
    psi: torch.Tensor,
    y: torch.Tensor,
    r_values: Iterable[int],
    center_labels: bool = True,
    gram_ridge: float = 1e-6,
) -> Dict[int, float]:
    """
    Empirically orthonormalized estimator:
        beta_r = E_n[Psi_r^T Y],  G_r = E_n[Psi_r^T Psi_r]
        B_r^orth = beta_r^T G_r^{-1} beta_r.

    This is the recommended default for experiments testing the theorem, because
    it stays faithful to the population quantity while correcting for empirical
    non-orthonormality on the labeled sample.
    """
    y = _as_pm_one(y)
    if center_labels:
        y = y - y.mean()

    if psi.ndim != 2:
        raise ValueError(f"psi must be 2D, got shape {psi.shape}")
    if psi.shape[0] != y.shape[0]:
        raise ValueError("psi and y must have the same number of samples")

    n, k = psi.shape
    r_values = sorted(set(int(r) for r in r_values))
    if len(r_values) == 0:
        return {}
    if min(r_values) < 1 or max(r_values) > k:
        raise ValueError(f"All r must satisfy 1 <= r <= {k}; got {r_values}")

    out: Dict[int, float] = {}
    for r in r_values:
        psi_r = psi[:, :r]
        beta = (psi_r.T @ y) / n
        g = (psi_r.T @ psi_r) / n
        g = g + gram_ridge * torch.eye(r, device=g.device, dtype=g.dtype)
        out[r] = float((beta @ torch.linalg.solve(g, beta)).item())
    return out


def estimate_B_r_raw(
    psi: torch.Tensor,
    y: torch.Tensor,
    r_values: Iterable[int],
    center_labels: bool = True,
) -> Dict[int, float]:
    """
    Raw plug-in estimator:
        B_r = sum_{j<=r} (E_n[Y psi_j])^2.

    This is close to the population formula but can be unstable when the top-r
    empirical coordinates are not orthonormal on the labeled sample.
    """
    y = _as_pm_one(y)
    if center_labels:
        y = y - y.mean()

    if psi.ndim != 2:
        raise ValueError(f"psi must be 2D, got shape {psi.shape}")
    if psi.shape[0] != y.shape[0]:
        raise ValueError("psi and y must have the same number of samples")

    n, k = psi.shape
    r_values = sorted(set(int(r) for r in r_values))
    if len(r_values) == 0:
        return {}
    if min(r_values) < 1 or max(r_values) > k:
        raise ValueError(f"All r must satisfy 1 <= r <= {k}; got {r_values}")

    beta = (psi * y[:, None]).mean(dim=0)
    beta_sq_cumsum = torch.cumsum(beta.pow(2), dim=0)
    return {r: float(beta_sq_cumsum[r - 1].item()) for r in r_values}


def estimate_B_r_projection(
    psi: torch.Tensor,
    y: torch.Tensor,
    r_values: Iterable[int],
    center_labels: bool = True,
    center_features: bool = True,
    svd_rel_tol: float = 1e-6,
) -> Dict[int, float]:
    """
    Projection-energy variant in [0,1].

    This is useful as a robustness check, but it is not the theorem-faithful
    default because it normalizes by label energy and works at the subspace level
    rather than through the coordinate correlation formula.
    """
    y = _as_pm_one(y)
    if center_labels:
        y = y - y.mean()

    if psi.ndim != 2:
        raise ValueError(f"psi must be 2D, got shape {psi.shape}")
    if psi.shape[0] != y.shape[0]:
        raise ValueError("psi and y must have the same number of samples")

    y_norm_sq = float(torch.dot(y, y).item())
    if y_norm_sq <= 1e-12:
        return {int(r): 0.0 for r in r_values}

    n, k = psi.shape
    r_values = sorted(set(int(r) for r in r_values))
    if len(r_values) == 0:
        return {}
    if min(r_values) < 1 or max(r_values) > k:
        raise ValueError(f"All r must satisfy 1 <= r <= {k}; got {r_values}")

    out: Dict[int, float] = {}
    for r in r_values:
        x = psi[:, :r]
        if center_features:
            x = x - x.mean(dim=0, keepdim=True)
        x = x / math.sqrt(n)

        u, s, _ = torch.linalg.svd(x, full_matrices=False)
        if s.numel() == 0:
            out[r] = 0.0
            continue
        tol = svd_rel_tol * float(s[0].item())
        rank = int((s > tol).sum().item())
        if rank == 0:
            out[r] = 0.0
            continue

        q = u[:, :rank]
        proj_y = q @ (q.T @ y)
        out[r] = float((torch.dot(proj_y, proj_y) / torch.dot(y, y)).item())
    return out
