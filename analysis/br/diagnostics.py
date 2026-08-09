import torch
from typing import Tuple, Optional
from dataclasses import dataclass

@dataclass
class GramStats:
    r: int
    eig_min: float
    eig_max: float
    cond: float
    fro_error: float


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------

def _as_pm_one(y: torch.Tensor) -> torch.Tensor:
    """Map binary labels {0,1} to {-1,+1}; leave {-1,+1} unchanged."""
    y = y.float().reshape(-1)
    if torch.all((y == 0) | (y == 1)):
        y = 2.0 * y - 1.0
    return y


def _sorted_eigh_psd(cov: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    eigvals, eigvecs = torch.linalg.eigh(cov)
    idx = torch.argsort(eigvals, descending=True)
    return eigvals[idx], eigvecs[:, idx]


def _resolve_k_eff(
    eigvals_full: torch.Tensor,
    k_cap: Optional[int],
    rel_eig_threshold: float,
) -> int:
    if eigvals_full.ndim != 1 or eigvals_full.numel() == 0:
        raise ValueError("eigvals_full must be a non-empty 1D tensor")

    lam_max = eigvals_full[0]
    if lam_max <= 0:
        return 1

    keep = eigvals_full >= rel_eig_threshold * lam_max
    k_eff = int(keep.sum().item())
    if k_cap is not None:
        k_eff = min(k_eff, int(k_cap))
    return max(1, min(k_eff, eigvals_full.numel()))

# -----------------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------------

def gram_stats(psi: torch.Tensor, r: int, gram_ridge: float = 0.0) -> GramStats:
    """Diagnostics for empirical orthonormality of the top-r coordinates."""
    n = psi.shape[0]
    psi_r = psi[:, :r]
    g = (psi_r.T @ psi_r) / n
    if gram_ridge > 0.0:
        g = g + gram_ridge * torch.eye(r, device=g.device, dtype=g.dtype)
    eigs = torch.linalg.eigvalsh(g)
    eye = torch.eye(r, device=g.device, dtype=g.dtype)
    eig_min = float(eigs.min().item())
    eig_max = float(eigs.max().item())
    cond = eig_max / max(eig_min, 1e-12)
    fro_error = float(torch.norm(g - eye).item())
    return GramStats(r=r, eig_min=eig_min, eig_max=eig_max, cond=cond, fro_error=fro_error)



def print_basis_stats(x: torch.Tensor, name: str) -> None:
    mean = x.mean(dim=0)
    cov = (x - mean).T @ (x - mean) / x.shape[0]
    eye = torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
    print(f"{name} max |mean| = {mean.abs().max().item():.4e}")
    print(f"{name} cov error Fro = {(cov - eye).norm().item():.4e}")
    print(f"{name} diag range = [{cov.diag().min().item():.4f}, {cov.diag().max().item():.4f}]")
