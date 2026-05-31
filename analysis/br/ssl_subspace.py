import torch
from dataclasses import dataclass
from typing import Optional
from .diagnostics import _sorted_eigh_psd, _resolve_k_eff

@dataclass
class SSLSubspaceEstimator:
    """
    Empirical SSL subspace estimator.

    Attributes
    ----------
    mean_ : torch.Tensor, shape [1, d]
        Mean of the single-view marginal estimated from both views.
    whiten_ : torch.Tensor, shape [d, k_eff]
        Regularized truncated whitening map.
    ssl_eigvecs_ : torch.Tensor, shape [k_eff, k_eff]
        Eigenvectors of the whitened empirical SSL operator, ordered by
        descending eigenvalue.
    ssl_eigvals_ : torch.Tensor, shape [k_eff]
        Corresponding SSL eigenvalues.
    cov_eigvals_ : torch.Tensor, shape [d]
        Sorted covariance eigenvalues of the single-view marginal.
    k_requested : Optional[int]
        External cap requested by the caller.
    k_eff : int
        Effective retained whitening dimension actually used.
    lam_max : float
        Largest covariance eigenvalue.
    rel_eig_threshold : float
        Relative spectral cutoff used to decide which covariance directions
        are retained.
    whiten_ridge_rel : float
        Whitening ridge used as whiten_ridge_rel * lam_max.
    """

    mean_: torch.Tensor
    whiten_: torch.Tensor
    ssl_eigvecs_: torch.Tensor
    ssl_eigvals_: torch.Tensor
    cov_eigvals_: torch.Tensor
    k_requested: Optional[int]
    k_eff: int
    lam_max: float
    rel_eig_threshold: float
    whiten_ridge_rel: float

    def transform(self, z: torch.Tensor, r: Optional[int] = None) -> torch.Tensor:
        """Project features into the empirical SSL basis."""
        phi = (z - self.mean_) @ self.whiten_
        psi = phi @ self.ssl_eigvecs_
        if r is not None:
            if r < 1 or r > psi.shape[1]:
                raise ValueError(f"r must be in [1, {psi.shape[1]}], got {r}")
            psi = psi[:, :r]
        return psi


# -----------------------------------------------------------------------------
# SSL subspace fitting
# -----------------------------------------------------------------------------
def fit_ssl_subspace(
    z1: torch.Tensor,
    z2: torch.Tensor,
    k_cap: Optional[int] = None,
    rel_eig_threshold: float = 1e-3,
    whiten_ridge_rel: float = 1e-3,
) -> SSLSubspaceEstimator:
    """
    Fit the empirical SSL subspace from paired unlabeled views.

    Main changes relative to the original implementation:
      1) choose whitening dimension by a relative covariance spectral cutoff,
      2) add ridge whitening scaled to the top covariance eigenvalue,
      3) return the effective retained rank and covariance spectrum.
    """
    if z1.shape != z2.shape:
        raise ValueError(f"z1 and z2 must have same shape, got {z1.shape} vs {z2.shape}")
    if z1.ndim != 2:
        raise ValueError(f"z1 and z2 must be 2D, got {z1.ndim}D")

    n, d = z1.shape
    z_all = torch.cat([z1, z2], dim=0)
    mean_ = z_all.mean(dim=0, keepdim=True)

    z1c = z1 - mean_
    z2c = z2 - mean_
    z_all_c = z_all - mean_

    cov = (z_all_c.T @ z_all_c) / z_all_c.shape[0]
    eigvals_full, eigvecs_full = _sorted_eigh_psd(cov)

    k_eff = _resolve_k_eff(
        eigvals_full=eigvals_full,
        k_cap=k_cap,
        rel_eig_threshold=rel_eig_threshold,
    )

    eigvals = eigvals_full[:k_eff]
    eigvecs = eigvecs_full[:, :k_eff]

    lam_max = float(max(eigvals_full[0].item(), 0.0))
    ridge = whiten_ridge_rel * lam_max
    inv_sqrt = (eigvals + ridge).rsqrt()
    whiten_ = eigvecs @ torch.diag(inv_sqrt)

    phi1 = z1c @ whiten_
    phi2 = z2c @ whiten_

    # Empirical whitened cross-view operator.
    m = (phi1.T @ phi2) / n
    s = 0.5 * (m + m.T)

    ssl_eigvals, ssl_eigvecs = torch.linalg.eigh(s)
    idx = torch.argsort(ssl_eigvals, descending=True)
    ssl_eigvals = ssl_eigvals[idx]
    ssl_eigvecs = ssl_eigvecs[:, idx]

    return SSLSubspaceEstimator(
        mean_=mean_,
        whiten_=whiten_,
        ssl_eigvecs_=ssl_eigvecs,
        ssl_eigvals_=ssl_eigvals,
        cov_eigvals_=eigvals_full,
        k_requested=k_cap,
        k_eff=k_eff,
        lam_max=lam_max,
        rel_eig_threshold=rel_eig_threshold,
        whiten_ridge_rel=whiten_ridge_rel,
    )