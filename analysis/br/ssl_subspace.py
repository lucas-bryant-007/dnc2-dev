import torch
from dataclasses import dataclass
from typing import Optional
from .whitening import fit_exact_whitener

@dataclass
class SSLSubspaceEstimator:
    """
    Empirical SSL subspace estimator.

    Attributes
    ----------
    mean_ : torch.Tensor, shape [1, d]
        Mean of the single-view marginal estimated from both views.
    whiten_ : torch.Tensor, shape [d, k_eff]
        Exact whitening map on the retained covariance subspace.
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
        Requested relative spectral cutoff.
    numerical_rel_eig_floor : float
        Dtype/dimension-aware lower bound on the relative cutoff.
    effective_rel_eig_threshold : float
        Larger of the requested cutoff and numerical floor.
    covariance_cutoff : float
        Absolute covariance-eigenvalue cutoff actually used.
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
    numerical_rel_eig_floor: float
    effective_rel_eig_threshold: float
    covariance_cutoff: float
    n_fit_latent_instances: int
    n_fit_view_rows: int

    def transform(self, z: torch.Tensor, r: Optional[int] = None) -> torch.Tensor:
        """Project features into the empirical SSL basis."""
        z = z.to(dtype=self.mean_.dtype)
        phi = (z - self.mean_) @ self.whiten_
        psi = phi @ self.ssl_eigvecs_
        if r is not None:
            if r < 1 or r > psi.shape[1]:
                raise ValueError(f"r must be in [1, {psi.shape[1]}], got {r}")
            psi = psi[:, :r]
        return psi

    def first_stage_whitener_provenance(
        self,
        *,
        fit_split: str,
        fit_population: str,
        view_marginal: str,
        frozen_for_test: Optional[bool],
    ) -> dict:
        """Identify the first-stage whitening map defining SSL coordinates."""
        if not fit_split or not fit_population or not view_marginal:
            raise ValueError(
                "fit_split, fit_population, and view_marginal must be nonempty"
            )
        return {
            "stage": "first_stage_ssl_marginal_whitener",
            "fit_split": fit_split,
            "fit_population": fit_population,
            "view_marginal": view_marginal,
            "n_fit_latent_instances": self.n_fit_latent_instances,
            "n_fit_view_rows": self.n_fit_view_rows,
            "input_dimension": int(self.mean_.shape[1]),
            "requested_rank_cap": self.k_requested,
            "retained_rank": self.k_eff,
            "relative_rank_cutoff": {
                "requested": self.rel_eig_threshold,
                "numerical_floor": self.numerical_rel_eig_floor,
                "effective": self.effective_rel_eig_threshold,
            },
            "absolute_covariance_eigenvalue_cutoff": self.covariance_cutoff,
            "retained_covariance_eigenvalue_range": {
                "min": float(self.cov_eigvals_[self.k_eff - 1].item()),
                "max": self.lam_max,
            },
            "transform_frozen_after_fit": True,
            "frozen_for_downstream_evaluation": True,
            "frozen_for_test": frozen_for_test,
        }


# -----------------------------------------------------------------------------
# SSL subspace fitting
# -----------------------------------------------------------------------------
def fit_ssl_subspace(
    z1: torch.Tensor,
    z2: torch.Tensor,
    k_cap: Optional[int] = None,
    rel_eig_threshold: float = 1e-3,
) -> SSLSubspaceEstimator:
    """
    Fit the empirical SSL subspace from paired unlabeled views.

    The retained marginal covariance directions are scaled by their exact
    inverse square roots. Directions below the relative spectral cutoff are
    discarded rather than ridge-shrunk, so the fitted coordinates have
    empirical covariance identity as required by the paper.
    """
    if z1.shape != z2.shape:
        raise ValueError(f"z1 and z2 must have same shape, got {z1.shape} vs {z2.shape}")
    if z1.ndim != 2:
        raise ValueError(f"z1 and z2 must be 2D, got {z1.ndim}D")

    n = z1.shape[0]
    z_all = torch.cat([z1, z2], dim=0)
    whitening = fit_exact_whitener(
        z_all,
        rel_eig_threshold=rel_eig_threshold,
        k_cap=k_cap,
    )
    z1_fit = z1.to(dtype=whitening.mean.dtype)
    z2_fit = z2.to(dtype=whitening.mean.dtype)
    phi1 = (z1_fit - whitening.mean) @ whitening.whitener
    phi2 = (z2_fit - whitening.mean) @ whitening.whitener

    # Empirical whitened cross-view operator.
    m = (phi1.T @ phi2) / n
    s = 0.5 * (m + m.T)

    ssl_eigvals, ssl_eigvecs = torch.linalg.eigh(s)
    idx = torch.argsort(ssl_eigvals, descending=True)
    ssl_eigvals = ssl_eigvals[idx]
    ssl_eigvecs = ssl_eigvecs[:, idx]

    return SSLSubspaceEstimator(
        mean_=whitening.mean,
        whiten_=whitening.whitener,
        ssl_eigvecs_=ssl_eigvecs,
        ssl_eigvals_=ssl_eigvals,
        cov_eigvals_=whitening.covariance_eigenvalues,
        k_requested=k_cap,
        k_eff=whitening.output_dim,
        lam_max=whitening.lambda_max,
        rel_eig_threshold=whitening.rel_eig_threshold,
        numerical_rel_eig_floor=whitening.numerical_rel_eig_floor,
        effective_rel_eig_threshold=whitening.effective_rel_eig_threshold,
        covariance_cutoff=whitening.cutoff,
        n_fit_latent_instances=int(n),
        n_fit_view_rows=int(2 * n),
    )
