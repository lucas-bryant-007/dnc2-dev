import torch
from dataclasses import dataclass
from typing import Optional
from torch.utils.data import DistributedSampler
from .whitening import fit_exact_whitener


def paired_view_loader_provenance(
    loader,
    first_view: torch.Tensor,
    second_view: torch.Tensor,
) -> dict:
    """Validate and describe the loader that supplied a paired-view fit.

    A record claiming the full empirical training population is only emitted
    when the loader covers its dataset once, without replacement, distributed
    sharding, or a dropped final batch, and the extracted tensors contain one
    row per latent instance.  This is an empirical data-lineage check, not a
    claim about population-level whitening.
    """
    if first_view.shape != second_view.shape or first_view.ndim != 2:
        raise ValueError(
            "paired loader outputs must be same-shape two-dimensional tensors"
        )
    declared = getattr(loader, "dnc2_analysis_provenance", None)
    if not isinstance(declared, dict):
        raise ValueError("paired analysis loader is missing provenance metadata")
    if declared.get("num_augmented_views_per_instance") != 2:
        raise ValueError("paired SSL fitting requires exactly two augmented views")

    dataset_instances = int(len(loader.dataset))
    sampler_instances = int(len(loader.sampler))
    extracted_instances = int(first_view.shape[0])
    sampler = loader.sampler
    sampler_replacement = bool(getattr(sampler, "replacement", False))
    problems = []
    if loader.drop_last:
        problems.append("drop_last=True")
    if isinstance(sampler, DistributedSampler):
        problems.append("distributed sampler")
    if sampler_replacement:
        problems.append("sampling with replacement")
    if sampler_instances != dataset_instances:
        problems.append(
            f"sampler emits {sampler_instances} rows for {dataset_instances} instances"
        )
    if extracted_instances != dataset_instances:
        problems.append(
            f"extracted {extracted_instances} rows for {dataset_instances} instances"
        )
    if problems:
        raise ValueError(
            "paired loader does not cover the full fit population exactly once: "
            + "; ".join(problems)
        )

    return {
        **declared,
        "dataset_instances": dataset_instances,
        "sampler_instances": sampler_instances,
        "extracted_latent_instances": extracted_instances,
        "batch_size": loader.batch_size,
        "num_workers": loader.num_workers,
        "drop_last": bool(loader.drop_last),
        "sampler_class": type(sampler).__name__,
        "sampler_replacement": sampler_replacement,
        "distributed_sharding": False,
        "covers_full_dataset_exactly_once_per_pass": True,
    }

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

    def ssl_eigenvalue_tolerance(self) -> float:
        """Numerical zero used only to audit the paper's ``lambda_r > 0`` condition."""
        scale = max(
            1.0,
            float(torch.max(torch.abs(self.ssl_eigvals_)).item()),
        )
        eps = torch.finfo(self.ssl_eigvals_.dtype).eps
        return float(64.0 * eps * max(self.k_eff, 1) * scale)

    def rank_spectral_diagnostics(self, r: int) -> dict:
        """Record positivity and boundary eigengap for a requested top-r space."""
        if r < 1 or r > self.k_eff:
            raise ValueError(f"r must be in [1, {self.k_eff}], got {r}")
        tolerance = self.ssl_eigenvalue_tolerance()
        lambda_r = float(self.ssl_eigvals_[r - 1].item())
        lambda_next = (
            None if r == self.k_eff else float(self.ssl_eigvals_[r].item())
        )
        eigengap = None if lambda_next is None else lambda_r - lambda_next
        return {
            "rank": int(r),
            "lambda_r": lambda_r,
            "lambda_r_positive_above_numerical_tolerance": lambda_r > tolerance,
            "positive_eigenvalue_tolerance": tolerance,
            "lambda_r_plus_1": lambda_next,
            "boundary_eigengap": eigengap,
            "boundary_eigengap_above_numerical_tolerance": (
                True if eigengap is None else eigengap > tolerance
            ),
            "positive_lambda_required_for_reported_theorem": True,
            "boundary_eigengap_required_for_optimal_value": False,
            "boundary_eigengap_interpretation": (
                "audits_uniqueness_and_stability_of_the_selected_basis_not_"
                "existence_of_a_top_r_optimum"
            ),
        }

    def ssl_spectrum_provenance(self) -> dict:
        """Serialize the empirical SSL spectrum needed to audit top-r claims."""
        tolerance = self.ssl_eigenvalue_tolerance()
        values = [float(value) for value in self.ssl_eigvals_.tolist()]
        return {
            "operator": "symmetrized_whitened_empirical_cross_view_operator",
            "self_adjoint_by_construction": True,
            "positive_semidefinite_population_operator_assumption": (
                "requires_conditionally_iid_exchangeable_views"
            ),
            "eigenvalues_descending": values,
            "positive_eigenvalue_tolerance": tolerance,
            "n_positive_above_numerical_tolerance": sum(
                value > tolerance for value in values
            ),
            "complete_finite_dimensional_eigenbasis": True,
        }

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
            "ssl_spectrum": self.ssl_spectrum_provenance(),
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
