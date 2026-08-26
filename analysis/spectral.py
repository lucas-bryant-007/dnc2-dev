import math
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
import random
from dataclasses import dataclass
from typing import Dict, Iterable, Optional
from training.config_loader import load_config, dict_to_namespace, namespace_to_dict
from data_utils import MiniImageNetDataModule, MiniImageNetCfg

from eval_utils import (find_checkpoint_files, load_model_from_checkpoint, 
                        extract_features, set_seed, freeze_model, get_subset_dataloader)
from geometry import GeometricEvaluator
from metrics_io import write_json
from br.whitening import (
    ABSOLUTE_WHITENING_ELIGIBILITY_POLICY,
    absolute_whitening_eligibility,
    canonicalize_balanced_binary_labels,
    fit_exact_whitener,
    split_balanced_paired_fit_eval,
    whitening_diagnostics,
)

@dataclass
class SSLSubspaceEstimator:
    """
    Container for the estimated SSL subspace.

    Attributes
    ----------
    mean_ : torch.Tensor, shape [1, d]
        Mean used to center features.

    whiten_ : torch.Tensor, shape [d, k]
        Truncated whitening map. If x is a raw feature vector in R^d,
        then (x - mean_) @ whiten_ gives whitened coordinates in R^k.

    ssl_eigvecs_ : torch.Tensor, shape [k, k]
        Eigenvectors of the empirical SSL cross-view operator in whitened
        coordinates. Columns are ordered by descending eigenvalue.

    ssl_eigvals_ : torch.Tensor, shape [k]
        Corresponding empirical SSL eigenvalues, sorted descending.

    k : int
        Whitening truncation dimension.
    """
    mean_: torch.Tensor
    whiten_: torch.Tensor
    ssl_eigvecs_: torch.Tensor
    ssl_eigvals_: torch.Tensor
    k: int
    rel_eig_threshold: float
    numerical_rel_eig_floor: float
    effective_rel_eig_threshold: float
    covariance_cutoff: float
    covariance_eigenvalues_: torch.Tensor
    k_requested: Optional[int]
    lambda_max: float
    n_fit_latent_instances: int

    def transform(self, z: torch.Tensor, r: Optional[int] = None) -> torch.Tensor:
        """
        Project features into the estimated SSL basis.

        Parameters
        ----------
        z : torch.Tensor, shape [N, d]
            Feature matrix from the same encoder used to fit the estimator.
        r : int or None
            Number of top SSL directions to return. If None, returns all k.

        Returns
        -------
        psi : torch.Tensor, shape [N, r] or [N, k]
            Coordinates in the empirical SSL basis. Columns are ordered from
            most to least dominant SSL direction.
        """
        z = z.to(dtype=self.mean_.dtype)
        phi = (z - self.mean_) @ self.whiten_          # [N, k]
        psi = phi @ self.ssl_eigvecs_                  # [N, k]
        if r is not None:
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
        return {
            "stage": "first_stage_ssl_marginal_whitener",
            "fit_split": fit_split,
            "fit_population": fit_population,
            "view_marginal": view_marginal,
            "n_fit_latent_instances": self.n_fit_latent_instances,
            "n_fit_view_rows": 2 * self.n_fit_latent_instances,
            "input_dimension": int(self.mean_.shape[1]),
            "requested_rank_cap": self.k_requested,
            "retained_rank": self.k,
            "relative_rank_cutoff": {
                "requested": self.rel_eig_threshold,
                "numerical_floor": self.numerical_rel_eig_floor,
                "effective": self.effective_rel_eig_threshold,
            },
            "absolute_covariance_eigenvalue_cutoff": self.covariance_cutoff,
            "retained_covariance_eigenvalue_range": {
                "min": float(self.covariance_eigenvalues_[self.k - 1].item()),
                "max": self.lambda_max,
            },
            "transform_frozen_after_fit": True,
            "frozen_for_downstream_evaluation": True,
            "frozen_for_test": frozen_for_test,
        }
    
def _as_pm_one(y: torch.Tensor) -> torch.Tensor:
    """
    Convert labels to approximately {-1, +1} if they are in {0, 1}.

    Parameters
    ----------
    y : torch.Tensor, shape [N]

    Returns
    -------
    y_out : torch.Tensor, shape [N], dtype float
    """
    y = y.float().reshape(-1)
    # If labels look binary {0,1}, map to {-1,+1}
    if torch.all((y == 0) | (y == 1)):
        y = 2 * y - 1
    return y

def fit_ssl_subspace(
    z1: torch.Tensor,
    z2: torch.Tensor,
    k: Optional[int] = None,
    eps: float = 1e-3,
) -> SSLSubspaceEstimator:
    """
    Estimate the empirical SSL directions from paired augmentations.

    This follows the operator-style pipeline:
      1) estimate the marginal mean and covariance from BOTH views,
      2) whiten into a k-dimensional space,
      3) estimate the cross-view operator in whitened coordinates,
      4) eigendecompose that operator to obtain SSL directions.

    Parameters
    ----------
    z1, z2 : torch.Tensor, shape [N, d]
        Paired features from two augmentations of the same instances,
        extracted by the same trained encoder.
    k : int or None
        Truncation dimension for whitening. If None, uses full dimension d.
        In practice, choose k <= min(N, d) and often much smaller than d.
    eps : float
        Relative covariance eigenvalue cutoff. Directions below
        ``eps * lambda_max`` are discarded before exact whitening.

    Returns
    -------
    estimator : SSLSubspaceEstimator
        Fitted object containing mean, whitening map, and SSL basis.

    Notes
    -----
    - Using BOTH views to estimate mean/covariance is more symmetric than using
      only z1.
    - The empirical cross-view matrix is symmetrized before eigendecomposition.
      This is a practical finite-sample stabilization.
    """
    if z1.shape != z2.shape:
        raise ValueError(f"z1 and z2 must have same shape, got {z1.shape} vs {z2.shape}")
    if z1.ndim != 2:
        raise ValueError(f"z1 and z2 must be 2D, got {z1.ndim}D")
    if not 0.0 < eps <= 1.0:
        raise ValueError(f"eps must lie in (0, 1], got {eps}")

    N, d = z1.shape
    k_requested = k
    if k is None:
        k = d
    k = min(k, d)

    # Fit through the shared primitive so float64 is preserved and a numerical
    # rank floor protects small user-supplied cutoffs in rank-deficient data.
    whitening = fit_exact_whitener(
        torch.cat((z1, z2), dim=0),
        rel_eig_threshold=eps,
        k_cap=k,
    )
    z1_fit = z1.to(dtype=whitening.mean.dtype)
    z2_fit = z2.to(dtype=whitening.mean.dtype)
    phi1 = (z1_fit - whitening.mean) @ whitening.whitener
    phi2 = (z2_fit - whitening.mean) @ whitening.whitener

    check_basis_stats(phi1, name="phi1")
    check_basis_stats(phi2, name="phi2")

    # ------------------------------------------------------------
    # 3) Estimate whitened cross-view operator
    # ------------------------------------------------------------
    M = (phi1.T @ phi2) / N                           # [k, k]

    # Symmetrize for numerical stability / self-adjoint population analogue
    S = 0.5 * (M + M.T)

    # ------------------------------------------------------------
    # 4) SSL directions = eigenvectors of the empirical operator
    # ------------------------------------------------------------
    ssl_eigvals, ssl_eigvecs = torch.linalg.eigh(S)   # ascending
    idx = torch.argsort(ssl_eigvals, descending=True)
    ssl_eigvals = ssl_eigvals[idx]
    ssl_eigvecs = ssl_eigvecs[:, idx]

    return SSLSubspaceEstimator(
        mean_=whitening.mean,
        whiten_=whitening.whitener,
        ssl_eigvecs_=ssl_eigvecs,
        ssl_eigvals_=ssl_eigvals,
        k=whitening.output_dim,
        rel_eig_threshold=whitening.rel_eig_threshold,
        numerical_rel_eig_floor=whitening.numerical_rel_eig_floor,
        effective_rel_eig_threshold=whitening.effective_rel_eig_threshold,
        covariance_cutoff=whitening.cutoff,
        covariance_eigenvalues_=whitening.covariance_eigenvalues,
        k_requested=k_requested,
        lambda_max=whitening.lambda_max,
        n_fit_latent_instances=int(N),
    )

def estimate_B_r(
    psi: torch.Tensor,
    y: torch.Tensor,
    r_values: Iterable[int],
    center_labels: bool = False,
) -> Dict[int, float]:
    """
    Estimate B_r from labeled data in the empirical SSL basis.

    Parameters
    ----------
    psi : torch.Tensor, shape [N, k]
        SSL coordinates for the labeled examples. Columns must be ordered
        from top to bottom SSL direction.
    y : torch.Tensor, shape [N]
        Binary labels in {-1,+1} or {0,1}. For the cleanest match to theory,
        labels should correspond to an approximately balanced binary task.
    r_values : iterable of int
        Values of r for which to return estimates. Each r must satisfy 1 <= r <= k.
    center_labels : bool
        If True, subtract the sample mean of y before computing the estimator.
        This can help when the labeled sample is not exactly balanced.
        If your task is already balanced and labels are in {-1,+1}, leave False.

    Returns
    -------
    B_r : dict
        Mapping r -> estimated captured posterior energy.

    Theory
    ------
    For balanced binary labels and whitened/orthonormal SSL coordinates:
        B_r = sum_{j=1}^r ( E[Y psi_j(X)] )^2
    [1]

    Empirical estimator:
        \\hat B_r = sum_{j=1}^r ( mean_i y_i psi_{ij} )^2
    """
    y = _as_pm_one(y)

    if center_labels:
        y = y - y.mean()

    if psi.ndim != 2:
        raise ValueError(f"psi must be 2D, got shape {psi.shape}")
    if psi.shape[0] != y.shape[0]:
        raise ValueError(f"psi and y must have same number of samples, got {psi.shape[0]} vs {y.shape[0]}")

    N, k = psi.shape
    r_values = list(r_values)
    if len(r_values) == 0:
        return {}

    if min(r_values) < 1 or max(r_values) > k:
        raise ValueError(f"All r must satisfy 1 <= r <= {k}; got {r_values}")

    # beta_j = E_n[Y psi_j(X)]
    beta = (psi * y[:, None]).mean(dim=0)             # [k]
    beta_sq_cumsum = torch.cumsum(beta.pow(2), dim=0)

    out = {}
    for r in r_values:
        out[r] = beta_sq_cumsum[r - 1].item()
    return out

def estimate_B_r_corrected(psi: torch.Tensor,
    y: torch.Tensor,
    r_values: Iterable[int],
    center_labels: bool = False,
    ridge=1e-4):
    y = _as_pm_one(y)
    if center_labels:
        y = y - y.mean()
    if psi.ndim != 2:
        raise ValueError(f"psi must be 2D, got shape {psi.shape}")
    if psi.shape[0] != y.shape[0]:
        raise ValueError("psi and y must have the same number of rows")
    N, k = psi.shape
    r_values = list(r_values)
    if r_values and (min(r_values) < 1 or max(r_values) > k):
        raise ValueError(f"all r must satisfy 1 <= r <= {k}; got {r_values}")
    out = {}
    for r in r_values:
        Psi = psi[:, :r]
        beta = (Psi * y[:, None]).mean(dim=0)         # [r]
        G = (Psi.T @ Psi) / N                         # [r, r]
        G = G + ridge * torch.eye(r, device=G.device, dtype=G.dtype)
        out[r] = (beta @ torch.linalg.solve(G, beta)).item()
    return out

def estimate_B_r_projection(
    psi: torch.Tensor,
    y: torch.Tensor,
    r_values,
    center_labels: bool = True,
    center_features: bool = True,
    svd_rel_tol: float = 1e-6,
):
    y = y.float().reshape(-1)
    if torch.all((y == 0) | (y == 1)):
        y = 2 * y - 1
    if center_labels:
        y = y - y.mean()
    y_norm_sq = torch.dot(y, y)
    if y_norm_sq <= 1e-12:
        return {r: 0.0 for r in r_values}
    N, k = psi.shape
    out = {}
    for r in r_values:
        X = psi[:, :r]
        if center_features:
            X = X - X.mean(dim=0, keepdim=True)
        # Scale so columns live in empirical L2(P_n)
        X = X / math.sqrt(N)
        # Stable orthonormal basis for span(X)
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        if S.numel() == 0:
            out[r] = 0.0
            continue
        tol = svd_rel_tol * S[0]
        rank = int((S > tol).sum().item())
        if rank == 0:
            out[r] = 0.0
            continue
        Q = U[:, :rank]          # orthonormal basis for span(X)
        proj_y = Q @ (Q.T @ y)   # P_r y
        out[r] = (torch.dot(proj_y, proj_y) / y_norm_sq).item()
    return out

def estimate_B_r_orth(psi, y, r_values, center_labels=True, ridge=1e-6):
    y = y.float().reshape(-1)
    if torch.all((y == 0) | (y == 1)):
        y = 2 * y - 1
    if center_labels:
        y = y - y.mean()

    N, k = psi.shape
    out = {}

    for r in r_values:
        Psi = psi[:, :r]                        # [N, r]
        G = (Psi.T @ Psi) / N                   # [r, r]
        G = G + ridge * torch.eye(r, device=G.device, dtype=G.dtype)
        beta = (Psi.T @ y) / N                  # [r]
        out[r] = (beta @ torch.linalg.solve(G, beta)).item()

    return out

def estimate_tilde_V(
    psi: torch.Tensor,
    y: torch.Tensor,
    r: int,
) -> float:
    """
    Estimate directional CDNV \u1e7dV on the labeled data using the top-r SSL coordinates.

    Parameters
    ----------
    psi : torch.Tensor, shape [N, k]
        SSL coordinates for labeled examples, ordered by SSL importance.
    y : torch.Tensor, shape [N]
        Binary labels in {-1,+1} or {0,1}.
    r : int
        Number of top SSL coordinates to use.

    Returns
    -------
    tilde_V : float
        Empirical directional CDNV:
            (u^T Sigma_+ u + u^T Sigma_- u) / ||Delta||^2
        where Delta is the class-mean gap and u = Delta / ||Delta||.

    Notes
    -----
    In the whitened population-optimal regime, the paper proves
        \\tilde V = (1 - B_r) / (2 B_r)
    for balanced binary tasks [1].
    """
    y = _as_pm_one(y)
    F = psi[:, :r]

    pos = (y == 1)
    neg = (y == -1)

    if pos.sum() == 0 or neg.sum() == 0:
        raise ValueError("Both classes must be present to estimate tilde_V.")

    mu_pos = F[pos].mean(dim=0)
    mu_neg = F[neg].mean(dim=0)
    delta = mu_pos - mu_neg

    delta_norm_sq = torch.dot(delta, delta)
    if delta_norm_sq <= 1e-12:
        return float("inf")

    u = delta / torch.sqrt(delta_norm_sq)

    def directional_variance(X: torch.Tensor) -> torch.Tensor:
        Xc = X - X.mean(dim=0, keepdim=True)
        proj = Xc @ u
        return (proj ** 2).mean()

    var_pos = directional_variance(F[pos])
    var_neg = directional_variance(F[neg])

    tilde_V = (var_pos + var_neg) / delta_norm_sq
    return tilde_V.item()


def predict_tilde_V_from_B(B_r: float, eps: float = 1e-12) -> float:
    """
    Predict directional CDNV from B_r using the population formula.

    Parameters
    ----------
    B_r : float
        Estimated captured posterior energy.
    eps : float
        Small stabilizer for numerical safety.

    Returns
    -------
    pred : float
        Predicted directional CDNV:
            (1 - B_r) / (2 B_r)
        [1]
    """
    if B_r <= eps:
        return float("inf")
    return (1.0 - B_r) / (2.0 * B_r)

def run_br_pipeline(
    z1_unlab: torch.Tensor,
    z2_unlab: torch.Tensor,
    z_lab: torch.Tensor,
    y_lab: torch.Tensor,
    r_values: Iterable[int],
    k: Optional[int] = None,
    eps: float = 1e-6,
    center_labels: bool = False,
    compute_observed_tilde_V: bool = True,
    fit_labels: Optional[torch.Tensor] = None,
    evaluation_instance_ids: Optional[torch.Tensor] = None,
    max_whiten_mean_l2_error: float = 0.05,
    max_whiten_operator_error: float = 0.10,
):
    """
    End-to-end pipeline:
      1) fit empirical SSL subspace on unlabeled paired views,
      2) transform labeled data into SSL coordinates,
      3) estimate B_r for requested r,
      4) optionally estimate observed \\tilde V and compare to prediction.

    Parameters
    ----------
    z1_unlab, z2_unlab : torch.Tensor, shape [N_u, d]
        Paired unlabeled features from two augmentations.
    z_lab : torch.Tensor, shape [N_l, d]
        Labeled features from the same encoder.
    y_lab : torch.Tensor, shape [N_l]
        Binary labels in {-1,+1} or {0,1}.
    r_values : iterable of int
        Requested subspace sizes.
    k : int or None
        Whitening truncation dimension used before SSL eigendecomposition.
    eps : float
        Relative covariance cutoff for exact rank-truncated whitening.
    center_labels : bool
        Whether to center labels when estimating B_r.
    compute_observed_tilde_V : bool
        If True, also compute empirical directional CDNV for each r.

    The whitening cutoffs are fixed absolute empirical-geometry criteria, not
    a population-whiteness test. Failed ranks remain in the returned
    diagnostics but are omitted from theorem formula dictionaries.

    Returns
    -------
    results : dict
        Dictionary with keys:
          - "estimator": SSLSubspaceEstimator
          - "psi_lab": labeled SSL coordinates
          - "B_r": dict r -> estimate
          - "tilde_V_pred": dict r -> predicted directional CDNV from B_r
          - "tilde_V_obs": dict r -> observed directional CDNV (optional)
    """
    if fit_labels is None or fit_labels.reshape(-1).numel() != z1_unlab.shape[0]:
        raise ValueError("fit_labels must contain one label per paired fit instance")
    if y_lab.reshape(-1).numel() != z_lab.shape[0]:
        raise ValueError("y_lab must contain one label per evaluation row")
    if (
        evaluation_instance_ids is None
        or evaluation_instance_ids.reshape(-1).numel() != z_lab.shape[0]
    ):
        raise ValueError(
            "evaluation_instance_ids must contain one ID per evaluation row"
        )
    n_independent_eval_instances = int(
        torch.unique(evaluation_instance_ids.detach().cpu()).numel()
    )
    _, fit_balance = canonicalize_balanced_binary_labels(
        fit_labels,
        "whitening-fit latent-instance population",
    )
    y_lab, eval_balance = canonicalize_balanced_binary_labels(
        y_lab,
        "theorem-evaluation population",
    )
    population_balance = {
        "whitening_fit_instances": fit_balance,
        "theorem_evaluation_rows": eval_balance,
    }
    estimator = fit_ssl_subspace(z1_unlab, z2_unlab, k=k, eps=eps)
    first_stage_ssl_whitener = estimator.first_stage_whitener_provenance(
        fit_split="balanced_whitening_fit_fold",
        fit_population=(
            "balanced_paired_training_instances_disjoint_from_theorem_evaluation"
        ),
        view_marginal=(
            "equal_weight_empirical_mixture_of_two_augmented_views_per_instance"
        ),
        frozen_for_test=None,
    )
    psi_lab = estimator.transform(z_lab)  # [N_l, k]
    requested_r_values = sorted(
        set(int(r) for r in r_values if 1 <= int(r) <= estimator.k)
    )
    if not requested_r_values:
        raise ValueError(
            f"No valid r values remain after truncation to k_eff={estimator.k}"
        )

    whitening_by_r = {}
    for r in requested_r_values:
        diagnostic = whitening_diagnostics(
            psi_lab[:, :r],
            n_independent_samples=n_independent_eval_instances,
        )
        whitening_by_r[r] = {
            **diagnostic.as_dict(),
            **absolute_whitening_eligibility(
                diagnostic,
                max_mean_l2_error=max_whiten_mean_l2_error,
                max_operator_error=max_whiten_operator_error,
            ),
            "scope": "out_of_sample_same_view_marginal",
            "check_kind": "absolute_empirical_geometry_eligibility",
        }
    eligible_r_values = [
        r for r in requested_r_values if whitening_by_r[r]["eligible"]
    ]
    ineligible_r_values = [
        r for r in requested_r_values if not whitening_by_r[r]["eligible"]
    ]
    r_values = eligible_r_values

    check_basis_stats(psi_lab, name="psi_lab")

    # for r in r_values:
    #     if r < 1 or r > psi_lab.shape[1]:
    #         raise ValueError(f"Invalid r={r} for psi_lab with {psi_lab.shape[1]} dimensions")
    #     psi_r = psi_lab[:, :r]
    #     N = psi_r.shape[0]
    #     G = (psi_r.T @ psi_r) / N
    #     evals = torch.linalg.eigvalsh(G)

    #     print(f"\nStatistics for top-{r} SSL subspace:")
    #     print("Gram eig min/max:", evals.min().item(), evals.max().item())
    #     print("||G-I||_F:", torch.norm(G - torch.eye(r, device=G.device)).item())

    B_r = estimate_B_r_corrected(
        psi=psi_lab,
        y=y_lab,
        r_values=r_values,
        center_labels=center_labels,
    )

    tilde_V_pred = {r: predict_tilde_V_from_B(B_r[r]) for r in r_values}

    results = {
        "estimator": estimator,
        "psi_lab": psi_lab,
        "first_stage_ssl_whitener": first_stage_ssl_whitener,
        "requested_r_values": requested_r_values,
        "r_values": r_values,
        "eligible_r_values": eligible_r_values,
        "ineligible_r_values": ineligible_r_values,
        "B_r": B_r,
        "tilde_V_pred": tilde_V_pred,
        "whitening_diagnostics": whitening_by_r,
        "all_requested_ranks_eligible": not ineligible_r_values,
        "whitening_eligibility_policy": (
            ABSOLUTE_WHITENING_ELIGIBILITY_POLICY
        ),
        "population_balance": population_balance,
    }

    if compute_observed_tilde_V:
        tilde_V_obs = {r: estimate_tilde_V(psi_lab, y_lab, r) for r in r_values}
        results["tilde_V_obs"] = tilde_V_obs

    return results

def check_basis_stats(psi, name="psi"):
    mean = psi.mean(0)
    cov = (psi - mean).T @ (psi - mean) / psi.shape[0]
    eye = torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
    print(f"{name} max |mean| = {mean.abs().max().item():.4e}")
    print(f"{name} cov error Fro = {(cov - eye).norm().item():.4e}")
    print(f"{name} diag range = [{cov.diag().min().item():.4f}, {cov.diag().max().item():.4f}]")

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
def plot_Br_vs_r(all_results, save_path=None, title="B_r vs r"):
    plt.figure(figsize=(7, 5))
    
    for k, res in sorted(all_results.items()):
        r_values = res["r_values"]
        b_vals = [res["B_r"][r] for r in r_values]
        plt.plot(r_values, b_vals, marker='o', label=f'k={k}')
    
    plt.axhline(1.0, color='gray', linestyle='--', linewidth=1, label='B_r = 1')
    plt.xlabel("r")
    plt.ylabel("B_r")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=200)
    plt.show()

def plot_tildeV_scatter_pretty(all_results, save_path=None,
                               title="Predicted vs Observed Directional CDNV"):
    plt.figure(figsize=(8.5, 6.5)) # Slightly wider to accommodate colorbar

    markers = ['o', 's', '^', 'D', 'P', 'X']
    cmap = plt.cm.viridis

    # --- STEP 1: Determine global r range with LogNorm ---
    all_r_values = []
    for k in all_results:
        all_r_values.extend(all_results[k]["r_values"])
    
    r_min, r_max = min(all_r_values), max(all_r_values)
    
    # Use LogNorm instead of Normalize for better color distribution
    norm = LogNorm(vmin=max(r_min, 1e-3), vmax=r_max) 
    
    # Try 'plasma' or 'magma' for higher contrast
    cmap = plt.cm.plasma
    # ----------------------------------------------------------------

    all_pred, all_obs = [], []
    ks = sorted(all_results.keys())
    
    for idx, k in enumerate(ks):
        res = all_results[k]
        r_values = np.array(res["r_values"])
        pred_vals = np.array([res["tilde_V_pred"][r] for r in r_values], dtype=float)
        obs_vals  = np.array([res["tilde_V_obs"][r]  for r in r_values], dtype=float)

        mask = np.isfinite(pred_vals) & np.isfinite(obs_vals) & (pred_vals > 0) & (obs_vals > 0)
        pred_vals = pred_vals[mask]
        obs_vals = obs_vals[mask]
        kept_r = r_values[mask]

        all_pred.extend(pred_vals.tolist())
        all_obs.extend(obs_vals.tolist())

        # --- STEP 2: Use the global norm for colors ---
        colors = cmap(norm(kept_r))

        plt.scatter(
            pred_vals,
            obs_vals,
            s=90,
            c=colors,
            marker=markers[idx % len(markers)],
            alpha=0.9,
            edgecolor='black',
            linewidth=0.4,
            label=f'k={k}'
        )

    if not all_pred:
        print("No valid points to plot.")
        return

    # Standardize axes and plot y=x
    all_pred, all_obs = np.array(all_pred), np.array(all_obs)
    lo, hi = min(all_pred.min(), all_obs.min()), max(all_pred.max(), all_obs.max())
    grid = np.logspace(np.log10(lo), np.log10(hi), 200)
    plt.plot(grid, grid, 'k--', linewidth=1.5, label='y=x')

    # Formatting
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Predicted directional CDNV")
    plt.ylabel("Observed directional CDNV")
    plt.title(title)
    plt.grid(True, which='both', alpha=0.25)
    
    # --- STEP 3: Add the Legends ---
    # Legend for the markers (k values)
    marker_legend = plt.legend(loc='upper left', title="Configurations")
    plt.gca().add_artist(marker_legend) # Add this back so the next legend doesn't overwrite it

    # Colorbar for the r values
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label('r values (log scale)', rotation=270, labelpad=15)
    
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=220, bbox_inches='tight')
    plt.show()

def main(args):
    set_seed(69)
    if not 0.0 < args.ssl_fit_fraction < 1.0:
        raise ValueError("--ssl_fit_fraction must lie in (0,1)")
    if (
        args.max_whiten_mean_l2_error <= 0
        or args.max_whiten_operator_error <= 0
    ):
        raise ValueError("absolute whitening eligibility cutoffs must be positive")
    # Load config from YAML file
    cfg = load_config(args.config)
    # Convert dict to namespace for easier access (cfg.data.x instead of cfg['data']['x'])
    cfg = dict_to_namespace(cfg)
    run_method = str(cfg.method.name)

    # build data module
    data_cfg = MiniImageNetCfg(**namespace_to_dict(cfg.data))
    data_cfg.method = cfg.method.name
    data_module = MiniImageNetDataModule(data_cfg)
    data_module.setup()

    train_loader = data_module.train_dataloader() # required (w/ augmentations) to estimate SSL subspace
    # Used only for raw-space descriptive geometry; theorem-facing quantities
    # use the held-out augmented-view marginal constructed below.
    sv_train_loader = data_module.probe_train_dataloader()
    # pick 2 random classes and filter the dataloaders to only include those classes
    selected_classes = random.sample(range(100), 2)
    # selected_classes = [0, 1]
    print(f"Selected classes: {selected_classes}")
    train_loader_b = get_subset_dataloader(train_loader.dataset, selected_classes, batch_size=128,
                                         collate_fn=data_module.train_collate)
    sv_train_loader_b = get_subset_dataloader(sv_train_loader.dataset, selected_classes, batch_size=128,
                                         collate_fn=data_module.eval_collate)
    # sv_test_loader_b = get_subset_dataloader(sv_test_loader.dataset, selected_classes, batch_size=128,
    #                                      collate_fn=data_module.eval_collate)

    # build model 
    # get all checkpoint files in the directory
    ckpt_files = find_checkpoint_files(args.ckpt_dir)

    epochs_to_evaluate = [1000]
    all_results = {}
    for epoch, ckpt_path in ckpt_files:
        if epoch in epochs_to_evaluate:
            print(f"Evaluating checkpoint: {ckpt_path} (epoch {epoch})")
            model, cfg = load_model_from_checkpoint(ckpt_path)
            model = model.to('cuda')
            freeze_model(model)

            # get features for both augmented views on the training set (required to fit SSL subspace)
            features_view1, features_view2, paired_labels = extract_features(train_loader_b, model.backbone, device='cuda', both_views=True)
            print(f"Extracted train features of shape: {features_view1.shape} and {features_view2.shape}")
            population_split = split_balanced_paired_fit_eval(
                features_view1,
                features_view2,
                paired_labels,
                fit_fraction=args.ssl_fit_fraction,
                seed=69,
            )
            ssl_fit_view1 = population_split.fit_view1
            ssl_fit_view2 = population_split.fit_view2
            theorem_eval_features = population_split.evaluation_features
            theorem_eval_labels = population_split.evaluation_labels
            print(
                "Balanced, population-matched theorem split: "
                f"SSL fit={ssl_fit_view1.shape[0]} instances, "
                f"evaluation={population_split.eval_instance_ids.numel()} "
                f"instances/{theorem_eval_features.shape[0]} augmented views, "
                f"counts={population_split.metadata()}"
            )

            # Single-view features are reported only as raw-space diagnostics.
            sv_train_features_b, train_labels_b = extract_features(sv_train_loader_b, model.backbone, device='cuda')
            # breakpoint()
            
            # estimate CDNV and directional CDNV
            geometric_evaluator = GeometricEvaluator(num_classes=2, device='cuda')
            cdnv = geometric_evaluator.compute_cdnv(sv_train_features_b, train_labels_b)
            directional_cdnv = geometric_evaluator.compute_directional_cdnv(sv_train_features_b, train_labels_b)
            print(f"Estimated CDNV on binary train set: {cdnv:.4f}")
            print(f"Estimated directional CDNV on binary train set: {directional_cdnv:.4f}")

            # K = [64, 128, 256, 512, 1024]
            K=[2048]
            R = [8, 16, 32, 64, 128, 256, 512, 1024]
            for k in K:
                print(f"\n=== Evaluating for k={k} ===")
                r_values = [r for r in R if r <= k]
                results = run_br_pipeline(
                    z1_unlab=ssl_fit_view1,
                    z2_unlab=ssl_fit_view2,
                    z_lab=theorem_eval_features,
                    y_lab=theorem_eval_labels,
                    r_values=r_values,
                    k=k,
                    eps=1e-3,
                    center_labels=False,
                    compute_observed_tilde_V=True,
                    fit_labels=population_split.fit_labels,
                    evaluation_instance_ids=(
                        population_split.evaluation_instance_ids
                    ),
                    max_whiten_mean_l2_error=args.max_whiten_mean_l2_error,
                    max_whiten_operator_error=args.max_whiten_operator_error,
                )

                used_r_values = sorted(results["B_r"])
                all_results[k] = {
                    "first_stage_ssl_whitener": results[
                        "first_stage_ssl_whitener"
                    ],
                    "requested_r_values": results["requested_r_values"],
                    "r_values": used_r_values,
                    "eligible_r_values": results["eligible_r_values"],
                    "ineligible_r_values": results["ineligible_r_values"],
                    "B_r": results["B_r"],
                    "tilde_V_pred": results["tilde_V_pred"],
                    "tilde_V_obs": results["tilde_V_obs"],
                    "whitening_diagnostics": results[
                        "whitening_diagnostics"
                    ],
                    "all_requested_ranks_eligible": results[
                        "all_requested_ranks_eligible"
                    ],
                    "whitening_eligibility_policy": results[
                        "whitening_eligibility_policy"
                    ],
                    "population_balance": results["population_balance"],
                }

                print("B_r estimates:")
                for r, v in results["B_r"].items():
                    print(f"r={r:3d}  B_r={v:.6f}")

                print("\nPredicted vs observed directional CDNV:")
                for r in used_r_values:
                    pred = results["tilde_V_pred"][r]
                    obs = results["tilde_V_obs"][r]
                    print(f"r={r:3d}  pred={pred:.6f}  obs={obs:.6f}")

            class_token = "-".join(str(value) for value in selected_classes)
            metrics_path = os.path.join(
                args.out_dir,
                "metrics",
                f"spectral_{run_method}_classes_{class_token}_epoch_{epoch}.json",
            )
            write_json(metrics_path, {
                "method": run_method,
                "epoch": epoch,
                "checkpoint": ckpt_path,
                "selected_classes": selected_classes,
                "population_split": population_split.metadata(),
                "whitening_eligibility_policy": {
                    "kind": ABSOLUTE_WHITENING_ELIGIBILITY_POLICY,
                    "meaning": (
                        "strict_absolute_empirical_geometry_not_population_test"
                    ),
                    "predeclared": True,
                    "applied_per_rank": True,
                    "ineligible_rank_handling": "theorem_formulas_suppressed",
                    "failed_rank_diagnostics_persisted": True,
                    "max_mean_l2_error": args.max_whiten_mean_l2_error,
                    "max_operator_error": args.max_whiten_operator_error,
                    "independent_sample_unit": "latent_instance",
                    "sampling_normalized_errors": (
                        "reported_against_isotropic_gaussian_reference_scales_"
                        "but_not_used_for_eligibility"
                    ),
                },
                "results": all_results,
            })
            print(f"Saved spectral eligibility artifact: {metrics_path}")

            if any(row["eligible_r_values"] for row in all_results.values()):
                plot_Br_vs_r(
                    all_results,
                    save_path=(
                        f"figures/Br_vs_r/Br_vs_r_epoch_{epoch}.png"
                    ),
                )
            else:
                print(
                    "No theorem-eligible ranks; skipped theorem plots after "
                    "persisting the whitening diagnostics."
                )
            # plot_tildeV_scatter_pretty(all_results, save_path=f"figures/pred_vs_obs/tildeV_pred_vs_obs_epoch_{epoch}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--ckpt_dir", "-ckpt", type=str, required=True, help="Directory containing model checkpoints")
    parser.add_argument("--ssl_fit_fraction", type=float, default=0.5)
    parser.add_argument(
        "--max_whiten_mean_l2_error",
        type=float,
        default=0.05,
        help="Fixed absolute eligibility cutoff for empirical mean L2 error",
    )
    parser.add_argument(
        "--max_whiten_operator_error",
        type=float,
        default=0.10,
        help=(
            "Fixed absolute eligibility cutoff for empirical second-moment "
            "operator error"
        ),
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=".",
        help="Output root for durable spectral eligibility artifacts",
    )

    args = parser.parse_args()
    main(args)
