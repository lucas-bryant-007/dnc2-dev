from dataclasses import dataclass
from typing import Optional

import torch


ABSOLUTE_WHITENING_ELIGIBILITY_POLICY = (
    "absolute_empirical_geometry_per_rank_v1"
)


@dataclass
class ExactWhiteningFit:
    """Rank-truncated empirical PCA whitening fit."""

    mean: torch.Tensor
    whitener: torch.Tensor
    covariance_eigenvalues: torch.Tensor
    retained_eigenvalues: torch.Tensor
    rel_eig_threshold: float
    numerical_rel_eig_floor: float
    effective_rel_eig_threshold: float
    cutoff: float
    lambda_max: float
    input_dim: int
    output_dim: int
    n_fit: int


@dataclass
class WhiteningDiagnostics:
    """Empirical spectral approximation checks for centered/white coordinates."""

    n_samples: int
    n_independent_samples: int
    rows_per_independent_sample: float
    dimension: int
    dimension_to_independent_sample_ratio: float
    mean_sampling_reference_scale: float
    operator_sampling_reference_scale: float
    mean_l2_sampling_normalized_error: float
    second_moment_operator_sampling_normalized_error: float
    max_abs_mean: float
    mean_l2_norm: float
    second_moment_max_abs_error: float
    second_moment_fro_error: float
    second_moment_operator_error: float
    second_moment_min_eigenvalue: float
    second_moment_max_eigenvalue: float
    covariance_max_abs_error: float
    covariance_fro_error: float

    def as_dict(self) -> dict:
        return {
            "n_samples": self.n_samples,
            "n_independent_samples": self.n_independent_samples,
            "rows_per_independent_sample": self.rows_per_independent_sample,
            "dimension": self.dimension,
            "dimension_to_independent_sample_ratio": (
                self.dimension_to_independent_sample_ratio
            ),
            "mean_sampling_reference_scale": self.mean_sampling_reference_scale,
            "operator_sampling_reference_scale": (
                self.operator_sampling_reference_scale
            ),
            "mean_l2_sampling_normalized_error": (
                self.mean_l2_sampling_normalized_error
            ),
            "second_moment_operator_sampling_normalized_error": (
                self.second_moment_operator_sampling_normalized_error
            ),
            "max_abs_mean": self.max_abs_mean,
            "mean_l2_norm": self.mean_l2_norm,
            "second_moment_max_abs_error": self.second_moment_max_abs_error,
            "second_moment_fro_error": self.second_moment_fro_error,
            "second_moment_operator_error": self.second_moment_operator_error,
            "second_moment_min_eigenvalue": (
                self.second_moment_min_eigenvalue
            ),
            "second_moment_max_eigenvalue": (
                self.second_moment_max_eigenvalue
            ),
            "covariance_max_abs_error": self.covariance_max_abs_error,
            "covariance_fro_error": self.covariance_fro_error,
        }

    def satisfies_absolute_cutoffs(
        self,
        max_mean_l2_error: float,
        max_operator_error: float,
    ) -> bool:
        """Apply fixed absolute empirical-geometry cutoffs.

        These cutoffs deliberately do not adapt to dimension or sample size and
        must not be interpreted as a test that the population is non-white.
        """
        return (
            self.mean_l2_norm <= max_mean_l2_error
            and self.second_moment_operator_error <= max_operator_error
        )


def absolute_whitening_eligibility(
    diagnostic: WhiteningDiagnostics,
    max_mean_l2_error: float,
    max_operator_error: float,
) -> dict:
    """Return a JSON-safe per-rank absolute eligibility decision.

    Sampling-normalized errors are descriptive reference ratios only. They use
    the standard-normal scales ``sqrt(d/n)`` for the sample mean and
    ``2*sqrt(d/n) + d/n`` for the sample second-moment operator error. Paired
    views can be dependent, so ``n`` is the supplied latent-instance count.
    No Gaussianity or independence claim is made, and these ratios do not
    affect pass/fail.
    """
    if max_mean_l2_error <= 0 or max_operator_error <= 0:
        raise ValueError("absolute whitening eligibility cutoffs must be positive")
    reasons = []
    if diagnostic.mean_l2_norm > max_mean_l2_error:
        reasons.append("mean_l2_exceeds_absolute_cutoff")
    if diagnostic.second_moment_operator_error > max_operator_error:
        reasons.append("second_moment_operator_error_exceeds_absolute_cutoff")
    return {
        "eligibility_policy": ABSOLUTE_WHITENING_ELIGIBILITY_POLICY,
        "policy_meaning": (
            "strict_absolute_empirical_geometry_not_population_compatibility_test"
        ),
        "eligible": not reasons,
        "ineligibility_reasons": reasons,
        "max_mean_l2_error": float(max_mean_l2_error),
        "max_operator_error": float(max_operator_error),
        "sampling_normalization_role": "diagnostic_only_not_decision_rule",
        "sampling_reference_model": (
            "isotropic_gaussian_scale_with_conservative_latent_instance_count"
        ),
    }


@dataclass
class BalancedPairedPopulationSplit:
    """Instance-grouped balanced populations for whitening and evaluation."""

    fit_view1: torch.Tensor
    fit_view2: torch.Tensor
    fit_labels: torch.Tensor
    fit_instance_ids: torch.Tensor
    eval_view1: torch.Tensor
    eval_view2: torch.Tensor
    eval_labels: torch.Tensor
    eval_instance_ids: torch.Tensor
    class_values: tuple
    label_mapping: tuple
    original_class_counts: tuple
    balanced_instances_per_class: int
    fit_instances_per_class: int
    eval_instances_per_class: int
    fit_fraction: float
    seed: int
    instance_id_source: str

    @property
    def evaluation_features(self) -> torch.Tensor:
        """Both held-out views, for estimating the single-view marginal."""
        return torch.cat((self.eval_view1, self.eval_view2), dim=0)

    @property
    def evaluation_labels(self) -> torch.Tensor:
        return torch.cat((self.eval_labels, self.eval_labels), dim=0)

    @property
    def evaluation_instance_ids(self) -> torch.Tensor:
        """Repeated IDs expose the two rows belonging to each latent instance."""
        return torch.cat((self.eval_instance_ids, self.eval_instance_ids), dim=0)

    def metadata(self) -> dict:
        return {
            "policy": "equal_binary_class_downsampling_before_fit_eval_split",
            "class_values": list(self.class_values),
            "label_mapping": [
                {"observed": observed, "canonical": canonical}
                for observed, canonical in self.label_mapping
            ],
            "original_instances_per_class": list(self.original_class_counts),
            "balanced_instances_per_class": self.balanced_instances_per_class,
            "fit_instances_per_class": self.fit_instances_per_class,
            "eval_instances_per_class": self.eval_instances_per_class,
            "fit_total_instances": int(self.fit_instance_ids.numel()),
            "eval_total_instances": int(self.eval_instance_ids.numel()),
            "views_per_instance": 2,
            "views_grouped_during_split": True,
            "fit_eval_instance_disjoint": True,
            "instance_id_source": self.instance_id_source,
            "fit_fraction": self.fit_fraction,
            "seed": self.seed,
        }


def canonicalize_binary_labels(
    labels: torch.Tensor,
    context: str = "binary labels",
) -> tuple[torch.Tensor, dict]:
    """Map the two observed class values deterministically to {-1,+1}."""
    labels = labels.reshape(-1)
    if labels.numel() < 2:
        raise ValueError(f"{context} require at least two labels")
    if labels.is_floating_point() and not bool(torch.isfinite(labels).all().item()):
        raise ValueError(f"{context} must contain only finite values")
    values, counts = torch.unique(labels.detach().cpu(), return_counts=True)
    if values.numel() != 2:
        raise ValueError(
            f"{context} must contain exactly two observed class values; "
            f"got {values.tolist()}"
        )
    class_values = [value.item() for value in values]
    class_counts = [int(count) for count in counts]
    canonical = torch.where(
        labels == class_values[1],
        torch.ones((), dtype=torch.float32, device=labels.device),
        -torch.ones((), dtype=torch.float32, device=labels.device),
    )
    diagnostics = {
        "n_samples": int(labels.numel()),
        "class_values": class_values,
        "class_counts": class_counts,
        "canonical_mapping": [
            {"observed": class_values[0], "canonical": -1},
            {"observed": class_values[1], "canonical": 1},
        ],
        "is_binary": True,
        "is_exactly_balanced": class_counts[0] == class_counts[1],
    }
    return canonical, diagnostics


def binary_balance_diagnostics(labels: torch.Tensor) -> dict:
    """Return exact empirical balance and encoding diagnostics."""
    _, diagnostics = canonicalize_binary_labels(labels)
    return diagnostics


def canonicalize_balanced_binary_labels(
    labels: torch.Tensor,
    context: str,
) -> tuple[torch.Tensor, dict]:
    """Canonicalize labels and enforce the paper's equal class probabilities."""
    canonical, diagnostics = canonicalize_binary_labels(labels, context)
    if not diagnostics["is_exactly_balanced"]:
        raise ValueError(
            f"{context} must contain exactly two equally represented classes; "
            f"diagnostics={diagnostics}"
        )
    return canonical, diagnostics


def require_balanced_binary_labels(labels: torch.Tensor, context: str) -> dict:
    """Reject theorem populations that violate the paper's balanced-label law."""
    _, diagnostics = canonicalize_balanced_binary_labels(labels, context)
    return diagnostics


def split_balanced_paired_fit_eval(
    z1: torch.Tensor,
    z2: torch.Tensor,
    labels: torch.Tensor,
    fit_fraction: float,
    seed: int,
    instance_ids: Optional[torch.Tensor] = None,
) -> BalancedPairedPopulationSplit:
    """Balance binary classes, then split grouped paired-view instances.

    The majority class is deterministically downsampled to the minority count
    before splitting. Both views of a latent instance remain in one fold, and
    each class contributes the same number of instances to both the whitening
    fit and theorem-evaluation populations.
    """
    if z1.shape != z2.shape or z1.shape[0] != labels.shape[0]:
        raise ValueError("paired features and labels must have matching sample counts")
    if z1.device != z2.device:
        raise ValueError("paired feature views must be on the same device")
    if not 0.0 < fit_fraction < 1.0:
        raise ValueError(f"fit_fraction must lie in (0,1), got {fit_fraction}")
    labels = labels.reshape(-1)
    canonical_labels, encoding = canonicalize_binary_labels(
        labels,
        "paired population labels",
    )
    if instance_ids is None:
        instance_ids = torch.arange(labels.numel(), device=labels.device)
        instance_id_source = "paired_extraction_row_index"
    else:
        instance_ids = instance_ids.reshape(-1)
        instance_id_source = "caller_supplied"
        if instance_ids.numel() != labels.numel():
            raise ValueError("instance_ids must contain one ID per paired instance")
    if torch.unique(instance_ids.detach().cpu()).numel() != labels.numel():
        raise ValueError("instance_ids must be unique before view duplication")

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    fit_parts = []
    eval_parts = []
    labels_cpu = labels.detach().cpu()
    class_values = torch.unique(labels_cpu)
    class_indices = [torch.where(labels_cpu == value)[0] for value in class_values]
    original_class_counts = tuple(int(indices.numel()) for indices in class_indices)
    balanced_instances_per_class = min(original_class_counts)
    if balanced_instances_per_class < 2:
        raise ValueError(
            "Every class needs at least two latent instances for independent "
            "whitening-fit and theorem-evaluation folds"
        )
    n_fit = min(
        max(1, int(round(fit_fraction * balanced_instances_per_class))),
        balanced_instances_per_class - 1,
    )
    for indices in class_indices:
        indices = indices[torch.randperm(indices.numel(), generator=generator)]
        indices = indices[:balanced_instances_per_class]
        fit_parts.append(indices[:n_fit])
        eval_parts.append(indices[n_fit:])
    fit_indices_cpu = torch.cat(fit_parts)
    eval_indices_cpu = torch.cat(eval_parts)
    fit_indices_cpu = fit_indices_cpu[
        torch.randperm(fit_indices_cpu.numel(), generator=generator)
    ]
    eval_indices_cpu = eval_indices_cpu[
        torch.randperm(eval_indices_cpu.numel(), generator=generator)
    ]
    fit_indices = fit_indices_cpu.to(z1.device)
    eval_indices = eval_indices_cpu.to(z1.device)
    label_fit_indices = fit_indices_cpu.to(labels.device)
    label_eval_indices = eval_indices_cpu.to(labels.device)
    id_fit_indices = fit_indices_cpu.to(instance_ids.device)
    id_eval_indices = eval_indices_cpu.to(instance_ids.device)
    return BalancedPairedPopulationSplit(
        fit_view1=z1[fit_indices],
        fit_view2=z2[fit_indices],
        fit_labels=canonical_labels[label_fit_indices],
        fit_instance_ids=instance_ids[id_fit_indices],
        eval_view1=z1[eval_indices],
        eval_view2=z2[eval_indices],
        eval_labels=canonical_labels[label_eval_indices],
        eval_instance_ids=instance_ids[id_eval_indices],
        class_values=tuple(value.item() for value in class_values),
        label_mapping=tuple(
            (row["observed"], row["canonical"])
            for row in encoding["canonical_mapping"]
        ),
        original_class_counts=original_class_counts,
        balanced_instances_per_class=balanced_instances_per_class,
        fit_instances_per_class=n_fit,
        eval_instances_per_class=balanced_instances_per_class - n_fit,
        fit_fraction=float(fit_fraction),
        seed=int(seed),
        instance_id_source=instance_id_source,
    )


def whitening_diagnostics(
    features: torch.Tensor,
    n_independent_samples: Optional[int] = None,
) -> WhiteningDiagnostics:
    """Measure empirical spectral proximity to the paper's population ideal.

    This is an approximation diagnostic, not proof that the exact population
    assumptions hold. For clustered augmented views, ``n_independent_samples``
    must count latent instances rather than view rows.
    """
    if features.ndim != 2 or features.shape[0] < 2 or features.shape[1] < 1:
        raise ValueError("features must have shape [N,D] with N >= 2 and D >= 1")
    if not bool(torch.all(torch.isfinite(features)).item()):
        raise ValueError("features must contain only finite values")
    if n_independent_samples is None:
        n_independent_samples = int(features.shape[0])
    if not 1 <= int(n_independent_samples) <= features.shape[0]:
        raise ValueError(
            "n_independent_samples must lie between 1 and the number of rows"
        )

    mean = features.mean(dim=0)
    second_moment = (features.T @ features) / features.shape[0]
    centered = features - mean
    covariance = (centered.T @ centered) / features.shape[0]
    identity = torch.eye(
        features.shape[1],
        dtype=features.dtype,
        device=features.device,
    )
    moment_error = second_moment - identity
    covariance_error = covariance - identity
    second_moment_eigenvalues = torch.linalg.eigvalsh(second_moment)
    operator_error = torch.linalg.eigvalsh(moment_error).abs().max()
    dimension_sample_ratio = float(features.shape[1]) / int(
        n_independent_samples
    )
    mean_reference_scale = dimension_sample_ratio ** 0.5
    operator_reference_scale = (
        2.0 * mean_reference_scale + dimension_sample_ratio
    )
    mean_l2_norm = float(mean.norm().item())
    second_moment_operator_error = float(operator_error.item())
    return WhiteningDiagnostics(
        n_samples=int(features.shape[0]),
        n_independent_samples=int(n_independent_samples),
        rows_per_independent_sample=(
            float(features.shape[0]) / int(n_independent_samples)
        ),
        dimension=int(features.shape[1]),
        dimension_to_independent_sample_ratio=dimension_sample_ratio,
        mean_sampling_reference_scale=mean_reference_scale,
        operator_sampling_reference_scale=operator_reference_scale,
        mean_l2_sampling_normalized_error=(
            mean_l2_norm / mean_reference_scale
        ),
        second_moment_operator_sampling_normalized_error=(
            second_moment_operator_error / operator_reference_scale
        ),
        max_abs_mean=float(mean.abs().max().item()),
        mean_l2_norm=mean_l2_norm,
        second_moment_max_abs_error=float(moment_error.abs().max().item()),
        second_moment_fro_error=float(moment_error.norm().item()),
        second_moment_operator_error=second_moment_operator_error,
        second_moment_min_eigenvalue=float(
            second_moment_eigenvalues.min().item()
        ),
        second_moment_max_eigenvalue=float(
            second_moment_eigenvalues.max().item()
        ),
        covariance_max_abs_error=float(covariance_error.abs().max().item()),
        covariance_fro_error=float(covariance_error.norm().item()),
    )


def fit_exact_whitener(
    features: torch.Tensor,
    rel_eig_threshold: float = 1e-3,
    k_cap: Optional[int] = None,
) -> ExactWhiteningFit:
    """Fit a map whose retained coordinates have empirical covariance identity.

    Directions below the larger of the requested relative cutoff and a
    dtype/dimension-aware numerical-rank floor are removed rather than
    ridge-shrunk. On the fitting population, ``(features - mean) @ whitener``
    therefore has covariance ``I_output_dim`` up to eigensolver roundoff.

    Float64 inputs remain float64. Lower-precision and non-floating inputs are
    promoted to float32 so fitting and later transforms share one safe dtype.

    PCA-whitened and rank-truncated ZCA-whitened coordinates differ only by an
    isometry when they use the same retained eigenspace and exact scaling.
    Changing the retained rank or replacing ridge scaling with exact inverse
    square-root scaling is not an isometry and can change distances, capture,
    and downstream task geometry.
    """
    if features.ndim != 2:
        raise ValueError(f"features must be 2D, got {features.ndim}D")
    if features.shape[0] < 2:
        raise ValueError("At least two samples are required to fit whitening")
    if features.shape[1] < 1:
        raise ValueError("features must contain at least one coordinate")
    if not 0.0 < rel_eig_threshold <= 1.0:
        raise ValueError(
            "rel_eig_threshold must lie in (0, 1], "
            f"got {rel_eig_threshold}"
        )
    if k_cap is not None and int(k_cap) < 1:
        raise ValueError(f"k_cap must be positive when supplied, got {k_cap}")
    if not bool(torch.all(torch.isfinite(features)).item()):
        raise ValueError("features must contain only finite values")

    compute_dtype = (
        torch.float64 if features.dtype == torch.float64 else torch.float32
    )
    features = features.to(dtype=compute_dtype)
    mean = features.mean(dim=0, keepdim=True)
    centered = features - mean
    covariance = (centered.T @ centered) / centered.shape[0]
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order].clamp_min(0.0)
    eigenvectors = eigenvectors[:, order]

    lambda_max = float(eigenvalues[0].item())
    if not torch.isfinite(eigenvalues[0]) or lambda_max <= 0.0:
        raise ValueError("Cannot whiten a zero-variance feature population")
    numerical_rel_eig_floor = float(
        torch.finfo(compute_dtype).eps * features.shape[1]
    )
    effective_rel_eig_threshold = max(
        float(rel_eig_threshold),
        numerical_rel_eig_floor,
    )
    cutoff = effective_rel_eig_threshold * lambda_max
    retained = eigenvalues >= cutoff
    output_dim = int(retained.sum().item())
    if k_cap is not None:
        output_dim = min(output_dim, int(k_cap))
    if output_dim < 1:
        raise ValueError(
            "No covariance directions survived the whitening cutoff "
            f"{cutoff:.6e}"
        )

    retained_eigenvalues = eigenvalues[:output_dim]
    retained_eigenvectors = eigenvectors[:, :output_dim]
    whitener = retained_eigenvectors * retained_eigenvalues.rsqrt()
    return ExactWhiteningFit(
        mean=mean,
        whitener=whitener,
        covariance_eigenvalues=eigenvalues,
        retained_eigenvalues=retained_eigenvalues,
        rel_eig_threshold=float(rel_eig_threshold),
        numerical_rel_eig_floor=numerical_rel_eig_floor,
        effective_rel_eig_threshold=effective_rel_eig_threshold,
        cutoff=cutoff,
        lambda_max=lambda_max,
        input_dim=int(features.shape[1]),
        output_dim=output_dim,
        n_fit=int(features.shape[0]),
    )
