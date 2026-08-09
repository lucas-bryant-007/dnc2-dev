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
    from .br.whitening import canonicalize_binary_labels
except ImportError:  # direct script execution from analysis/
    from br.whitening import canonicalize_binary_labels


def _validate_B(B: float, eps: float = 1e-12) -> float:
    B = float(B)
    if not math.isfinite(B) or B < 0.0 or B > 1.0 + eps:
        raise ValueError(f"B must lie in [0, 1], got {B}")
    return min(B, 1.0)


# ---------------------------------------------------------------------------
# Prop 4.1: directional-collapse law
# ---------------------------------------------------------------------------
def directional_cdnv_from_B(B: float, eps: float = 1e-12) -> float:
    """V~_F = (1 - B)/(2B); +inf at B = 0."""
    B = _validate_B(B, eps)
    if B <= eps:
        return float("inf")
    return (1.0 - B) / (2.0 * B)


def cdnv_from_B(B: float, r: int, eps: float = 1e-12) -> float:
    """V_F = (r - B)/(2B); +inf at B = 0."""
    B = _validate_B(B, eps)
    if r < 1:
        raise ValueError(f"r must be positive, got {r}")
    if B <= eps:
        return float("inf")
    return (float(r) - B) / (2.0 * B)


# ---------------------------------------------------------------------------
# Thm 4.5: direct few-shot NCC bound via captured posterior energy
# ---------------------------------------------------------------------------
def nccc_error_bound(B: float, r: int, m: int, clamp: bool = True) -> float:
    """m-shot NCC error bound err_NCC_m(F) <= (1-B) + (r-B)/m + (1-B)/(1-B+2mB).

    The leading (1 - B) is the uncaptured posterior energy (irreducible floor);
    the rest are finite-sample centroid-estimation terms vanishing as m -> inf.
    B = 0 gives a vacuous bound (+inf, or 1.0 if ``clamp``).
    """
    B = _validate_B(B)
    if r < 1 or m < 1:
        raise ValueError(f"r and m must be positive, got r={r}, m={m}")
    if B <= 0.0:
        return 1.0 if clamp else float("inf")
    val = (1.0 - B) + (r - B) / m + (1.0 - B) / (1.0 - B + 2.0 * m * B)
    return min(val, 1.0) if clamp else val


def nccc_error_bound_from_tilde_v(tilde_v: float, r: int, m: int, clamp: bool = True) -> float:
    """Thm 4.5 in directional-CDNV form (algebraically identical to nccc_error_bound):

        2V~/(1+2V~) + ((r-1)+2rV~)/(m(1+2V~)) + V~/(V~+m).
    """
    tv = float(tilde_v)
    if tv < 0 or r < 1 or m < 1:
        raise ValueError(
            f"tilde_v must be nonnegative and r,m positive; got {tv}, {r}, {m}"
        )
    if not math.isfinite(tv):
        return 1.0 if clamp else float("inf")
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
    """Backward-compatible alias; values are half-sides, not full side lengths."""
    return hyperrectangle_half_side_lengths(Bs)


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
# Few-Shot Transfer in SSL" (Luthra, Salunkhe, Galanti 2026). These act on the
# *raw* frozen features and are the curves in that paper's Figure 3:
#   - "Our bound"   = Thm 4.1 (optimized) / Thm C.2 (lambda=1)
#   - "Lim bound"   = m->inf limit = 4 * Vtilde
#   - "Luthra 2025" = the prior bound (Luthra et al. 2025b), a=16 instance
# Each pairwise dict entry has keys d2, vi, vj, Vij, Vtilde_ij, Theta_ij
# (exactly what geometry.GeometricEvaluator.compute_pairwise_metrics returns).
# ---------------------------------------------------------------------------
def _E_terms(V: float, Theta: float, m: int):
    """E1, E2, E3 from Prop C.1 (the finite-shot leakage / tail terms)."""
    E1 = (4.0 / m) * (V * V + 0.25 * V)
    E2 = V / m
    E3 = (Theta + 2.0 * (m - 1) * V * V) / (m ** 3)
    return E1, E2, E3


def _imbalance_denom(vi: float, vj: float, dij2: float, m: int) -> float:
    """(1 + (v_j - v_i)/(m d_ij^2))^2 variance-imbalance factor."""
    return (1.0 + (vj - vi) / (m * dij2)) ** 2


def nccc_pair_thm41(Vt, V, Theta, vi, vj, dij2, m) -> float:
    """Pairwise NCC bound, Thm 4.1 form: 4*Vt + (sqrt E1 + sqrt E2 + sqrt E3)^2."""
    E1, E2, E3 = _E_terms(V, Theta, m)
    num = 4.0 * Vt + (math.sqrt(E1) + math.sqrt(E2) + math.sqrt(E3)) ** 2
    return num / _imbalance_denom(vi, vj, dij2, m)


def nccc_pair_c2(Vt, V, Theta, vi, vj, dij2, m) -> float:
    """Pairwise NCC bound, Thm C.2 form (lambda=1): 4*Vt + 3*(E1+E2+E3).

    By Cauchy-Schwarz (sqrt E1+sqrt E2+sqrt E3)^2 <= 3(E1+E2+E3), so the Thm 4.1
    bound is always <= this one.
    """
    E1, E2, E3 = _E_terms(V, Theta, m)
    num = 4.0 * Vt + 3.0 * (E1 + E2 + E3)
    return num / _imbalance_denom(vi, vj, dij2, m)


def _pairwise_classes(pairwise: dict):
    cs = set()
    for (i, j) in pairwise:
        cs.add(i); cs.add(j)
    return sorted(cs)


def directional_nccc_bound(pairwise: dict, m: int, kind: str = "thm41") -> float:
    """Multiclass-averaged directional bound: (1/C') sum_i sum_{j!=i} pair(i,j).

    kind in {"thm41", "c2", "lim"} (lim = m->inf limit, 4*Vtilde, m-independent).
    """
    classes = _pairwise_classes(pairwise)
    Cp = len(classes)
    if kind == "lim":
        return sum(4.0 * p["Vtilde_ij"] for p in pairwise.values()) / Cp
    fn = {"thm41": nccc_pair_thm41, "c2": nccc_pair_c2}[kind]
    return sum(
        fn(p["Vtilde_ij"], p["Vij"], p["Theta_ij"], p["vi"], p["vj"], p["d2"], m)
        for p in pairwise.values()
    ) / Cp


def luthra2025_nccc_bound(pairwise: dict, m: int) -> float:
    """Prior bound (Luthra et al. 2025b), a=16 instance:

        (C'-1) [ 8 Vtilde_f + (sqrt8/m) Vs_f + (sqrt8/m + 4/m) V_f ],
    with Vtilde_f, V_f, Vs_f = averaged directional CDNV, CDNV, sqrt(CDNV).
    """
    classes = _pairwise_classes(pairwise)
    Cp = len(classes)
    pairs = list(pairwise.values())
    n = len(pairs)
    Vt_f = sum(p["Vtilde_ij"] for p in pairs) / n
    V_f = sum(p["Vij"] for p in pairs) / n
    Vs_f = sum(math.sqrt(max(p["Vij"], 0.0)) for p in pairs) / n
    s8 = math.sqrt(8.0)
    return (Cp - 1) * (8.0 * Vt_f + (s8 / m) * Vs_f + (s8 / m + 4.0 / m) * V_f)


@torch.no_grad()
def directional_fewshot_curves(
    features: torch.Tensor, labels: torch.Tensor, pairwise: dict,
    m_values, n_trials: int = 100, seed: int = 0, max_query: int = 5000,
) -> dict:
    """Figure-3 curves: empirical NCC error + Our/Lim/Luthra bounds, per shot m.

    ``pairwise`` is precomputed (e.g. GeometricEvaluator.compute_pairwise_metrics)
    on the same ``features`` so the bound geometry and empirical error match.
    """
    out = {}
    for m in m_values:
        m = int(m)
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
            "our_thm41": directional_nccc_bound(pairwise, m, "thm41"),
            "our_c2": directional_nccc_bound(pairwise, m, "c2"),
            "lim": directional_nccc_bound(pairwise, m, "lim"),
            "luthra2025": luthra2025_nccc_bound(pairwise, m),
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
    of psi[:, :r]). Returns {r: {"B": B, "curves": {m: {...}}}}.
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
        curves = {}
        for m in m_values:
            m = int(m)
            class_counts = sampling["class_instance_counts"]
            query_negative = class_counts["-1"] - m
            query_positive = class_counts["+1"] - m
            if max_query is not None:
                query_negative = min(query_negative, max_query)
                query_positive = min(query_positive, max_query)
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
                "thm45_B": nccc_error_bound(B, r, m),                  # NEW (draft)
                "thm41_dir": directional_nccc_bound(pw, m, "thm41"),   # OLD (published)
                "luthra2025": luthra2025_nccc_bound(pw, m),            # oldest
                "lim": directional_nccc_bound(pw, m, "lim"),           # 4*Vtilde
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
    Returns {r: {"B": B, "empirical": {m: err}, "bound": {m: err}}}.
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
        bnd = {int(m): nccc_error_bound(B, r, int(m)) for m in m_values}
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
            "empirical_sampling": sampling,
            "empirical_group_counts": group_counts,
        }
    return out
