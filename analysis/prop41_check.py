"""Prop 4.1 identity check under exact (in-sample) whitening.

Prop 4.1 states ``Vtilde_t = (1 - B_t) / (2 B_t)`` for a representation that is
centered *and whitened*, with a balanced binary task. The held-out cube protocol
deliberately freezes a train-fitted whitener and applies it out of sample, which
is the right choice for a prediction claim but leaves the evaluation fold only
approximately white -- second-moment eigenvalues on the shipped runs span
0.08-4.88 rather than sitting at 1, and the identity misses by 26-157%.

This module evaluates the identity in the regime the proposition actually
assumes: the whitener is fitted on the sample being analysed, so whiteness holds
by construction at any D/N. Whitening is label-free, so fitting it in sample is
not label leakage; capture is still estimated cross-fit across two halves so the
D/N plug-in bias stays controlled.

Kept separate from the cube on purpose. The cube's claim is that *train-only*
geometry predicts held-out corners, which is stronger than the proposition needs
and would be weakened by re-whitening. Both can be produced from one run.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Sequence

import torch

import hyperrect as H


def _half_split(per_cell: int, selected: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a balanced selection into two halves that stay balanced per cell."""
    view = selected.view(8, per_cell)
    cut = per_cell // 2
    return view[:, :cut].reshape(-1), view[:, cut: 2 * cut].reshape(-1)


def _one_sided_tilde_v(
    fit: torch.Tensor, fit_y: torch.Tensor, evaluate: torch.Tensor, evaluate_y: torch.Tensor
) -> tuple[float, float]:
    """Within-class spread on one fold, along the axis fitted on the other.

    Returns ``(numerator, delta_fit)`` where the numerator is the summed
    within-class variance measured on ``evaluate`` along the unit axis taken from
    ``fit``. Keeping the axis independent of the data it is scored on is what
    removes the optimism in the plug-in estimate.
    """
    delta_fit = fit[fit_y == 1].mean(0) - fit[fit_y == 0].mean(0)
    norm = torch.linalg.vector_norm(delta_fit)
    if float(norm) <= 1e-12:
        return math.nan, delta_fit
    axis = delta_fit / norm
    projected = evaluate @ axis
    numerator = (
        projected[evaluate_y == 1].var(unbiased=True)
        + projected[evaluate_y == 0].var(unbiased=True)
    )
    return float(numerator), delta_fit


def crossfit_tilde_v(
    features_a: torch.Tensor,
    labels_a: torch.Tensor,
    features_b: torch.Tensor,
    labels_b: torch.Tensor,
) -> float:
    """Directional CDNV with both numerator and denominator held out.

    The plug-in estimate reuses one sample for the class means, the axis and the
    spread, which inflates ``||mu+ - mu-||^2`` by roughly tr(Cov)/n and so
    deflates Vtilde. That is the same D/N inflation the cross-fit capture
    estimator removes from ``B``, in the opposite direction -- which is why
    comparing a cross-fit ``B`` against a plug-in ``Vtilde`` misses badly while
    comparing two plug-in quantities cannot miss at all.

    Here the squared gap is the cross term ``<delta_a, delta_b>``, unbiased for
    ``||delta||^2`` because the two fold estimates carry independent noise, and
    each fold's spread is measured along the other fold's axis. Symmetrized over
    which fold plays which role.
    """
    numerator_b, delta_a = _one_sided_tilde_v(features_a, labels_a, features_b, labels_b)
    numerator_a, delta_b = _one_sided_tilde_v(features_b, labels_b, features_a, labels_a)
    if math.isnan(numerator_a) or math.isnan(numerator_b):
        return math.nan
    denominator = float(torch.dot(delta_a, delta_b))
    if denominator <= 1e-12:
        # Independent estimates disagreeing on direction means no usable signal.
        return math.nan
    return 0.5 * (numerator_a + numerator_b) / denominator


def prop41_identity(
    features: torch.Tensor,
    attr3: torch.Tensor,
    triple_names: Sequence[str],
    *,
    seed: int = 7,
    max_per_cell: int | None = None,
    rel_eig_threshold: float = 1e-3,
    subspace_dim: int | None = None,
) -> Dict[str, Any]:
    """Compare measured Vtilde against (1-B)/(2B) under exact whitening.

    ``features`` is [N, D] raw (un-whitened) representations; ``attr3`` is the
    [N, 3] binary label matrix whose columns correspond to ``triple_names``.

    ``subspace_dim`` first projects onto that many leading principal directions
    of the balanced sample. This matters more than it looks: at D/N above roughly
    0.05 the cross-fit comparison has a noise floor large enough to swamp the
    quantity being measured. On synthetic data where the proposition holds
    exactly, N=6632 gives a floor of 1.3% at D=256 but 36.9% at D=2048, so a
    full-dimension result cannot distinguish "the identity fails" from "we cannot
    measure it". The projection is label-free, and the proposition applies to
    whatever representation it is handed, so testing it on the leading subspace
    is a valid test of a smaller representation.
    """
    if features.ndim != 2:
        raise ValueError(f"features must be 2D, got {features.ndim}D")
    if attr3.ndim != 2 or attr3.shape[1] != 3:
        raise ValueError(f"attr3 must be [N,3], got {tuple(attr3.shape)}")
    if len(triple_names) != 3:
        raise ValueError("triple_names must name exactly three attributes")

    selected, cell_counts, per_cell = H.balanced_joint_indices(
        attr3, seed=seed, max_per_cell=max_per_cell
    )
    balanced_attrs = attr3[selected]
    input_dim = int(features.shape[1])

    working = features.float()
    variance_kept = 1.0
    if subspace_dim is not None and 0 < subspace_dim < input_dim:
        # Label-free, fitted on the balanced sample, applied to every fold.
        centre = working[selected].mean(dim=0, keepdim=True)
        _u, singular, right = torch.linalg.svd(
            working[selected] - centre, full_matrices=False
        )
        spectrum = singular ** 2
        variance_kept = float(spectrum[:subspace_dim].sum() / spectrum.sum())
        working = (working - centre) @ right[:subspace_dim].T

    raw = working[selected]
    # The whole point: fit on the analysed sample so whiteness is exact here.
    rewhitener = H.fit_rewhitener(raw, rel_eig_threshold=rel_eig_threshold)
    balanced = H.apply_rewhitener(raw, rewhitener)

    first, second = _half_split(per_cell, selected)
    first_features = H.apply_rewhitener(working[first], rewhitener)
    second_features = H.apply_rewhitener(working[second], rewhitener)
    crossfit = H.crossfit_probe_geometry(
        first_features,
        attr3[first],
        second_features,
        attr3[second],
        triple_names,
        task_selection_status="frozen_from_independent_training_split",
    )
    crossfit_tilde = [
        crossfit_tilde_v(
            first_features, attr3[first][:, column],
            second_features, attr3[second][:, column],
        )
        for column in range(3)
    ]
    analysis = H.analyze(
        balanced,
        balanced_attrs,
        list(triple_names),
        compute_capture=True,
        viz_triple=list(triple_names),
        cos_ceiling=1.0,
    )

    crossfit_capture = crossfit.get("capture_B")
    if isinstance(crossfit_capture, dict):
        crossfit_capture = [crossfit_capture.get(name) for name in triple_names]

    attributes = []
    for index, metric in enumerate(analysis["metrics"]):
        plug_in_b = metric.get("capture_B")
        tilde_v = metric.get("directional_cdnv")
        cross_b = None
        if crossfit_capture is not None and index < len(crossfit_capture):
            cross_b = crossfit_capture[index]
        cross_v = crossfit_tilde[index] if index < len(crossfit_tilde) else None
        if cross_v is not None and math.isnan(cross_v):
            cross_v = None
        record: Dict[str, Any] = {
            "name": metric["name"],
            "pos_frac": metric.get("pos_frac"),
            "capture_B_plug_in": plug_in_b,
            "capture_B_crossfit": cross_b,
            "directional_cdnv": tilde_v,
            "directional_cdnv_crossfit": cross_v,
        }
        # Three comparisons, and only the last is informative:
        #   plug_in   both sides biased the same way -- structurally forced
        #   mixed     cross-fit B against plug-in Vtilde -- misses by construction
        #   crossfit  both sides held out -- the one that can actually fail
        for label, capture, measured in (
            ("plug_in", plug_in_b, tilde_v),
            ("mixed", cross_b, tilde_v),
            ("crossfit", cross_b, cross_v),
        ):
            if (
                capture is None
                or measured is None
                or not (0.0 < capture < 1.0)
            ):
                record[f"predicted_tilde_v_{label}"] = None
                record[f"relative_error_{label}"] = None
                continue
            predicted = (1.0 - capture) / (2.0 * capture)
            record[f"predicted_tilde_v_{label}"] = predicted
            record[f"relative_error_{label}"] = (
                abs(measured - predicted) / predicted if predicted else math.nan
            )
        attributes.append(record)

    diagnostics = H.whitening_diagnostics(balanced).as_dict()
    return {
        "estimand": "prop_4_1_identity_under_exact_in_sample_whitening",
        "whitening": {
            "scope": "fitted_on_the_analysed_balanced_sample",
            "exact_whiteness_claimed": True,
            "label_free": True,
            "rel_eig_threshold": rel_eig_threshold,
            "retained_dim": int(balanced.shape[1]),
            "input_dim": input_dim,
            "subspace_dim": subspace_dim,
            "variance_kept": variance_kept,
            "dimension_to_sample_ratio": int(balanced.shape[1]) / max(int(selected.numel()), 1),
        },
        "capture_estimator": "symmetrized split-half cross-Gram on the same whitened sample",
        "balance": {
            "seed": seed,
            "samples_per_cell": int(per_cell),
            "total_balanced_samples": int(selected.numel()),
            "original_cell_counts": [int(c) for c in cell_counts],
        },
        "triple_names": list(triple_names),
        "attributes": attributes,
        "whitening_diagnostics": diagnostics,
    }
