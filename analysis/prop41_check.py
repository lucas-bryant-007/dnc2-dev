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


def prop41_identity(
    features: torch.Tensor,
    attr3: torch.Tensor,
    triple_names: Sequence[str],
    *,
    seed: int = 7,
    max_per_cell: int | None = None,
    rel_eig_threshold: float = 1e-3,
) -> Dict[str, Any]:
    """Compare measured Vtilde against (1-B)/(2B) under exact whitening.

    ``features`` is [N, D] raw (un-whitened) representations; ``attr3`` is the
    [N, 3] binary label matrix whose columns correspond to ``triple_names``.
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
    raw = features[selected]
    balanced_attrs = attr3[selected]

    # The whole point: fit on the analysed sample so whiteness is exact here.
    rewhitener = H.fit_rewhitener(raw, rel_eig_threshold=rel_eig_threshold)
    balanced = H.apply_rewhitener(raw, rewhitener)

    first, second = _half_split(per_cell, selected)
    crossfit = H.crossfit_probe_geometry(
        H.apply_rewhitener(features[first], rewhitener),
        attr3[first],
        H.apply_rewhitener(features[second], rewhitener),
        attr3[second],
        triple_names,
        task_selection_status="frozen_from_independent_training_split",
    )
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
        record: Dict[str, Any] = {
            "name": metric["name"],
            "pos_frac": metric.get("pos_frac"),
            "capture_B_plug_in": plug_in_b,
            "capture_B_crossfit": cross_b,
            "directional_cdnv": tilde_v,
        }
        for label, capture in (("plug_in", plug_in_b), ("crossfit", cross_b)):
            if capture is None or tilde_v is None or not (0.0 < capture < 1.0):
                record[f"predicted_tilde_v_{label}"] = None
                record[f"relative_error_{label}"] = None
                continue
            predicted = (1.0 - capture) / (2.0 * capture)
            record[f"predicted_tilde_v_{label}"] = predicted
            record[f"relative_error_{label}"] = (
                abs(tilde_v - predicted) / predicted if predicted else math.nan
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
            "input_dim": int(features.shape[1]),
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
