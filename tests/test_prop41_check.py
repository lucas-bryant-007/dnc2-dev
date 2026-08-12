"""Tests for the exact-whitening Prop 4.1 check.

The substantive test is `test_in_sample_whitening_restores_identity`: on data
built to satisfy the proposition, the identity must come out near-exact, and it
must still miss when the whitener is fitted out of sample. That contrast is the
reason the module exists, so if it ever stops holding the check is not measuring
what it claims to.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

import hyperrect as H  # noqa: E402
from prop41_check import prop41_identity  # noqa: E402

NAMES = ["a", "b", "c"]


def _synthetic(n: int = 6000, d: int = 48, gaps=(1.4, 1.0, 0.7), seed: int = 0):
    """Three independent balanced factors, each shifting the mean on its own axis."""
    generator = torch.Generator().manual_seed(seed)
    attrs = torch.randint(0, 2, (n, 3), generator=generator)
    signs = attrs.float() * 2 - 1
    features = torch.randn(n, d, generator=generator)
    for index, gap in enumerate(gaps):
        features[:, index] += signs[:, index] * gap
    rotation, _ = torch.linalg.qr(torch.randn(d, d, generator=generator))
    return features @ rotation, attrs


def test_whitening_is_exact_in_sample():
    features, attrs = _synthetic()
    diagnostics = prop41_identity(features, attrs, NAMES)["whitening_diagnostics"]
    assert diagnostics["second_moment_min_eigenvalue"] == pytest.approx(1.0, abs=1e-3)
    assert diagnostics["second_moment_max_eigenvalue"] == pytest.approx(1.0, abs=1e-3)


def test_plug_in_comparison_is_vacuous_on_pure_noise():
    """The trap this module exists to avoid.

    With exact whitening and balanced labels the plug-in Vtilde and the plug-in B
    carry the same finite-sample optimism, so their ratio satisfies the identity
    whatever the data. Here there is no signal at all and it still "passes" --
    which is why a small plug-in error is not evidence for the proposition.
    """
    generator = torch.Generator().manual_seed(11)
    features = torch.randn(6000, 64, generator=generator)
    attrs = torch.randint(0, 2, (6000, 3), generator=generator)
    result = prop41_identity(features, attrs, NAMES)
    errors = [a["relative_error_plug_in"] for a in result["attributes"]]
    assert max(errors) < 0.01, (
        "plug-in comparison is expected to pass even on noise; if this fails the "
        f"forced-identity account is wrong: {errors}"
    )


def test_crossfit_identity_holds_when_the_proposition_applies():
    features, attrs = _synthetic(n=12000, d=48)
    result = prop41_identity(features, attrs, NAMES)
    errors = [a["relative_error_crossfit"] for a in result["attributes"]]
    assert all(e is not None for e in errors)
    assert max(errors) < 0.15, f"both-sides-held-out identity should hold: {errors}"


def test_crossfit_check_can_fail():
    """A check that cannot fail is not a check.

    Scaling the measured spread away from its true value must show up as error,
    otherwise the comparison is forced the way the plug-in one is.
    """
    features, attrs = _synthetic(n=12000, d=48)
    result = prop41_identity(features, attrs, NAMES)
    record = result["attributes"][0]
    honest = record["relative_error_crossfit"]
    corrupted_v = record["directional_cdnv_crossfit"] * 2.0
    predicted = record["predicted_tilde_v_crossfit"]
    corrupted = abs(corrupted_v - predicted) / predicted
    assert corrupted > 0.5 > honest, (honest, corrupted)


def test_mixed_comparison_misses_as_expected():
    """Cross-fit B against plug-in Vtilde: debiased on one side only."""
    features, attrs = _synthetic(n=12000, d=512)
    result = prop41_identity(features, attrs, NAMES)
    mixed = [a["relative_error_mixed"] for a in result["attributes"]]
    crossfit = [a["relative_error_crossfit"] for a in result["attributes"]]
    assert max(mixed) > max(crossfit), (
        f"mixing a debiased B with a plug-in Vtilde should be worse than "
        f"debiasing both: mixed={mixed} crossfit={crossfit}"
    )


def test_out_of_sample_whitening_still_misses():
    """The contrast the module is built around: same data, frozen whitener, worse."""
    features, attrs = _synthetic()
    half = features.shape[0] // 2
    frozen = H.fit_rewhitener(features[half:])
    whitened = H.apply_rewhitener(features[:half], frozen)

    analysis = H.analyze(whitened, attrs[:half], NAMES, compute_capture=True,
                         viz_triple=NAMES, cos_ceiling=1.0)
    worst = 0.0
    for metric in analysis["metrics"]:
        capture, tilde_v = metric.get("capture_B"), metric.get("directional_cdnv")
        if capture is None or tilde_v is None or not 0.0 < capture < 1.0:
            continue
        predicted = (1.0 - capture) / (2.0 * capture)
        worst = max(worst, abs(tilde_v - predicted) / predicted)
    assert worst > 0.05, "out-of-sample whitening is expected to miss the identity"


def test_whitening_stays_exact_at_high_dimension_ratio():
    """Exactness comes from fitting in sample, so it should survive large D/N."""
    features, attrs = _synthetic(n=1200, d=400, seed=3)
    diagnostics = prop41_identity(features, attrs, NAMES)["whitening_diagnostics"]
    assert diagnostics["second_moment_min_eigenvalue"] == pytest.approx(1.0, abs=1e-3)
    assert diagnostics["second_moment_max_eigenvalue"] == pytest.approx(1.0, abs=1e-3)


def test_reports_balance_and_is_label_free():
    features, attrs = _synthetic(n=2400, d=32, seed=5)
    result = prop41_identity(features, attrs, NAMES, seed=11)
    assert result["whitening"]["exact_whiteness_claimed"] is True
    assert result["whitening"]["label_free"] is True
    assert result["balance"]["seed"] == 11
    assert result["balance"]["total_balanced_samples"] == 8 * result["balance"]["samples_per_cell"]
    assert [a["name"] for a in result["attributes"]] == NAMES


def test_rejects_malformed_input():
    features, attrs = _synthetic(n=800, d=16, seed=7)
    with pytest.raises(ValueError):
        prop41_identity(features[0], attrs, NAMES)
    with pytest.raises(ValueError):
        prop41_identity(features, attrs[:, :2], NAMES)
    with pytest.raises(ValueError):
        prop41_identity(features, attrs, NAMES[:2])
