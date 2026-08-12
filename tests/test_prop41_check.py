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


def test_in_sample_whitening_restores_identity():
    features, attrs = _synthetic()
    result = prop41_identity(features, attrs, NAMES)

    diagnostics = result["whitening_diagnostics"]
    assert diagnostics["second_moment_min_eigenvalue"] == pytest.approx(1.0, abs=1e-3)
    assert diagnostics["second_moment_max_eigenvalue"] == pytest.approx(1.0, abs=1e-3)

    errors = [a["relative_error_plug_in"] for a in result["attributes"]]
    assert all(e is not None for e in errors)
    assert max(errors) < 0.05, f"identity should be near-exact under whitening, got {errors}"


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


def test_identity_holds_at_high_dimension_ratio():
    """Exactness comes from fitting in sample, so it should survive large D/N."""
    features, attrs = _synthetic(n=1200, d=400, seed=3)
    result = prop41_identity(features, attrs, NAMES)
    errors = [a["relative_error_plug_in"] for a in result["attributes"]]
    assert max(errors) < 0.05, errors


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
