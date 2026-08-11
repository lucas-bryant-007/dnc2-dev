import pytest
import torch

from analysis.cdnv_conventions import (
    ORDERED_SINGLE_CLASS,
    ORIGINAL_HALF_SYMMETRIC,
    UNHALVED_SYMMETRIC,
)
from analysis.geometry import GeometricEvaluator


def test_geometry_reports_explicit_normalizations():
    features = torch.tensor([[-2.0, 0.0], [0.0, 0.0], [0.0, 0.0], [2.0, 0.0]])
    labels = torch.tensor([0, 0, 1, 1])
    evaluator = GeometricEvaluator(num_classes=2, device="cpu")

    original = evaluator.compute_cdnv(
        features,
        labels,
        normalization=ORIGINAL_HALF_SYMMETRIC,
    )
    unhalved = evaluator.compute_cdnv(
        features,
        labels,
        normalization=UNHALVED_SYMMETRIC,
    )
    assert unhalved == pytest.approx(2.0 * original)

    original_directional = evaluator.compute_directional_cdnv(
        features,
        labels,
        normalization=ORIGINAL_HALF_SYMMETRIC,
    )
    unhalved_directional = evaluator.compute_directional_cdnv(
        features,
        labels,
        normalization=UNHALVED_SYMMETRIC,
    )
    assert unhalved_directional == pytest.approx(2.0 * original_directional)

    pair = evaluator.compute_pairwise_metrics(features, labels)[(0, 1)]
    assert pair["Vij_normalization"] == UNHALVED_SYMMETRIC
    assert pair["Vtilde_ij_normalization"] == ORDERED_SINGLE_CLASS
    assert pair["Vij_unhalved_symmetric"] == pytest.approx(pair["Vij"])
    assert pair["Vij_original_half_symmetric"] == pytest.approx(0.5 * pair["Vij"])


def test_geometry_rejects_undefined_gaps_and_singleton_covariances():
    evaluator = GeometricEvaluator(num_classes=2, device="cpu")
    with pytest.raises(ValueError, match="nonpositive squared mean gap"):
        evaluator.compute_cdnv(
            torch.zeros(4, 2),
            torch.tensor([0, 0, 1, 1]),
        )
    with pytest.raises(ValueError, match="at least two rows"):
        evaluator.compute_pairwise_metrics(
            torch.tensor([[0.0], [1.0], [2.0]]),
            torch.tensor([0, 1, 1]),
        )
