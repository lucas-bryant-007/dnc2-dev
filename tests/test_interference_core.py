import numpy as np

from analysis.interference_core import (
    coefficient_of_determination,
    evaluate_bottleneck_splits,
)


def test_r2_is_not_squared_correlation():
    target = np.array([-1.0, 0.0, 1.0])
    shifted = target + 10.0
    assert np.corrcoef(target, shifted)[0, 1] ** 2 == 1.0
    assert coefficient_of_determination(shifted, target) < 0


def test_shared_bottleneck_reports_split_uncertainty_and_train_whitening():
    rng = np.random.default_rng(4)
    features = rng.normal(size=(400, 6))
    targets = np.column_stack(
        [
            np.where(features[:, 0] > 0, 1.0, -1.0),
            np.where(features[:, 1] > 0, 1.0, -1.0),
        ]
    )
    result = evaluate_bottleneck_splits(
        features, targets, [1, 2], [0, 1, 2], train_frac=0.6
    )
    assert result["whitening_scope"] == "train_split_only"
    assert result["mean_bal_acc"].shape == (2,)
    assert result["per_task_bal_acc_sd"].shape == (2, 2)
    assert result["capacity"][1] >= result["capacity"][0]
    assert result["mean_bal_acc"][1] > 0.9
