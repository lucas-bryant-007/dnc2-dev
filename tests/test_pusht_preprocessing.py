import numpy as np

from analysis.pusht.pusht_common import standardize_rows


def test_input_standardization_uses_training_rows_only():
    rows = {
        "e_t": np.array([[0.0], [2.0], [1000.0]], dtype=np.float32),
        "act": np.zeros((3, 1), dtype=np.float32),
    }
    standardized, stats = standardize_rows(rows, np.array([0, 1]))
    assert stats["input_mean"].item() == 1.0
    assert stats["input_std"].item() == 1.0
    assert standardized["e_t"][:, 0].tolist() == [-1.0, 1.0, 999.0]
