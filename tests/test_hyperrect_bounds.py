import json

import numpy as np
import pytest

from analysis.hyperrect_bounds import load


def _payload():
    return {
        "attributes": ["a", "b", "c", "d", "e"],
        "triple_names": ["d", "b", "e"],
        "metrics": [
            {"name": name, "capture_B": 0.1 * (index + 1)}
            for index, name in enumerate(["a", "b", "c", "d", "e"])
        ],
        "cosine_matrix": [
            [1.0, 0.01, 0.02, 0.03, 0.04],
            [0.01, 1.0, 0.12, 0.13, 0.14],
            [0.02, 0.12, 1.0, 0.23, 0.24],
            [0.03, 0.13, 0.23, 1.0, 0.34],
            [0.04, 0.14, 0.24, 0.34, 1.0],
        ],
        "box": [],
    }


def test_load_indexes_cosines_by_triple_names(tmp_path):
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")

    _data, names, capture, cosine, centers = load(path)

    assert names == ["d", "b", "e"]
    np.testing.assert_allclose(capture, [0.4, 0.2, 0.5])
    np.testing.assert_allclose(
        cosine,
        [
            [1.0, 0.13, 0.34],
            [0.13, 1.0, 0.14],
            [0.34, 0.14, 1.0],
        ],
    )
    assert centers == {}


def test_load_rejects_ambiguous_nontriple_matrix_without_attribute_order(tmp_path):
    payload = _payload()
    payload.pop("attributes")
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="needs 'attributes'"):
        load(path)
