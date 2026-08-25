from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from analysis.build_hyperrectangle_review import _pairwise_axis_values, _write_manifest


def test_pairwise_axis_values_deduplicate_directions_and_folds(tmp_path: Path) -> None:
    source = tmp_path / "geometry.csv"
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "target",
                "context",
                "valid",
                "target_context_abs_cosine",
            ),
        )
        writer.writeheader()
        writer.writerows(
            (
                {"target": "a", "context": "b", "valid": "True", "target_context_abs_cosine": 0.1},
                {"target": "b", "context": "a", "valid": "True", "target_context_abs_cosine": 0.3},
                {"target": "a", "context": "b", "valid": "False", "target_context_abs_cosine": 0.9},
                {"target": "c", "context": "d", "valid": "True", "target_context_abs_cosine": 0.8},
            )
        )

    values = np.sort(_pairwise_axis_values(source))
    assert values.tolist() == [0.2, 0.8]


def test_review_manifest_hashes_inputs_and_scientific_outputs(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    source.write_text("x\n1\n", encoding="utf-8")
    output = tmp_path / "output"
    output.mkdir()
    (output / "README.md").write_text("documentation\n", encoding="utf-8")
    (output / "figure.pdf").write_bytes(b"%PDF-review")

    _write_manifest(output, [source])

    with (output / "MANIFEST.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [(row["role"], Path(row["path"]).name) for row in rows] == [
        ("input", "source.csv"),
        ("output", "figure.pdf"),
    ]
    assert all(len(row["sha256"]) == 64 for row in rows)
