from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from analysis import build_paper_release as release
from analysis import paper_figures_v2 as figures
from analysis import tg_style


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_mean_range_ignores_nonfinite_values() -> None:
    mean, low, high = tg_style.mean_range([0.2, float("nan"), 0.6, 0.4])
    assert mean == pytest.approx(0.4)
    assert low == pytest.approx(0.2)
    assert high == pytest.approx(0.6)


def test_save_writes_deterministic_native_pdf_and_png(tmp_path: Path) -> None:
    tg_style.apply_style()
    outputs = []
    for name in ("first", "second"):
        fig, ax = plt.subplots(figsize=(2.0, 1.5))
        ax.plot([0, 1], [0, 1])
        outputs.append(tg_style.save(fig, tmp_path / name))

    first_pdf, first_png = outputs[0]
    second_pdf, second_png = outputs[1]
    assert first_pdf.read_bytes().startswith(b"%PDF")
    assert first_png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert _hash(first_pdf) == _hash(second_pdf)
    assert _hash(first_png) == _hash(second_png)


def test_manifest_hashes_inputs_generated_data_and_outputs(tmp_path: Path) -> None:
    input_path = tmp_path / "values.csv"
    generated_path = tmp_path / "derived.json"
    output_path = tmp_path / "figure.pdf"
    input_path.write_text("x,y\n1,2\n", encoding="utf-8")
    generated_path.write_text('{"value": 2}\n', encoding="utf-8")
    output_path.write_bytes(b"%PDF-test")

    manifest = release._write_manifest(
        tmp_path / "manifest.csv",
        {"figure": [input_path]},
        [generated_path],
        [output_path],
    )
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert {row["role"] for row in rows} == {"input", "generated_data", "output"}
    assert {Path(row["path"]).name for row in rows} == {"values.csv", "derived.json", "figure.pdf"}
    assert all(len(row["sha256"]) == 64 for row in rows)


def test_failure_panel_reads_current_association_schema(tmp_path: Path) -> None:
    source = tmp_path / "geometry_associations.csv"
    source.write_text(
        "dataset,encoder_id,x,y,spearman,ci_low,ci_high,cluster_unit\n"
        "celeba,vicreg_celeba_epoch1000,interaction_defect_normalized,"
        "source_transfer_gap,-0.4,-0.6,-0.2,target_attribute\n"
        "celeba,vicreg_celeba_epoch1000,midpoint_drift_abs,"
        "source_transfer_gap,-0.2,-0.5,0.1,target_attribute\n",
        encoding="utf-8",
    )
    tg_style.apply_style()
    pdf, png = figures.figS5_failures([("celeba", source)], tmp_path / "figureS5")
    assert pdf.is_file()
    assert png.is_file()


def test_bound_legend_identifies_empirical_plugin_rhs() -> None:
    labels = [label for _key, label, _color, _marker, _line in figures.BOUND_SERIES]
    assert "Thm 4.5 plug-in RHS (ours)" in labels


def test_figure1_design_has_four_conditions_and_three_tasks() -> None:
    tasks = ("scale", "posX", "posY")
    assert figures.CONDITION_ORDER == (
        "all_shared",
        "scale_varies",
        "posX_varies",
        "posY_varies",
    )
    assert len(figures.CONDITION_ORDER) * len(tasks) == 12
    assert (len(figures.CONDITION_ORDER) - 1) * len(tasks) == 9
