from pathlib import Path

import numpy as np

from analysis.plot_strict_pretrained_paper import (
    FullPipelineNull,
    RunSummary,
    write_results_note,
)


def _run(dataset, method, label, rmse):
    return RunSummary(
        dataset=dataset,
        method=method,
        label=label,
        triple=("a", "b", "c"),
        capture=(0.2, 0.3, 0.4),
        aggregate_max_cos=0.1,
        rmse=rmse,
        corner_stability_status="valid_current_geometry",
        pass_count=2,
        n_resamples=2,
        samples_per_cell=500,
        feasible_train_candidates=10,
        plot_coords=np.zeros((8, 3)),
        granular_task=np.arange(8),
        observed_box=np.zeros((8, 3)),
        predicted_box=np.zeros((8, 3)),
        source=Path("fixture.json"),
    )


def test_fresh_results_note_reports_current_resampling(tmp_path):
    runs = [
        _run("celeba", "vicreg", "VICReg / CelebA", (0.14, 0.16)),
        _run("celeba", "ijepa", "I-JEPA / CelebA", (0.25, 0.27)),
        _run(
            "cub200",
            "vicreg_official_imagenet1k",
            "VICReg / CUB-200",
            (0.29, 0.31),
        ),
    ]
    nulls = [
        {
            "method": run.method,
            "dataset": run.dataset,
            "observed_normalized_centroid_rmse": run.rmse[0],
            "null_mean": 1.0,
            "empirical_lower_tail_p": 1 / 5001,
        }
        for run in runs
    ]
    full_nulls = [
        FullPipelineNull(
            method=method,
            label=label,
            seed=3101,
            feasible_train_candidates=0,
            selection_succeeded=False,
            source=Path("null.json"),
        )
        for method, label in (("vicreg", "VICReg"), ("ijepa", "I-JEPA"))
    ]
    output = tmp_path / "RESULTS.md"
    write_results_note(runs, nulls, full_nulls, output)
    text = output.read_text(encoding="utf-8")
    assert "VICReg / CelebA: mean=0.150" in text
    assert "Across 2 correlated balance resamples" in text
    assert "obsolete 20-resample corner values" not in text
    assert "pending full rerun" not in text
