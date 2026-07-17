import json

from analysis.build_results_manifest import build_manifest
from analysis.plot_pretrained_summary import (
    load_completed_runs,
    plot_interference,
    plot_summary,
)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _central(method):
    return {
        "method": method,
        "attribute": "Male",
        "epoch": 1000,
        "config": f"configs/eval/{method}_celeba_hf.yaml",
        "ckpt_path": f"/checkpoints/{method}.ckpt",
        "results": {
            "adaptive": {
                "k_eff": 8,
                "r_values": [2, 4],
                "B_r": {"2": 0.2, "4": 0.35},
                "tilde_V_pred": {"2": 2.0, "4": 1.0},
                "tilde_V_obs": {"2": 1.9, "4": 1.05},
            }
        },
    }


def _hyper(method):
    values = [0.2, 0.3, 0.25]
    return {
        "method": method,
        "epoch": 1000,
        "split": "test",
        "whitened": True,
        "tag": "test",
        "mean_abs_offdiag_cosine": 0.1,
        "metrics": [
            {
                "name": name,
                "capture_B": capture,
                "pos_frac": prevalence,
                "usable": True,
            }
            for name, capture, prevalence in zip(
                ("Attractive", "Male", "Smiling"),
                values,
                (0.5, 0.4, 0.5),
                strict=True,
            )
        ],
        "cosine_matrix": [[1.0, 0.1, 0.05], [0.1, 1.0, -0.15], [0.05, -0.15, 1.0]],
    }


def _bundle(tmp_path):
    for method in ("ijepa", "vicreg"):
        metrics = tmp_path / f"celeba_{method}_hf" / "metrics"
        _write_json(metrics / f"metrics_{method}_male_epoch_1000_test.json", _central(method))
        _write_json(
            metrics / f"hyperrect_{method}_celeba_epoch_1000_test.json",
            _hyper(method),
        )
    return tmp_path


def test_matched_report_writes_png_and_pdf(tmp_path):
    root = _bundle(tmp_path / "results")
    runs = load_completed_runs(root)

    summary = plot_summary(runs, root / "paper" / "summary")
    heatmap = plot_interference(runs, root / "paper" / "interference")

    assert {path.suffix for path in summary} == {".png", ".pdf"}
    assert {path.suffix for path in heatmap} == {".png", ".pdf"}
    assert all(path.stat().st_size > 0 for path in summary + heatmap)


def test_manifest_hashes_artifacts_and_records_run_commits(tmp_path):
    root = _bundle(tmp_path / "results")
    output = root / "reproducibility_manifest.json"

    manifest = build_manifest(
        root,
        output,
        run_commits={"celeba_ijepa_hf": "abc123"},
    )

    assert manifest["run_commits"]["celeba_ijepa_hf"] == "abc123"
    assert len(manifest["artifacts"]) == 4
    assert all(len(row["sha256"]) == 64 for row in manifest["artifacts"])
    assert {row["method"] for row in manifest["metric_records"]} == {"ijepa", "vicreg"}
