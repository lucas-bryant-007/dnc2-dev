from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "analysis" / "run_paper_rerun_s2.sh"


def test_s2_launcher_freezes_distinct_reproduction_and_stability_estimands():
    text = LAUNCHER.read_text(encoding="utf-8")
    assert '"$FULL_TAG" none none' in text
    assert '"$STABILITY_TAG" 500 none' in text
    assert "--max_test_cell_samples 350" in text
    assert "--require_reproduction" in text
    assert "REPRODUCTION_ATOL" in text


def test_s2_launcher_uses_all_four_workers_and_records_required_controls():
    text = LAUNCHER.read_text(encoding="utf-8")
    assert 'VICREG_GPU="${VICREG_GPU:-0}"' in text
    assert 'IJEPA_GPU="${IJEPA_GPU:-1}"' in text
    assert 'CUB_GPU="${CUB_GPU:-2}"' in text
    assert 'STABILITY_GPU="${STABILITY_GPU:-3}"' in text
    assert "--fewshot_dir" in text
    assert "--n_permutations \"$N_PERMUTATIONS\"" in text
    assert "full_pipeline_label_permutation" in text
    assert "SHA256SUMS" in text


def test_server_checkout_preserves_checksummed_artifact_bytes():
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert (
        "paper_outputs/pretrained_crossfit_postaudit_20260810/** -text -diff"
        in attributes
    )
    text = LAUNCHER.read_text(encoding="utf-8")
    assert '[[ -s "$1" ]]' in text
