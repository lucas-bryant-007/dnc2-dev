from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "analysis" / "run_paper_rerun_s2.sh"


def test_s2_launcher_freezes_distinct_reproduction_and_stability_estimands():
    text = LAUNCHER.read_text(encoding="utf-8")
    assert '"$FULL_TAG" none none' in text
    assert '"$STABILITY_TAG" 500 none' in text
    assert "--max_test_cell_samples 350" in text
    assert "--require_reproduction" not in text
    assert "--finalize-existing" in text
    assert '--reference_json "$VICREG_FULL_JSON" "$IJEPA_FULL_JSON"' in text
    assert "capped_stability_vs_fresh_full_support" in text
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


def test_finalization_is_resumable_and_completion_marker_is_written_last():
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "record_finalization_provenance" in text
    assert "source_compute_commit=" in text
    assert "finalizer_commit=" in text
    assert "validate_worker_outputs" in text
    assert "environment_vs_archived_reference.diff" in text
    assert "exact_match=" in text
    assert "! -name COMPLETE" in text
    finalizer = text.index("finalize_outputs()")
    checksum = text.index("    write_checksums", finalizer)
    complete = text.index('    date --iso-8601=seconds >"$OUT_BASE/COMPLETE"', finalizer)
    assert checksum < complete


def test_current_geometry_writers_emit_explicit_protocol_versions():
    celeba = (ROOT / "analysis" / "celeba_hyperrect_crossfit.py").read_text(
        encoding="utf-8"
    )
    cub = (ROOT / "analysis" / "cub200_hyperrect_crossfit.py").read_text(
        encoding="utf-8"
    )
    assert "celeba_full_train_ssl_whitening_independent_third_fold_v1" in celeba
    assert "cub200_independent_third_fold_whitening_v1" in cub
    assert '"analysis_protocol_version": ANALYSIS_PROTOCOL_VERSION' in celeba
    assert '"analysis_protocol_version": ANALYSIS_PROTOCOL_VERSION' in cub
