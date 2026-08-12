"""Rebuild the compact natural-image CSVs from a finished paper-rerun directory.

The shipped `publication_data_20260811/*.csv` were extracted from the Aug-11 run,
which predates the eval-mode fix (`freeze_model` never called `model.eval()`, so
BatchNorm used per-batch statistics). This regenerates them from a corrected run.

Source fields were identified by matching the shipped CSV values back to the
Aug-11 JSON rather than guessed, because two of them are not the obvious choice:

  capture_values / max_abs_cos  <- test_stability.aggregate_crossfit_probe_geometry
                                   (the 20-resample aggregate, NOT test_evaluation)
  primary_rmse                  <- test_box_diagnostics.normalized_centroid_rmse
  stability_*                   <- test_stability.statistics.normalized_centroid_rmse
  passes / n_resamples          <- test_stability.pass_count / .n_resamples
  permutation columns           <- controls/heldout/.../heldout_permutation_null_*.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

PANELS = (
    ("celeba_vicreg", "vicreg_celeba", "VICReg · CelebA"),
    ("celeba_ijepa", "ijepa_celeba", "I-JEPA · CelebA"),
    ("cub200_vicreg", "vicreg_cub200", "VICReg · CUB-200"),
)


def _only(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one {pattern} in {directory}, got {len(matches)}")
    return matches[0]


def rebuild(run_root: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    geometry_rows, permutation_rows = [], []

    for slug, null_slug, label in PANELS:
        payload = json.loads(
            _only(run_root / "full_support" / slug / "metrics", "*.json").read_text(encoding="utf-8")
        )
        stability = payload["test_stability"]
        aggregate = stability["aggregate_crossfit_probe_geometry"]
        statistics = stability["statistics"]["normalized_centroid_rmse"]

        capture = aggregate["capture_B"]
        if isinstance(capture, dict):
            capture = [capture[name] for name in payload["test_evaluation"]["triple_names"]]

        geometry_rows.append({
            "label": label,
            "capture_values": "|".join(f"{value:.4f}" for value in capture),
            "max_abs_cos": f"{aggregate['max_abs_cos']:.4f}",
            "primary_rmse": f"{payload['test_box_diagnostics']['normalized_centroid_rmse']:.4f}",
            "stability_mean": f"{statistics['mean']:.4f}",
            "stability_min": f"{statistics['min']:.4f}",
            "stability_max": f"{statistics['max']:.4f}",
            "passes": stability["pass_count"],
            "n_resamples": stability["n_resamples"],
        })

        null_dir = run_root / "controls" / "heldout" / "full_support" / null_slug
        null = json.loads(_only(null_dir, "*.json").read_text(encoding="utf-8"))
        permutation_rows.append({
            "label": label,
            "observed": f"{null['observed_normalized_centroid_rmse']:.4f}",
            "null_min": f"{null['null_quantiles']['q00']:.4f}",
            "null_mean": f"{null['null_mean']:.4f}",
            "p_value": null["empirical_lower_tail_p"],
            "n_permutations": null.get("n_permutations", null.get("permutations")),
        })

    written = []
    for name, rows in (("natural_geometry_summary.csv", geometry_rows),
                       ("permutation_summary.csv", permutation_rows)):
        path = out_dir / name
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    for path in rebuild(Path(args.run_root), Path(args.out_dir)):
        print(f"Wrote: {path}")
        print(path.read_text(encoding="utf-8").rstrip())


if __name__ == "__main__":
    main()
