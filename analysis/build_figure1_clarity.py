"""Build the clarified controlled-pairing Figure 1 with source provenance."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

from analysis import paper_figures_v2 as figures
from analysis.tg_style import apply_style

REPO_ROOT = Path(__file__).resolve().parents[1]

README = """# Clarified Figure 1

This candidate changes presentation only; it uses the same frozen CSV as the
audited 2026-08-24 release.

The revised label makes clear that:

- four positive-pair conditions are trained independently;
- each condition has seeds 6, 17, and 29;
- the plot measures three downstream tasks;
- "9 shared" counts condition-task paths, not factors or models;
- curves are seed means and shading is the seed minimum-maximum range.

See `docs/figure1_pairing_ablation.md` for the complete design explanation.

Rebuild to a new directory with
`python -m analysis.build_figure1_clarity --output <new-directory>`.
`MANIFEST.csv` hashes the frozen input, code, and figure outputs.
"""


def _manifest(output: Path, inputs: list[Path]) -> None:
    rows = []
    for path in inputs:
        rows.append(
            {
                "role": "input",
                "path": Path(os.path.relpath(path, REPO_ROOT)).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    for path in sorted(
        item
        for item in output.iterdir()
        if item.is_file() and item.name not in {"MANIFEST.csv", "README.md"}
    ):
        rows.append(
            {
                "role": "output",
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    with (output / "MANIFEST.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build(config: Path, output: Path) -> Path:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing Figure 1 package: {output}")
    release = json.loads(config.read_text(encoding="utf-8"))
    source = (
        (REPO_ROOT / release["controlled_dir"]).resolve()
        / "paper_summary"
        / "augmentation_survival.csv"
    )
    if not source.is_file():
        raise FileNotFoundError(f"missing frozen Figure 1 input: {source}")
    output.mkdir(parents=True)
    (output / "README.md").write_text(README, encoding="utf-8")
    apply_style()
    figures.fig1_augmentation(source, output / "figure1_augmentation")
    _manifest(
        output,
        [config, source, Path(__file__).resolve(), Path(figures.__file__).resolve()],
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "paper_release_20260824.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "paper_outputs" / "figure1_clarity_20260825",
    )
    args = parser.parse_args()
    print(build(args.config.resolve(), args.output.resolve()))


if __name__ == "__main__":
    main()
