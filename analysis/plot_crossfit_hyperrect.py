"""Re-render a cross-fitted hyper-rectangle JSON without model inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from box_viz import plot_box_3d
import metrics_io as mio


def render_crossfit_json(json_path: Path, output_dir: Path) -> list[Path]:
    with json_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    protocol = payload.get("protocol") or {}
    if protocol.get("selection_split") != "train" or protocol.get("evaluation_split") != "test":
        raise ValueError("Expected a train-selection/test-evaluation cross-fit record")
    if not protocol.get("triple_frozen_before_test_label_analysis"):
        raise ValueError("The record does not certify that the triple was frozen before test")
    test = payload["test_evaluation"]
    triple = list(payload["selected_triple"])
    if test.get("triple_names") != triple:
        raise ValueError("Selected and evaluated triples do not match")
    if test.get("box") is None or test.get("predicted_box") is None:
        raise ValueError("Cross-fit record does not contain both observed and predicted boxes")

    method = mio.slug(payload["method"])
    epoch = payload["epoch"]
    stem = f"celeba_balanced_cube_{method}_epoch_{epoch}"
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [output_dir / f"{stem}.png", output_dir / f"{stem}.pdf"]
    plot_box_3d(
        np.empty((0, 3), dtype=np.float32),
        test["box"],
        np.empty((0,), dtype=np.int64),
        triple,
        [str(path) for path in outputs],
        predicted_box=test["predicted_box"],
        axis_labels=[name.replace("_", " ") for name in triple],
        show_samples=False,
        show_centroid_se=True,
        publication_compact=True,
    )
    return outputs


def main(args: argparse.Namespace) -> None:
    json_path = Path(args.json).expanduser().resolve()
    if not json_path.is_file():
        raise SystemExit(f"Metrics JSON does not exist: {json_path}")
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else json_path.parent.parent / "paper_figures"
    )
    for path in render_crossfit_json(json_path, output_dir):
        print(f"Saved: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="Cross-fit metrics JSON")
    parser.add_argument("--output_dir", default=None)
    main(parser.parse_args())
