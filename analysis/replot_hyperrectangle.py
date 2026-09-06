"""Regenerate a hyperrectangle figure from validated JSON and point sidecar."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from analysis.hyperrectangle import plot_hyperrectangle
except ModuleNotFoundError:  # Direct execution: python analysis/replot_hyperrectangle.py
    from hyperrectangle import plot_hyperrectangle


SUBTITLES = {
    "vicreg_celeba": "VICReg, pretrained on CelebA",
    "vicreg_imagenet": "VICReg, pretrained on ImageNet-1K",
    "ijepa_imagenet": "I-JEPA, pretrained on ImageNet-1K",
}


def replot_result(json_path, output_path, *, sample_size=8, sample_alpha=0.24):
    json_path = Path(json_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    record = payload.get("plot_points") or {}
    points_path = (json_path.parent / record["artifact"]).resolve()
    if points_path.parent != json_path.parent:
        raise ValueError("plot-point sidecar must be beside its result JSON")
    with np.load(points_path, allow_pickle=False) as archive:
        coordinates = np.asarray(archive["coords"], dtype=np.float32)
        cells = np.asarray(archive["granular_task"], dtype=np.int64)
        triple = [str(value) for value in archive["triple_names"].tolist()]
    if triple != payload["selected_triple"]:
        raise ValueError("point-sidecar triple does not match the result JSON")
    model_name = payload["model"]["name"]
    geometry = payload["test_evaluation"]["crossfit_probe_geometry"]
    plot_hyperrectangle(
        output_path,
        payload["selected_triple"],
        payload["test_evaluation"]["box"],
        payload["test_evaluation"]["predicted_box"],
        subtitle=SUBTITLES.get(model_name, model_name),
        diagnostics=payload["test_box_diagnostics"],
        maximum_cosine=geometry["max_abs_cos"],
        side_lengths=payload["test_side_length_diagnostics"],
        passed=payload["headline_criteria_passed"],
        sample_coordinates=coordinates,
        sample_cells=cells,
        samples_per_cell=record["samples_per_cell"],
        sample_size=sample_size,
        sample_alpha=sample_alpha,
    )
    return output_path.with_suffix(".png"), output_path.with_suffix(".pdf")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-size", type=float, default=8)
    parser.add_argument("--sample-alpha", type=float, default=0.24)
    args = parser.parse_args()
    for path in replot_result(
            args.json, args.output,
            sample_size=args.sample_size, sample_alpha=args.sample_alpha):
        print(path)


if __name__ == "__main__":
    main()
