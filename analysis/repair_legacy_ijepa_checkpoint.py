"""Repair inconsistent legacy I-JEPA geometry metadata in a checkpoint copy.

The repair is deliberately conservative: the encoder name and positional-
embedding token count must agree before any metadata is changed. The input is
never overwritten.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import torch


ENCODER_GEOMETRY = {
    "vit_base_patch16_224": {"patch_size": 16, "image_size": 224, "tokens": 197},
    "vit_large_patch16_224": {"patch_size": 16, "image_size": 224, "tokens": 197},
    "vit_base_patch32_224": {"patch_size": 32, "image_size": 224, "tokens": 50},
}


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Checkpoint {name} must be a dictionary")
    return value


def repair_checkpoint_dict(checkpoint: dict[str, Any]) -> dict[str, Any]:
    hparams = _mapping(checkpoint.get("hyper_parameters"), "hyper_parameters")
    model = _mapping(hparams.get("model"), "hyper_parameters.model")
    data = _mapping(hparams.get("data"), "hyper_parameters.data")
    state = _mapping(checkpoint.get("state_dict"), "state_dict")

    encoder_type = str(model.get("encoder_type", "")).lower()
    if encoder_type not in ENCODER_GEOMETRY:
        raise ValueError(
            f"Unsupported or missing encoder_type={encoder_type!r}; "
            f"supported values are {sorted(ENCODER_GEOMETRY)}"
        )
    expected = ENCODER_GEOMETRY[encoder_type]

    observed = {
        key: int(value.shape[1])
        for key, value in state.items()
        if key.endswith("pos_embed")
        and isinstance(value, torch.Tensor)
        and value.ndim == 3
    }
    matching = {key: tokens for key, tokens in observed.items()
                if tokens == expected["tokens"]}
    if not matching:
        raise ValueError(
            f"No positional embedding matches {encoder_type}: expected "
            f"{expected['tokens']} tokens, observed {observed or 'none'}"
        )

    original = {
        "model.patch_size": model.get("patch_size"),
        "model.image_size": model.get("image_size"),
        "data.patch_size": data.get("patch_size"),
        "data.img_size": data.get("img_size"),
    }
    model["patch_size"] = expected["patch_size"]
    model["image_size"] = expected["image_size"]
    data["patch_size"] = expected["patch_size"]
    data["img_size"] = expected["image_size"]
    checkpoint["legacy_metadata_repair"] = {
        "utility": "analysis/repair_legacy_ijepa_checkpoint.py",
        "encoder_type": encoder_type,
        "validated_position_embeddings": matching,
        "original": original,
        "corrected": {
            "patch_size": expected["patch_size"],
            "image_size": expected["image_size"],
        },
    }
    return checkpoint


def main(args: argparse.Namespace) -> None:
    source = Path(args.input).expanduser().resolve()
    destination = Path(args.output).expanduser().resolve()
    if source == destination:
        raise SystemExit("Refusing to overwrite the input checkpoint")
    if not source.is_file():
        raise SystemExit(f"Input checkpoint does not exist: {source}")
    if destination.exists() and not args.force:
        raise SystemExit(f"Output already exists (use --force to replace): {destination}")

    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    repaired = repair_checkpoint_dict(checkpoint)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        torch.save(repaired, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()

    metadata = repaired["legacy_metadata_repair"]
    print(f"Validated encoder: {metadata['encoder_type']}")
    print(f"Position embeddings: {metadata['validated_position_embeddings']}")
    print(f"Original metadata: {metadata['original']}")
    print(f"Corrected metadata: {metadata['corrected']}")
    print(f"Saved repaired copy: {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Legacy checkpoint")
    parser.add_argument("--output", required=True, help="New repaired checkpoint path")
    parser.add_argument("--force", action="store_true", help="Replace an existing output")
    main(parser.parse_args())
