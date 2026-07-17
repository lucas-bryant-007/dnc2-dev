"""Repair inconsistent legacy I-JEPA geometry metadata in a checkpoint copy.

The repair is deliberately conservative: the encoder name and positional-
embedding token count must agree before any metadata is changed. The input is
never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
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


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _state_signature(checkpoint: dict[str, Any]) -> dict[str, dict[str, Any]]:
    state = _mapping(checkpoint.get("state_dict"), "state_dict")
    return {
        key: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for key, value in state.items()
        if isinstance(value, torch.Tensor)
    }


def build_repair_record(
    source: Path,
    destination: Path,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Build an auditable sidecar after both checkpoint files exist."""
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "utility": "analysis/repair_legacy_ijepa_checkpoint.py",
        "source": {
            "path": str(source),
            "bytes": source.stat().st_size,
            "sha256": _sha256_file(source),
        },
        "repaired_copy": {
            "path": str(destination),
            "bytes": destination.stat().st_size,
            "sha256": _sha256_file(destination),
        },
        "repair": metadata,
    }


def _write_repair_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(args: argparse.Namespace) -> None:
    source = Path(args.input).expanduser().resolve()
    destination = Path(args.output).expanduser().resolve()
    if source == destination:
        raise SystemExit("Refusing to overwrite the input checkpoint")
    if not source.is_file():
        raise SystemExit(f"Input checkpoint does not exist: {source}")
    if destination.exists() and not args.force and not args.record_only:
        raise SystemExit(f"Output already exists (use --force to replace): {destination}")

    record_path = (
        Path(args.record).expanduser().resolve()
        if args.record
        else destination.with_name(destination.name + ".repair.json")
    )

    if args.record_only:
        if not destination.is_file():
            raise SystemExit(f"Repaired checkpoint does not exist: {destination}")
        source_checkpoint = torch.load(source, map_location="cpu", weights_only=False)
        expected = repair_checkpoint_dict(source_checkpoint)
        repaired = torch.load(destination, map_location="cpu", weights_only=False)
        metadata = repaired.get("legacy_metadata_repair")
        if metadata != expected["legacy_metadata_repair"]:
            raise SystemExit("Repaired checkpoint metadata does not match the validated source")
        if _state_signature(repaired) != _state_signature(source_checkpoint):
            raise SystemExit("Repaired checkpoint tensor structure differs from the source")
        record = build_repair_record(source, destination, metadata)
        _write_repair_record(record_path, record)
        print(f"Validated existing repaired copy: {destination}")
        print(f"Saved repair record: {record_path}")
        return

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
    record = build_repair_record(source, destination, metadata)
    _write_repair_record(record_path, record)
    print(f"Validated encoder: {metadata['encoder_type']}")
    print(f"Position embeddings: {metadata['validated_position_embeddings']}")
    print(f"Original metadata: {metadata['original']}")
    print(f"Corrected metadata: {metadata['corrected']}")
    print(f"Saved repaired copy: {destination}")
    print(f"Saved repair record: {record_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Legacy checkpoint")
    parser.add_argument("--output", required=True, help="New repaired checkpoint path")
    parser.add_argument("--force", action="store_true", help="Replace an existing output")
    parser.add_argument(
        "--record",
        default=None,
        help="Repair sidecar path (default: <output>.repair.json)",
    )
    parser.add_argument(
        "--record_only",
        action="store_true",
        help="Validate existing input/output files and write only the repair sidecar",
    )
    main(parser.parse_args())
