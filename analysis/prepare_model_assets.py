"""Download and validate the frozen model assets used by the paper analyses.

The CelebA checkpoints are published as safetensors files.  The evaluation
code consumes PyTorch Lightning checkpoints, so this utility performs the
format conversion explicitly and records byte-level provenance.  Revisions and
source hashes are pinned below; a changed remote file is rejected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from analysis.repair_legacy_ijepa_checkpoint import repair_checkpoint_dict
from models.public_weights import (
    PUBLIC_WEIGHTS,
    PublicWeights,
    load_supervised_imagenet_resnet50,
    load_vicreg_imagenet_resnet50,
)
from training.config_loader import dict_to_namespace


@dataclass(frozen=True)
class PublishedCheckpoint:
    name: str
    repo_id: str
    revision: str
    epoch: int
    safetensors_sha256: str
    safetensors_bytes: int
    output_subdir: str

    @property
    def source_filename(self) -> str:
        return f"checkpoints/epoch_{self.epoch:04d}/model.safetensors"

    @property
    def metadata_filename(self) -> str:
        return f"checkpoints/epoch_{self.epoch:04d}/metadata.json"


PUBLISHED_CHECKPOINTS = {
    "vicreg-celeba": PublishedCheckpoint(
        name="vicreg-celeba",
        repo_id="dlf-ssl/vicreg-resnet50-celeba",
        revision="8aaad82ff7eb05092cda9b6ff7231db5f71430b0",
        epoch=1000,
        safetensors_sha256=(
            "2b4a43a833839d3a4aa7fa2bce3295c71478f75fe97df9a0f9dd5df4ac43b132"
        ),
        safetensors_bytes=127_870_936,
        output_subdir="vicreg-resnet50-celeba",
    ),
    "ijepa-celeba": PublishedCheckpoint(
        name="ijepa-celeba",
        repo_id="dlf-ssl/ijepa-resnet50-celeba",
        revision="8e28f7fc9c720061d4dc6a246bcf2301c8ece4e2",
        epoch=1000,
        safetensors_sha256=(
            "11e3c1ed0efa27efa01e131c1302f257eb253c63b5897276aa9db649857faa1b"
        ),
        safetensors_bytes=723_673_528,
        output_subdir="ijepa-resnet50-celeba",
    ),
}


ALL_ASSETS = {**PUBLISHED_CHECKPOINTS, **PUBLIC_WEIGHTS}


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _torch_save_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _download_file(spec: PublishedCheckpoint, filename: str, cache_dir: Path) -> Path:
    return Path(
        hf_hub_download(
            repo_id=spec.repo_id,
            filename=filename,
            revision=spec.revision,
            cache_dir=cache_dir,
        )
    )


def _load_published_payload(
    spec: PublishedCheckpoint,
    cache_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, str]]:
    tensor_path = _download_file(spec, spec.source_filename, cache_dir)
    config_path = _download_file(spec, "config.json", cache_dir)
    metadata_path = _download_file(spec, spec.metadata_filename, cache_dir)

    if tensor_path.stat().st_size != spec.safetensors_bytes:
        raise RuntimeError(
            f"{spec.name} size mismatch: expected {spec.safetensors_bytes}, "
            f"got {tensor_path.stat().st_size}"
        )
    observed_hash = _sha256_file(tensor_path)
    if observed_hash != spec.safetensors_sha256:
        raise RuntimeError(
            f"{spec.name} SHA-256 mismatch: expected {spec.safetensors_sha256}, "
            f"got {observed_hash}"
        )

    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("epoch") != spec.epoch:
        raise RuntimeError(
            f"{spec.name} metadata epoch is {metadata.get('epoch')!r}, "
            f"expected {spec.epoch}"
        )
    if metadata.get("exported_format") != ".safetensors":
        raise RuntimeError(f"{spec.name} metadata does not declare safetensors")

    state_dict = load_file(tensor_path, device="cpu")
    if not state_dict:
        raise RuntimeError(f"{spec.name} contains an empty state dictionary")
    source_hashes = {
        "model.safetensors": observed_hash,
        "config.json": _sha256_file(config_path),
        "metadata.json": _sha256_file(metadata_path),
    }
    return state_dict, config, metadata, source_hashes


def _validate_state_dict(
    spec: PublishedCheckpoint,
    state_dict: dict[str, torch.Tensor],
    config: dict[str, Any],
) -> None:
    validation_config = json.loads(json.dumps(config))
    if spec.name == "ijepa-celeba":
        # The published legacy config says patch 14 / image 128, while its
        # positional embeddings and encoder name unambiguously encode ViT-B/16
        # at 224 px.  The repaired checkpoint records this correction.
        validation_config["model"]["patch_size"] = 16
        validation_config["model"]["image_size"] = 224
        validation_config["data"]["patch_size"] = 16
        validation_config["data"]["img_size"] = 224

    cfg = dict_to_namespace(validation_config)
    if spec.name == "vicreg-celeba":
        from models.vicreg import LightlyVICReg

        model = LightlyVICReg(cfg)
    else:
        from models.ijepa import LightlyIJEPA

        model = LightlyIJEPA(cfg)
    incompatible = model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"{spec.name} state mismatch: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    del model


def _checkpoint_payload(
    spec: PublishedCheckpoint,
    state_dict: dict[str, torch.Tensor],
    config: dict[str, Any],
    metadata: dict[str, Any],
    source_hashes: dict[str, str],
) -> dict[str, Any]:
    return {
        "epoch": spec.epoch,
        "global_step": 0,
        "state_dict": state_dict,
        "hyper_parameters": config,
        "asset_provenance": {
            "schema_version": 1,
            "repository": spec.repo_id,
            "revision": spec.revision,
            "source_filename": spec.source_filename,
            "source_hashes": source_hashes,
            "source_metadata": metadata,
            "conversion": "safetensors state_dict to minimal Lightning checkpoint",
        },
    }


def _expected_outputs(spec: PublishedCheckpoint, asset_root: Path) -> list[Path]:
    base = asset_root / "hf_models" / spec.output_subdir
    outputs = [
        base / "converted_checkpoints" / f"epoch_{spec.epoch:04d}.ckpt",
        base / "ASSET_PROVENANCE.json",
    ]
    if spec.name == "ijepa-celeba":
        outputs.extend(
            [
                base / "repaired_checkpoints" / f"epoch_{spec.epoch:04d}.ckpt",
                base
                / "repaired_checkpoints"
                / f"epoch_{spec.epoch:04d}.ckpt.repair.json",
            ]
        )
    return outputs


def _relative(path: Path, asset_root: Path) -> str:
    return path.resolve().relative_to(asset_root.resolve()).as_posix()


def _prepare_checkpoint(
    spec: PublishedCheckpoint,
    asset_root: Path,
    cache_dir: Path,
    *,
    force: bool,
) -> dict[str, Any]:
    outputs = _expected_outputs(spec, asset_root)
    existing = [path for path in outputs if path.exists()]
    if existing and not force:
        rendered = "\n".join(f"  - {path}" for path in existing)
        raise RuntimeError(
            f"Refusing to replace existing {spec.name} outputs. "
            f"Use --force only after reviewing them:\n{rendered}"
        )

    state_dict, config, metadata, source_hashes = _load_published_payload(
        spec, cache_dir
    )
    _validate_state_dict(spec, state_dict, config)
    checkpoint = _checkpoint_payload(
        spec, state_dict, config, metadata, source_hashes
    )

    base = asset_root / "hf_models" / spec.output_subdir
    converted = base / "converted_checkpoints" / f"epoch_{spec.epoch:04d}.ckpt"
    _torch_save_atomic(converted, checkpoint)

    repaired = None
    repair_record = None
    if spec.name == "ijepa-celeba":
        repaired = base / "repaired_checkpoints" / f"epoch_{spec.epoch:04d}.ckpt"
        repaired_payload = repair_checkpoint_dict(checkpoint)
        _torch_save_atomic(repaired, repaired_payload)
        repair_record = repaired.with_name(repaired.name + ".repair.json")
        _write_json_atomic(
            repair_record,
            {
                "schema_version": 1,
                "source": {
                    "path": _relative(converted, asset_root),
                    "bytes": converted.stat().st_size,
                    "sha256": _sha256_file(converted),
                },
                "repaired_copy": {
                    "path": _relative(repaired, asset_root),
                    "bytes": repaired.stat().st_size,
                    "sha256": _sha256_file(repaired),
                },
                "repair": repaired_payload["legacy_metadata_repair"],
            },
        )

    record: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "name": spec.name,
        "source": {
            "repository": spec.repo_id,
            "revision": spec.revision,
            "filename": spec.source_filename,
            "bytes": spec.safetensors_bytes,
            "sha256": spec.safetensors_sha256,
            "config_sha256": source_hashes["config.json"],
            "metadata_sha256": source_hashes["metadata.json"],
        },
        "converted_checkpoint": {
            "path": _relative(converted, asset_root),
            "bytes": converted.stat().st_size,
            "sha256": _sha256_file(converted),
            "state_dict_keys": len(state_dict),
            "strict_load_validated": True,
        },
    }
    if repaired is not None and repair_record is not None:
        record["repaired_checkpoint"] = {
            "path": _relative(repaired, asset_root),
            "bytes": repaired.stat().st_size,
            "sha256": _sha256_file(repaired),
            "repair_record": _relative(repair_record, asset_root),
        }
    _write_json_atomic(base / "ASSET_PROVENANCE.json", record)
    return record


def _require_recorded_file(
    path: Path,
    record: dict[str, Any],
    label: str,
) -> None:
    if not path.is_file():
        raise RuntimeError(f"Missing {label}: {path}")
    observed_bytes = path.stat().st_size
    if observed_bytes != record.get("bytes"):
        raise RuntimeError(
            f"{label} size mismatch: recorded {record.get('bytes')}, "
            f"observed {observed_bytes}"
        )
    observed_hash = _sha256_file(path)
    if observed_hash != record.get("sha256"):
        raise RuntimeError(
            f"{label} SHA-256 mismatch: recorded {record.get('sha256')}, "
            f"observed {observed_hash}"
        )


def _public_weights_path(spec: PublicWeights, asset_root: Path) -> Path:
    return asset_root / "cache" / "torch" / "hub" / "checkpoints" / spec.filename


def _validate_public_weights(spec: PublicWeights, path: Path) -> None:
    if spec.name == "vicreg-imagenet":
        load_vicreg_imagenet_resnet50(path)
    else:
        load_supervised_imagenet_resnet50(path)


def _verify_public_file(spec: PublicWeights, path: Path) -> None:
    if not path.is_file():
        raise RuntimeError(f"Missing {spec.name}: {path}")
    if path.stat().st_size != spec.size_bytes:
        raise RuntimeError(
            f"{spec.name} size mismatch: expected {spec.size_bytes}, "
            f"observed {path.stat().st_size}"
        )
    observed_hash = _sha256_file(path)
    if observed_hash != spec.sha256:
        raise RuntimeError(
            f"{spec.name} SHA-256 mismatch: expected {spec.sha256}, "
            f"observed {observed_hash}"
        )
    _validate_public_weights(spec, path)


def _prepare_public_weights(
    spec: PublicWeights,
    asset_root: Path,
    *,
    force: bool,
) -> dict[str, Any]:
    destination = _public_weights_path(spec, asset_root)
    if destination.exists() and not force:
        # Adopt an existing weight file only if its complete bytes and model
        # structure match the same pin.
        _verify_public_file(spec, destination)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        try:
            with urllib.request.urlopen(spec.url) as response, temporary.open(
                "wb"
            ) as handle:
                shutil.copyfileobj(response, handle, length=8 * 1024 * 1024)
            _verify_public_file(spec, temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "name": spec.name,
        "source": {
            "url": spec.url,
            "bytes": spec.size_bytes,
            "sha256": spec.sha256,
        },
        "weights": {
            "path": _relative(destination, asset_root),
            "bytes": destination.stat().st_size,
            "sha256": _sha256_file(destination),
            "strict_structure_validated": True,
        },
    }


def _verify_public_weights(spec: PublicWeights, asset_root: Path) -> dict[str, Any]:
    path = _public_weights_path(spec, asset_root)
    _verify_public_file(spec, path)
    return {
        "name": spec.name,
        "source": {
            "url": spec.url,
            "bytes": spec.size_bytes,
            "sha256": spec.sha256,
        },
        "weights": {
            "path": _relative(path, asset_root),
            "bytes": path.stat().st_size,
            "sha256": spec.sha256,
            "strict_structure_validated": True,
        },
    }


def _verify_checkpoint(spec: PublishedCheckpoint, asset_root: Path) -> dict[str, Any]:
    base = asset_root / "hf_models" / spec.output_subdir
    provenance_path = base / "ASSET_PROVENANCE.json"
    if not provenance_path.is_file():
        raise RuntimeError(f"Missing asset provenance: {provenance_path}")
    with provenance_path.open("r", encoding="utf-8") as handle:
        record = json.load(handle)

    expected_source = {
        "repository": spec.repo_id,
        "revision": spec.revision,
        "filename": spec.source_filename,
        "bytes": spec.safetensors_bytes,
        "sha256": spec.safetensors_sha256,
    }
    observed_source = record.get("source", {})
    mismatches = {
        key: {"expected": value, "observed": observed_source.get(key)}
        for key, value in expected_source.items()
        if observed_source.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"{spec.name} pinned source mismatch: {mismatches}")

    converted_record = record.get("converted_checkpoint", {})
    converted = (
        base / "converted_checkpoints" / f"epoch_{spec.epoch:04d}.ckpt"
    )
    if converted_record.get("path") != _relative(converted, asset_root):
        raise RuntimeError(f"{spec.name} converted path does not match the contract")
    _require_recorded_file(converted, converted_record, f"{spec.name} checkpoint")
    checkpoint = torch.load(converted, map_location="cpu", weights_only=False)
    embedded = checkpoint.get("asset_provenance", {})
    if (
        embedded.get("repository") != spec.repo_id
        or embedded.get("revision") != spec.revision
        or embedded.get("source_hashes", {}).get("model.safetensors")
        != spec.safetensors_sha256
    ):
        raise RuntimeError(f"{spec.name} embedded provenance does not match its pin")
    state_dict = checkpoint.get("state_dict")
    config = checkpoint.get("hyper_parameters")
    if not isinstance(state_dict, dict) or not isinstance(config, dict):
        raise RuntimeError(f"{spec.name} is not a usable Lightning checkpoint")
    _validate_state_dict(spec, state_dict, config)

    if spec.name == "ijepa-celeba":
        repaired_record = record.get("repaired_checkpoint", {})
        repaired = base / "repaired_checkpoints" / f"epoch_{spec.epoch:04d}.ckpt"
        if repaired_record.get("path") != _relative(repaired, asset_root):
            raise RuntimeError("I-JEPA repaired path does not match the contract")
        _require_recorded_file(repaired, repaired_record, "repaired I-JEPA checkpoint")
        repaired_checkpoint = torch.load(
            repaired, map_location="cpu", weights_only=False
        )
        expected_repair = repair_checkpoint_dict(checkpoint)["legacy_metadata_repair"]
        if repaired_checkpoint.get("legacy_metadata_repair") != expected_repair:
            raise RuntimeError("I-JEPA repair metadata does not match the source")
        repaired_state = repaired_checkpoint.get("state_dict")
        repaired_config = repaired_checkpoint.get("hyper_parameters")
        if not isinstance(repaired_state, dict) or not isinstance(repaired_config, dict):
            raise RuntimeError("Repaired I-JEPA file is not a usable checkpoint")
        _validate_state_dict(spec, repaired_state, repaired_config)

    return record


def _print_plan(names: list[str], asset_root: Path, cache_dir: Path) -> None:
    print(f"Asset root: {asset_root}")
    print(f"Download cache: {cache_dir}")
    for name in names:
        print(f"\n{name}")
        if name in PUBLISHED_CHECKPOINTS:
            spec = PUBLISHED_CHECKPOINTS[name]
            print(f"  source: https://huggingface.co/{spec.repo_id}")
            print(f"  revision: {spec.revision}")
            print(f"  file: {spec.source_filename}")
            print(f"  SHA-256: {spec.safetensors_sha256}")
            for output in _expected_outputs(spec, asset_root):
                print(f"  output: {output}")
        else:
            public = PUBLIC_WEIGHTS[name]
            print(f"  source: {public.url}")
            print(f"  SHA-256: {public.sha256}")
            print(f"  output: {_public_weights_path(public, asset_root)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=Path.cwd(),
        help="Root containing hf_models/ (default: current directory)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Hugging Face cache (default: <asset-root>/cache/huggingface/hub)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=[*ALL_ASSETS, "all"],
        default=["all"],
        help="Published checkpoints to prepare",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print pinned sources and outputs without network access",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify existing outputs, hashes, provenance, and strict model loading",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing generated checkpoints after source validation",
    )
    parser.add_argument(
        "--keep-cache",
        action="store_true",
        help="Keep downloaded safetensors blobs after conversion",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asset_root = args.asset_root.expanduser().resolve()
    cache_dir = (
        args.cache_dir.expanduser().resolve()
        if args.cache_dir is not None
        else asset_root / "cache" / "huggingface" / "hub"
    )
    names = (
        list(ALL_ASSETS)
        if "all" in args.models
        else list(dict.fromkeys(args.models))
    )
    _print_plan(names, asset_root, cache_dir)
    if args.dry_run and args.verify:
        raise SystemExit("Choose either --dry-run or --verify")
    if args.dry_run:
        print("\nDry run only; no files downloaded or written.")
        return
    if args.verify:
        for name in names:
            print(f"\nVerifying {name}...")
            if name in PUBLISHED_CHECKPOINTS:
                _verify_checkpoint(PUBLISHED_CHECKPOINTS[name], asset_root)
            else:
                _verify_public_weights(PUBLIC_WEIGHTS[name], asset_root)
            print(f"Verified {name}.")
        print("\nAll selected model assets are valid.")
        return

    cache_existed = cache_dir.exists()
    records = []
    for name in names:
        print(f"\nPreparing {name}...")
        if name in PUBLISHED_CHECKPOINTS:
            records.append(
                _prepare_checkpoint(
                    PUBLISHED_CHECKPOINTS[name],
                    asset_root,
                    cache_dir,
                    force=args.force,
                )
            )
        else:
            records.append(
                _prepare_public_weights(
                    PUBLIC_WEIGHTS[name], asset_root, force=args.force
                )
            )
        print(f"Validated {name}.")

    manifest = asset_root / "PAPER_MODEL_ASSETS.json"
    _write_json_atomic(
        manifest,
        {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "assets": records,
        },
    )
    print(f"\nWrote combined manifest: {manifest}")

    if not args.keep_cache:
        # Delete only the exact cache created/selected by this command.  Never
        # remove a pre-existing cache tree because it may contain other assets.
        if not cache_existed and cache_dir.is_dir():
            shutil.rmtree(cache_dir)
            print(f"Removed temporary download cache: {cache_dir}")
        elif cache_existed:
            print(f"Retained pre-existing download cache: {cache_dir}")


if __name__ == "__main__":
    main()
