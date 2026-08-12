"""Context-held-out linear transfer on frozen visual representations.

The protocol has three deliberately separate stages:

1. ``prepare`` reads training labels only and freezes attributes, ordered
   target/context pairs, folds, and shot seeds in a hashed manifest.
2. ``cache`` extracts frozen features without fitting any task head.
3. ``evaluate`` verifies the manifest and cache identities before fitting a
   nearest-centroid head in one context and evaluating it in the other.

The encoder may have encountered similar images or attribute combinations
during pretraining.  The measured estimand is therefore labeled-context
transfer on a frozen representation, not whole-model novelty.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

try:
    from . import hyperrect as H
    from .eval_utils import extract_backbone_features, freeze_model, load_model_from_checkpoint
except ImportError:  # direct execution from analysis/
    import hyperrect as H
    from eval_utils import extract_backbone_features, freeze_model, load_model_from_checkpoint


PROTOCOL_VERSION = "context_heldout_ncc_crossfit_v1"
FOLD_ALGORITHM = "seeded_global_permutation_alternating_v1"
SAMPLING_ALGORITHM = "blake2b_keyed_without_replacement_v1"
DEFAULT_SHOT_SEEDS = tuple(range(3101, 3121))

TRANSFER_FIELDS = (
    "dataset",
    "encoder_id",
    "pair_id",
    "target",
    "context",
    "source_context",
    "destination_context",
    "geometry_fold",
    "head_fold",
    "shot",
    "shot_seed",
    "valid",
    "invalid_reason",
    "test_n_00",
    "test_n_10",
    "test_n_01",
    "test_n_11",
    "source_id_balanced_accuracy",
    "source_ood_balanced_accuracy",
    "source_transfer_gap",
    "source_id_auroc",
    "source_ood_auroc",
    "oracle_id_balanced_accuracy",
    "oracle_ood_balanced_accuracy",
    "oracle_transfer_gap",
    "oracle_id_auroc",
    "oracle_ood_auroc",
)

GEOMETRY_FIELDS = (
    "dataset",
    "encoder_id",
    "pair_id",
    "target",
    "context",
    "source_context",
    "destination_context",
    "geometry_fold",
    "head_fold",
    "valid",
    "invalid_reason",
    "n_00",
    "n_10",
    "n_01",
    "n_11",
    "conditional_axis_cosine",
    "conditional_axis_abs_cosine",
    "interaction_defect",
    "interaction_defect_normalized",
    "midpoint_drift_signed",
    "midpoint_drift_abs",
    "transported_margin",
    "target_context_abs_cosine",
    "target_capture_balanced",
    "context_capture_balanced",
    "geometry_fold_phi",
    "train_phi",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(b"|")
    digest.update(",".join(str(value) for value in contiguous.shape).encode("ascii"))
    digest.update(b"|")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _validate_binary_matrix(attributes: np.ndarray) -> np.ndarray:
    attributes = np.asarray(attributes)
    if attributes.ndim != 2:
        raise ValueError("attributes must be a two-dimensional matrix")
    if not np.all(np.isin(attributes, (0, 1))):
        raise ValueError("attributes must contain only binary 0/1 values")
    return np.ascontiguousarray(attributes, dtype=np.uint8)


def make_fold_assignment(n_rows: int, seed: int) -> np.ndarray:
    """Return a deterministic, model-independent two-fold assignment."""
    if n_rows < 2:
        raise ValueError("at least two training rows are required")
    permutation = np.random.default_rng(seed).permutation(n_rows)
    assignment = np.empty(n_rows, dtype=np.uint8)
    assignment[permutation[0::2]] = 0
    assignment[permutation[1::2]] = 1
    return assignment


def _binary_phi(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    denominator = first.std() * second.std()
    if denominator <= 1e-12:
        return float("nan")
    return float(np.mean((first - first.mean()) * (second - second.mean())) / denominator)


def _four_cell_counts(
    attributes: np.ndarray,
    target_index: int,
    context_index: int,
    rows: np.ndarray | None = None,
) -> list[int]:
    if rows is None:
        target = attributes[:, target_index]
        context = attributes[:, context_index]
    else:
        target = attributes[rows, target_index]
        context = attributes[rows, context_index]
    return [
        int(np.sum((target == target_value) & (context == context_value)))
        for context_value in (0, 1)
        for target_value in (0, 1)
    ]


def _attribute_family(name: str) -> str:
    return name.split("=", maxsplit=1)[0]


def select_cub_family_representatives(
    attributes: np.ndarray,
    attribute_names: Sequence[str],
    folds: np.ndarray,
    minimum_per_class_per_fold: int,
) -> tuple[list[int], list[dict[str, Any]]]:
    """Select one train-only, prevalence-balanced value from each CUB family."""
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, name in enumerate(attribute_names):
        grouped[_attribute_family(name)].append(index)

    selected: list[int] = []
    records: list[dict[str, Any]] = []
    for family in sorted(grouped):
        candidates = []
        for index in grouped[family]:
            fold_counts = []
            eligible = True
            for fold in (0, 1):
                values = attributes[folds == fold, index]
                counts = [int(np.sum(values == value)) for value in (0, 1)]
                fold_counts.append(counts)
                eligible = eligible and min(counts) >= minimum_per_class_per_fold
            prevalence = float(attributes[:, index].mean())
            candidates.append(
                {
                    "index": index,
                    "name": attribute_names[index],
                    "prevalence": prevalence,
                    "fold_class_counts": fold_counts,
                    "eligible": eligible,
                }
            )
        eligible_candidates = [row for row in candidates if row["eligible"]]
        chosen = min(
            eligible_candidates,
            key=lambda row: (abs(row["prevalence"] - 0.5), row["name"]),
            default=None,
        )
        if chosen is not None:
            selected.append(int(chosen["index"]))
        records.append(
            {
                "family": family,
                "chosen": None if chosen is None else chosen["name"],
                "chosen_index": None if chosen is None else chosen["index"],
                "candidates": candidates,
            }
        )
    return selected, records


def build_train_manifest(
    *,
    dataset: str,
    attributes: np.ndarray,
    attribute_names: Sequence[str],
    row_ids: np.ndarray,
    fold_seed: int,
    shot_seeds: Sequence[int],
    primary_shot: int,
    minimum_heldout_cell_count: int = 20,
    dataset_source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the complete selection object using training labels only."""
    attributes = _validate_binary_matrix(attributes)
    row_ids = np.ascontiguousarray(row_ids, dtype=np.int64)
    attribute_names = [str(name) for name in attribute_names]
    if attributes.shape[0] != row_ids.shape[0]:
        raise ValueError("attributes and row_ids must contain the same number of rows")
    if np.unique(row_ids).size != row_ids.size:
        raise ValueError("row_ids must be unique")
    if attributes.shape[1] != len(attribute_names):
        raise ValueError("attribute_names must match the attribute matrix width")
    if len(set(attribute_names)) != len(attribute_names):
        raise ValueError("attribute_names must be unique")
    if primary_shot < 2 or primary_shot % 2:
        raise ValueError("primary_shot must be an even integer of at least two")
    if minimum_heldout_cell_count < 1:
        raise ValueError("minimum_heldout_cell_count must be positive")
    shot_seeds = [int(seed) for seed in shot_seeds]
    if not shot_seeds or len(set(shot_seeds)) != len(shot_seeds):
        raise ValueError("shot_seeds must be a nonempty sequence of unique integers")
    if dataset not in {"celeba", "cub200"}:
        raise ValueError(f"unsupported dataset: {dataset}")

    folds = make_fold_assignment(attributes.shape[0], fold_seed)
    if dataset == "cub200":
        selected_indices, selection_records = select_cub_family_representatives(
            attributes,
            attribute_names,
            folds,
            minimum_per_class_per_fold=primary_shot,
        )
        selection_policy = "one_prevalence_balanced_binary_value_per_semantic_family"
    else:
        selected_indices = list(range(len(attribute_names)))
        selection_records = [
            {
                "family": name,
                "chosen": name,
                "chosen_index": index,
                "candidates": [],
            }
            for index, name in enumerate(attribute_names)
        ]
        selection_policy = "all_standard_attributes_with_pairwise_four_cell_support"

    pairs: list[dict[str, Any]] = []
    excluded_pairs: list[dict[str, Any]] = []
    for target_index in selected_indices:
        for context_index in selected_indices:
            if target_index == context_index:
                continue
            counts_by_fold = [
                _four_cell_counts(
                    attributes,
                    target_index,
                    context_index,
                    np.flatnonzero(folds == fold),
                )
                for fold in (0, 1)
            ]
            record = {
                "pair_id": f"a{target_index:03d}_c{context_index:03d}",
                "target_index": int(target_index),
                "target": attribute_names[target_index],
                "context_index": int(context_index),
                "context": attribute_names[context_index],
                "train_cell_counts_by_fold": counts_by_fold,
                "train_phi": _binary_phi(
                    attributes[:, target_index], attributes[:, context_index]
                ),
            }
            minimum = min(value for counts in counts_by_fold for value in counts)
            if minimum >= primary_shot:
                pairs.append(record)
            else:
                excluded_pairs.append(
                    {
                        **record,
                        "reason": "at_least_one_fold_cell_below_primary_shot",
                        "minimum_fold_cell_count": minimum,
                    }
                )

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": dataset,
        "selection_split": "train",
        "heldout_data_accessed": False,
        "claim_scope": "labeled_context_transfer_on_a_frozen_representation",
        "source_commit": _git_commit(),
        "dataset_source": dataset_source or {},
        "n_train": int(attributes.shape[0]),
        "attribute_names": attribute_names,
        "train_attribute_sha256": array_sha256(attributes),
        "train_row_ids_sha256": array_sha256(row_ids),
        "fold_algorithm": FOLD_ALGORITHM,
        "fold_seed": int(fold_seed),
        "fold_assignment": folds.tolist(),
        "fold_assignment_sha256": array_sha256(folds),
        "sampling_algorithm": SAMPLING_ALGORITHM,
        "shot_seeds": shot_seeds,
        "primary_shot_per_target_class": int(primary_shot),
        "oracle_samples_per_target_class_and_context": int(primary_shot // 2),
        "minimum_heldout_cell_count": int(minimum_heldout_cell_count),
        "attribute_selection_policy": selection_policy,
        "attribute_selection": selection_records,
        "selected_attribute_indices": [int(index) for index in selected_indices],
        "eligible_pair_count": len(pairs),
        "excluded_pair_count": len(excluded_pairs),
        "pairs": pairs,
        "excluded_pairs": excluded_pairs,
        "analysis_plan": {
            "primary_shot_per_target_class": int(primary_shot),
            "primary_outcome": "heldout_destination_context_balanced_accuracy",
            "secondary_outcomes": [
                "source_context_balanced_accuracy",
                "source_minus_destination_balanced_accuracy",
                "heldout_destination_context_auroc",
                "same_budget_all_context_reference_accuracy",
            ],
            "averaging": "shot_seeds_then_source_directions_then_crossfit_directions",
            "inferential_resampling_unit": "target_attribute",
            "interval_interpretation": (
                "across_target_empirical_variability_not_iid_image_sampling_inference"
            ),
            "unevaluable_heldout_rule": (
                "mark_invalid_without_replacement_when_any_heldout_pair_cell_"
                f"has_fewer_than_{minimum_heldout_cell_count}_rows"
            ),
            "directional_hypotheses": {
                "conditional_axis_cosine_vs_destination_accuracy": "positive",
                "transported_margin_vs_destination_accuracy": "positive",
                "interaction_defect_vs_transfer_gap": "positive",
                "midpoint_drift_vs_transfer_gap": "positive",
                "absolute_train_phi_vs_transfer_gap": "positive",
            },
            "incremental_prediction_test": (
                "target_clustered_cross_validated_capture_only_vs_"
                "capture_plus_representation_geometry_then_label_dependence"
            ),
        },
    }
    if not pairs:
        raise ValueError("the frozen training rules produced no eligible ordered pairs")
    return manifest


def _write_manifest(path: str | Path, manifest: dict[str, Any]) -> tuple[Path, str]:
    path = _write_json(path, manifest)
    digest = sha256_file(path)
    checksum_path = path.with_suffix(path.suffix + ".sha256")
    checksum_path.write_text(f"{digest}  {path.name}\n", encoding="ascii")
    return path, digest


def _resolve_cub_attribute_names_path(root: Path) -> Path:
    candidates = (
        root / "attributes.txt",
        root / "attributes" / "attributes.txt",
        root.parent / "attributes.txt",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("CUB attribute-name table was not found")


def _load_cub_training_labels(root: str | Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Read CUB training attributes without parsing held-out attribute values."""
    root = Path(root).expanduser().resolve()
    split = np.loadtxt(root / "train_test_split.txt", dtype=np.int64)
    if split.ndim != 2 or split.shape[1] != 2:
        raise ValueError("unexpected CUB split table")
    train_image_ids = split[split[:, 1] == 1, 0].astype(np.int64)
    train_lookup = {int(image_id): row for row, image_id in enumerate(train_image_ids)}

    names_path = _resolve_cub_attribute_names_path(root)
    attribute_names = []
    with names_path.open(encoding="utf-8") as handle:
        for expected, line in enumerate(handle, start=1):
            identifier, value = line.rstrip("\n").split(maxsplit=1)
            if int(identifier) != expected:
                raise ValueError("CUB attribute identifiers must be consecutive")
            attribute_names.append(value.removeprefix("has_").replace("::", "="))

    attributes = np.zeros((len(train_image_ids), len(attribute_names)), dtype=np.uint8)
    seen = np.zeros_like(attributes, dtype=np.uint8)
    labels_path = root / "attributes" / "image_attribute_labels.txt"
    with labels_path.open(encoding="utf-8") as handle:
        for line in handle:
            image_parts = line.split(maxsplit=1)
            image_id = int(image_parts[0])
            train_row = train_lookup.get(image_id)
            if train_row is None:
                continue
            if len(image_parts) != 2:
                raise ValueError("incomplete CUB training attribute row")
            fields = image_parts[1].split()
            attribute_id = int(fields[0]) - 1
            is_present = int(fields[1])
            if is_present not in (0, 1):
                raise ValueError("CUB attribute values must be binary")
            attributes[train_row, attribute_id] = is_present
            seen[train_row, attribute_id] += 1
    if not np.all(seen == 1):
        raise ValueError("each CUB training image/attribute pair must occur exactly once")
    return attributes, attribute_names, train_image_ids


def _standard_celeba_attributes() -> list[str]:
    try:
        from .celeba_hyperrect import CELEBA_40
    except ImportError:
        from celeba_hyperrect import CELEBA_40
    return list(CELEBA_40)


def _load_celeba_split(config_path: str, split_role: str):
    from datasets import load_dataset
    from training.config_loader import load_config

    config = load_config(config_path)
    data = config["data"]
    split_name = data[f"{split_role}_split"]
    dataset = load_dataset(
        data["hf_repo"],
        name=data.get("hf_name"),
        split=split_name,
        cache_dir=data.get("hf_cache_dir"),
    )
    return dataset, data, split_name


def _load_celeba_training_labels(
    config_path: str,
) -> tuple[np.ndarray, list[str], np.ndarray, dict[str, Any]]:
    dataset, data, split_name = _load_celeba_split(config_path, "train")
    available = set(dataset.column_names)
    names = [name for name in _standard_celeba_attributes() if name in available]
    if not names:
        raise ValueError("none of the standard CelebA attributes are available")
    attributes = np.column_stack(
        [np.asarray(dataset[name], dtype=np.uint8) for name in names]
    )
    row_ids = np.arange(len(dataset), dtype=np.int64)
    source = {
        "repository": data["hf_repo"],
        "configuration": data.get("hf_name"),
        "split": split_name,
        "cache_dir": data.get("hf_cache_dir"),
        "analysis_config": str(Path(config_path).expanduser().resolve()),
        "analysis_config_sha256": sha256_file(config_path),
    }
    return attributes, names, row_ids, source


def prepare_command(args: argparse.Namespace) -> None:
    if args.dataset == "celeba":
        attributes, names, row_ids, source = _load_celeba_training_labels(args.config)
    else:
        attributes, names, row_ids = _load_cub_training_labels(args.data_root)
        source = {
            "official_root": str(Path(args.data_root).expanduser().resolve()),
            "split_table": "train_test_split.txt",
            "attribute_table": "attributes/image_attribute_labels.txt",
        }
    manifest = build_train_manifest(
        dataset=args.dataset,
        attributes=attributes,
        attribute_names=names,
        row_ids=row_ids,
        fold_seed=args.fold_seed,
        shot_seeds=args.shot_seeds,
        primary_shot=args.primary_shot,
        minimum_heldout_cell_count=args.minimum_heldout_cell_count,
        dataset_source=source,
    )
    path, digest = _write_manifest(args.output, manifest)
    print(f"Frozen {manifest['eligible_pair_count']} ordered pairs: {path}")
    print(f"SHA256 {digest}")


class _IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        return index, self.dataset[index]


@dataclass
class _CelebACollator:
    transform: Any
    image_key: str
    attribute_names: list[str]

    def __call__(self, batch):
        row_ids, samples = zip(*batch, strict=True)
        images = torch.stack(
            [self.transform(sample[self.image_key].convert("RGB")) for sample in samples]
        )
        attributes = torch.tensor(
            [[int(sample[name]) for name in self.attribute_names] for sample in samples],
            dtype=torch.uint8,
        )
        return images, attributes, torch.tensor(row_ids, dtype=torch.int64)


def _sha256_if_file(path: str | Path | None) -> str | None:
    if path is None:
        return None
    path = Path(path)
    return sha256_file(path) if path.is_file() else None


def _load_encoder(args: argparse.Namespace):
    if args.encoder_kind == "checkpoint":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for encoder-kind=checkpoint")
        model, _ = load_model_from_checkpoint(args.checkpoint)
        freeze_model(model)
        model.eval()
        return model.backbone.to(args.device), {
            "kind": args.encoder_kind,
            "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
            "checkpoint_sha256": _sha256_if_file(args.checkpoint),
        }
    if args.encoder_kind == "vicreg_imagenet1k":
        try:
            from .cub200_hyperrect_crossfit import _load_official_vicreg
        except ImportError:
            from cub200_hyperrect_crossfit import _load_official_vicreg
        model = _load_official_vicreg(args.device)
        return model, {
            "kind": args.encoder_kind,
            "architecture": "resnet50",
            "pretraining": "vicreg_imagenet1k",
            "weights": args.weights_path,
            "weights_sha256": _sha256_if_file(args.weights_path),
        }
    if args.encoder_kind == "supervised_imagenet1k":
        from torchvision.models import ResNet50_Weights, resnet50

        weights = ResNet50_Weights.IMAGENET1K_V1
        model = resnet50(weights=weights)
        model.fc = torch.nn.Identity()
        model.eval().requires_grad_(False)
        return model.to(args.device), {
            "kind": args.encoder_kind,
            "architecture": "resnet50",
            "pretraining": "supervised_imagenet1k_v1",
            "weights_url": weights.url,
            "weights": args.weights_path,
            "weights_sha256": _sha256_if_file(args.weights_path),
        }
    raise ValueError(f"unsupported encoder kind: {args.encoder_kind}")


def _cache_celeba_dataset(args: argparse.Namespace):
    from data_utils import CelebACfg, CelebADataModule
    from training.config_loader import load_config

    dataset, data, split_name = _load_celeba_split(args.config, args.split)
    names = [name for name in _standard_celeba_attributes() if name in dataset.column_names]
    if args.encoder_kind == "checkpoint":
        config = load_config(args.config)
        data_config = dict(config["data"])
        data_config["method"] = config["method"]["name"]
        data_config["batch_size"] = args.batch_size
        data_config["num_workers"] = args.num_workers
        module = CelebADataModule(CelebACfg(**data_config))
        transform = module.test_tfms
        transform_description = f"checkpoint_eval_{data_config['img_size']}px"
    else:
        try:
            from .cub200_hyperrect_crossfit import _eval_transform
        except ImportError:
            from cub200_hyperrect_crossfit import _eval_transform
        transform = _eval_transform(args.image_size)
        transform_description = f"imagenet_eval_resize_center_crop_{args.image_size}px"
    loader = DataLoader(
        _IndexedDataset(dataset),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=_CelebACollator(transform, data["image_key"], names),
    )
    source = {
        "repository": data["hf_repo"],
        "configuration": data.get("hf_name"),
        "split": split_name,
        "analysis_config": str(Path(args.config).expanduser().resolve()),
        "analysis_config_sha256": sha256_file(args.config),
    }
    return loader, names, transform_description, source


def _cache_cub_dataset(args: argparse.Namespace):
    from data_utils import CUB200AttributeDataset, load_cub200_metadata

    try:
        from .cub200_hyperrect_crossfit import _eval_transform
    except ImportError:
        from cub200_hyperrect_crossfit import _eval_transform
    metadata = load_cub200_metadata(args.data_root)
    dataset = CUB200AttributeDataset(
        args.data_root,
        metadata,
        args.split,
        transform=_eval_transform(args.image_size),
        crop_to_bbox=not args.no_crop_to_bbox,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    source = {
        "official_root": str(Path(args.data_root).expanduser().resolve()),
        "split": args.split,
        "crop_to_official_bounding_box": not args.no_crop_to_bbox,
    }
    return (
        loader,
        metadata.attribute_names,
        f"imagenet_eval_resize_center_crop_{args.image_size}px",
        source,
    )


def _validate_existing_cache(path: Path, args: argparse.Namespace) -> bool:
    if not path.is_file():
        return False
    payload = torch.load(path, map_location="cpu", weights_only=True)
    metadata = payload.get("metadata", {})
    expected = {
        "dataset": args.dataset,
        "split": args.split,
        "encoder_id": args.encoder_id,
        "protocol_version": PROTOCOL_VERSION,
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"existing cache identity mismatch: {mismatches}")
    encoder = metadata.get("encoder", {})
    if args.encoder_kind == "checkpoint":
        requested_hash = _sha256_if_file(args.checkpoint)
        if encoder.get("checkpoint_sha256") != requested_hash:
            raise ValueError("existing cache checkpoint hash mismatch")
    elif args.weights_path and Path(args.weights_path).is_file():
        requested_hash = _sha256_if_file(args.weights_path)
        if encoder.get("weights_sha256") != requested_hash:
            raise ValueError("existing cache weight hash mismatch")
    for key in ("features", "attributes", "row_ids"):
        if key not in payload:
            raise ValueError(f"existing cache is missing {key}")
    print(f"Validated existing cache: {path}")
    return True


@torch.no_grad()
def cache_command(args: argparse.Namespace) -> None:
    output = Path(args.output)
    if _validate_existing_cache(output, args):
        return
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch size must be positive and workers nonnegative")
    if args.dataset == "celeba":
        loader, attribute_names, transform_description, source = _cache_celeba_dataset(args)
    else:
        loader, attribute_names, transform_description, source = _cache_cub_dataset(args)
    encoder, encoder_metadata = _load_encoder(args)
    encoder.eval().requires_grad_(False)

    feature_batches = []
    attribute_batches = []
    row_id_batches = []
    for batch in tqdm(loader, desc=f"{args.dataset}/{args.split}/{args.encoder_id}"):
        if args.dataset == "celeba":
            images, attributes, row_ids = batch
        else:
            images = batch["image"]
            attributes = batch["attributes"].to(torch.uint8)
            row_ids = batch["image_id"].to(torch.int64)
        images = images.to(args.device, non_blocking=True)
        features = extract_backbone_features(encoder, images)
        feature_batches.append(F.normalize(features.float(), dim=1).cpu())
        attribute_batches.append(attributes.cpu())
        row_id_batches.append(row_ids.cpu())

    features = torch.cat(feature_batches, dim=0).contiguous()
    attributes = torch.cat(attribute_batches, dim=0).to(torch.uint8).contiguous()
    row_ids = torch.cat(row_id_batches, dim=0).to(torch.int64).contiguous()
    if not bool(torch.isfinite(features).all().item()):
        raise ValueError("extracted features contain non-finite values")
    if torch.unique(row_ids).numel() != row_ids.numel():
        raise ValueError("cache row identifiers must be unique")
    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "source_commit": _git_commit(),
        "dataset": args.dataset,
        "split": args.split,
        "encoder_id": args.encoder_id,
        "n_rows": int(features.shape[0]),
        "feature_dimension": int(features.shape[1]),
        "attribute_names": list(attribute_names),
        "attribute_sha256": array_sha256(attributes.numpy()),
        "row_ids_sha256": array_sha256(row_ids.numpy()),
        "features_l2_normalized": True,
        "transform": transform_description,
        "dataset_source": source,
        "encoder": encoder_metadata,
    }
    payload = {
        "features": features,
        "attributes": attributes,
        "row_ids": row_ids,
        "metadata": metadata,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, output)
    digest = sha256_file(output)
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{digest}  {output.name}\n", encoding="ascii"
    )
    print(f"Saved {tuple(features.shape)} feature cache: {output}")
    print(f"SHA256 {digest}")


def _load_manifest(path: str | Path, expected_sha256: str) -> dict[str, Any]:
    observed = sha256_file(path)
    if not expected_sha256:
        raise ValueError("--manifest-sha256 is required for held-out evaluation")
    if observed.lower() != expected_sha256.lower():
        raise ValueError(
            f"manifest checksum mismatch: expected {expected_sha256}, observed {observed}"
        )
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("manifest protocol version does not match this evaluator")
    if manifest.get("heldout_data_accessed") is not False:
        raise ValueError("manifest does not certify train-only preparation")
    return manifest


def _load_cache(path: str | Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = {"features", "attributes", "row_ids", "metadata"}
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"cache is missing fields: {missing}")
    return payload


def _validate_cache_pair(
    train_cache: dict[str, Any],
    test_cache: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    train_meta = train_cache["metadata"]
    test_meta = test_cache["metadata"]
    for metadata, split in ((train_meta, "train"), (test_meta, "test")):
        if metadata.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError(f"{split} cache protocol version mismatch")
        if metadata.get("dataset") != manifest["dataset"]:
            raise ValueError(f"{split} cache dataset mismatch")
        if metadata.get("split") != split:
            raise ValueError(f"expected a {split} cache")
    if train_meta.get("encoder_id") != test_meta.get("encoder_id"):
        raise ValueError("train and test cache encoder identities differ")
    if train_meta.get("attribute_names") != manifest["attribute_names"]:
        raise ValueError("train cache attribute ordering differs from the manifest")
    if test_meta.get("attribute_names") != manifest["attribute_names"]:
        raise ValueError("test cache attribute ordering differs from the manifest")
    train_attributes = train_cache["attributes"].numpy()
    train_row_ids = train_cache["row_ids"].numpy()
    if array_sha256(train_attributes) != manifest["train_attribute_sha256"]:
        raise ValueError("train cache labels differ from the frozen manifest")
    if array_sha256(train_row_ids) != manifest["train_row_ids_sha256"]:
        raise ValueError("train cache row ordering differs from the frozen manifest")
    if len(manifest["fold_assignment"]) != train_cache["features"].shape[0]:
        raise ValueError("fold assignment length differs from the train cache")


def conditional_geometry(
    features: torch.Tensor,
    target: np.ndarray,
    context: np.ndarray,
    source_context: int,
) -> dict[str, Any]:
    """Compute four-cell geometry and the source-to-destination margin."""
    target = np.asarray(target, dtype=np.uint8)
    context = np.asarray(context, dtype=np.uint8)
    if features.ndim != 2 or features.shape[0] != target.size or target.size != context.size:
        raise ValueError("features and labels must have matching row counts")
    means: dict[tuple[int, int], torch.Tensor] = {}
    counts: dict[tuple[int, int], int] = {}
    for context_value in (0, 1):
        for target_value in (0, 1):
            indices = np.flatnonzero(
                (target == target_value) & (context == context_value)
            )
            counts[(target_value, context_value)] = int(indices.size)
            if indices.size == 0:
                return {
                    "valid": False,
                    "invalid_reason": "empty_geometry_cell",
                    **{
                        f"n_{target_key}{context_key}": counts.get(
                            (target_key, context_key), 0
                        )
                        for context_key in (0, 1)
                        for target_key in (0, 1)
                    },
                }
            index_tensor = torch.as_tensor(indices, device=features.device)
            means[(target_value, context_value)] = features[index_tensor].mean(dim=0)

    mu00 = means[(0, 0)]
    mu10 = means[(1, 0)]
    mu01 = means[(0, 1)]
    mu11 = means[(1, 1)]
    directions = {0: mu10 - mu00, 1: mu11 - mu01}
    context_directions = {0: mu01 - mu00, 1: mu11 - mu10}
    d0, d1 = directions[0], directions[1]
    epsilon = torch.finfo(features.dtype).eps
    norm0 = d0.norm()
    norm1 = d1.norm()
    if float(torch.minimum(norm0, norm1).item()) <= epsilon:
        return {
            "valid": False,
            "invalid_reason": "zero_conditional_target_direction",
            "n_00": counts[(0, 0)],
            "n_10": counts[(1, 0)],
            "n_01": counts[(0, 1)],
            "n_11": counts[(1, 1)],
        }
    cosine = torch.dot(d0, d1) / (norm0 * norm1)
    interaction = mu11 - mu10 - mu01 + mu00
    axis_scale = 0.5 * (norm0 + norm1)
    target_direction = 0.5 * (d0 + d1)
    context_direction = 0.5 * (context_directions[0] + context_directions[1])
    task_cosine = torch.dot(target_direction, context_direction) / (
        target_direction.norm().clamp_min(epsilon)
        * context_direction.norm().clamp_min(epsilon)
    )

    destination_context = 1 - int(source_context)
    source_direction = directions[int(source_context)]
    source_midpoint = 0.5 * (
        means[(0, int(source_context))] + means[(1, int(source_context))]
    )
    destination_midpoint = 0.5 * (
        means[(0, destination_context)] + means[(1, destination_context)]
    )
    source_norm = source_direction.norm().clamp_min(epsilon)
    midpoint_drift = torch.dot(
        source_direction / source_norm,
        destination_midpoint - source_midpoint,
    ) / (0.5 * source_norm)
    destination_positive_margin = torch.dot(
        source_direction,
        means[(1, destination_context)] - source_midpoint,
    )
    destination_negative_margin = torch.dot(
        source_direction,
        source_midpoint - means[(0, destination_context)],
    )
    reference_margin = 0.5 * source_norm.square()
    transported_margin = torch.minimum(
        destination_positive_margin, destination_negative_margin
    ) / reference_margin

    return {
        "valid": True,
        "invalid_reason": "",
        "n_00": counts[(0, 0)],
        "n_10": counts[(1, 0)],
        "n_01": counts[(0, 1)],
        "n_11": counts[(1, 1)],
        "conditional_axis_cosine": float(cosine.item()),
        "conditional_axis_abs_cosine": float(cosine.abs().item()),
        "interaction_defect": float(interaction.norm().item()),
        "interaction_defect_normalized": float((interaction.norm() / axis_scale).item()),
        "midpoint_drift_signed": float(midpoint_drift.item()),
        "midpoint_drift_abs": float(midpoint_drift.abs().item()),
        "transported_margin": float(transported_margin.item()),
        "target_context_abs_cosine": float(task_cosine.abs().item()),
        "target_capture_balanced": float((0.5 * target_direction).square().sum().item()),
        "context_capture_balanced": float((0.5 * context_direction).square().sum().item()),
        "geometry_fold_phi": _binary_phi(target, context),
    }


def _seed_from_key(*parts: Any) -> int:
    encoded = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(encoded, digest_size=8).digest(), "little")


def deterministic_sample_matrix(
    indices: np.ndarray,
    count: int,
    shot_seeds: Sequence[int],
    *key_parts: Any,
) -> np.ndarray:
    """Sample fixed row indices without replacement for every requested seed."""
    indices = np.asarray(indices, dtype=np.int64)
    if count < 1 or indices.size < count:
        raise ValueError(f"cannot sample {count} rows from a pool of {indices.size}")
    sampled = []
    for shot_seed in shot_seeds:
        rng = np.random.default_rng(
            _seed_from_key(SAMPLING_ALGORITHM, *key_parts, int(shot_seed))
        )
        sampled.append(rng.choice(indices, size=count, replace=False))
    return np.stack(sampled, axis=0)


def _centroids_from_samples(
    features: torch.Tensor,
    sample_indices: np.ndarray,
) -> torch.Tensor:
    indices = torch.as_tensor(sample_indices, dtype=torch.long, device=features.device)
    return features[indices].mean(dim=1)


def _ncc_heads(
    negative_centroids: torch.Tensor,
    positive_centroids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    weights = positive_centroids - negative_centroids
    bias = 0.5 * (
        positive_centroids.square().sum(dim=1)
        - negative_centroids.square().sum(dim=1)
    )
    return weights, bias


def _score_metrics(
    features: torch.Tensor,
    labels: np.ndarray,
    weights: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[list[float], list[float]]:
    scores = (features @ weights.T - bias).float()
    labels_tensor = torch.as_tensor(labels, dtype=torch.bool, device=features.device)
    positive = labels_tensor
    negative = ~labels_tensor
    prediction = scores >= 0.0
    balanced = 0.5 * (
        prediction[positive].float().mean(dim=0)
        + (~prediction[negative]).float().mean(dim=0)
    )

    sorted_scores, order = torch.sort(scores, dim=0)
    if bool(torch.any(sorted_scores[1:] == sorted_scores[:-1]).item()):
        from scipy.stats import rankdata

        ranks = torch.as_tensor(
            rankdata(scores.cpu().numpy(), method="average", axis=0),
            dtype=torch.float64,
            device=features.device,
        )
    else:
        base_ranks = torch.arange(
            1,
            scores.shape[0] + 1,
            dtype=torch.float64,
            device=features.device,
        )[:, None]
        ranks = torch.empty_like(scores, dtype=torch.float64)
        ranks.scatter_(0, order, base_ranks.expand_as(scores))
    n_positive = int(positive.sum().item())
    n_negative = int(negative.sum().item())
    positive_rank_sum = ranks[positive].sum(dim=0)
    u_statistic = positive_rank_sum - n_positive * (n_positive + 1) / 2
    aucs = u_statistic / (n_positive * n_negative)
    return balanced.cpu().tolist(), aucs.cpu().tolist()


def evaluate_pair_transfer(
    *,
    head_features: torch.Tensor,
    head_target: np.ndarray,
    head_context: np.ndarray,
    test_features: torch.Tensor,
    test_target: np.ndarray,
    test_context: np.ndarray,
    source_context: int,
    shot: int,
    shot_seeds: Sequence[int],
    sampling_key: Sequence[Any],
    minimum_test_cell_count: int = 1,
) -> list[dict[str, Any]]:
    """Fit source-only and same-budget all-context NCC heads."""
    if shot < 2 or shot % 2:
        raise ValueError("shot must be an even integer of at least two")
    source_context = int(source_context)
    destination_context = 1 - source_context
    pools = {
        (target_value, context_value): np.flatnonzero(
            (head_target == target_value) & (head_context == context_value)
        )
        for context_value in (0, 1)
        for target_value in (0, 1)
    }
    minimum_required = {
        (target_value, context_value): (
            shot if context_value == source_context else shot // 2
        )
        for context_value in (0, 1)
        for target_value in (0, 1)
    }
    invalid = [
        f"cell_{target_value}{context_value}={pools[(target_value, context_value)].size}"
        for (target_value, context_value), required in minimum_required.items()
        if pools[(target_value, context_value)].size < required
    ]
    test_masks = {
        value: np.flatnonzero(test_context == value) for value in (0, 1)
    }
    test_counts = {
        (target_value, context_value): int(
            np.sum(
                (test_target == target_value) & (test_context == context_value)
            )
        )
        for context_value in (0, 1)
        for target_value in (0, 1)
    }
    for value, indices in test_masks.items():
        labels = test_target[indices]
        for target_value in (0, 1):
            count = int(np.sum(labels == target_value))
            if count < minimum_test_cell_count:
                invalid.append(
                    f"test_cell_{target_value}{value}={count}"
                    f"<{minimum_test_cell_count}"
                )
    count_fields = {
        f"test_n_{target_value}{context_value}": test_counts[
            (target_value, context_value)
        ]
        for context_value in (0, 1)
        for target_value in (0, 1)
    }
    if invalid:
        return [
            {
                "shot_seed": int(seed),
                "valid": False,
                "invalid_reason": ";".join(invalid),
                **count_fields,
            }
            for seed in shot_seeds
        ]

    source_centroids = {}
    oracle_centroids = {}
    for target_value in (0, 1):
        source_indices = deterministic_sample_matrix(
            pools[(target_value, source_context)],
            shot,
            shot_seeds,
            *sampling_key,
            "source",
            source_context,
            target_value,
            shot,
        )
        source_centroids[target_value] = _centroids_from_samples(
            head_features, source_indices
        )
        oracle_parts = []
        for context_value in (0, 1):
            oracle_indices = deterministic_sample_matrix(
                pools[(target_value, context_value)],
                shot // 2,
                shot_seeds,
                *sampling_key,
                "oracle",
                context_value,
                target_value,
                shot,
            )
            oracle_parts.append(_centroids_from_samples(head_features, oracle_indices))
        oracle_centroids[target_value] = 0.5 * (oracle_parts[0] + oracle_parts[1])

    source_weights, source_bias = _ncc_heads(
        source_centroids[0], source_centroids[1]
    )
    oracle_weights, oracle_bias = _ncc_heads(
        oracle_centroids[0], oracle_centroids[1]
    )
    weights = torch.cat((source_weights, oracle_weights), dim=0)
    bias = torch.cat((source_bias, oracle_bias), dim=0)
    n_seeds = len(shot_seeds)
    metrics = {}
    for role, context_value in (
        ("id", source_context),
        ("ood", destination_context),
    ):
        indices = test_masks[context_value]
        index_tensor = torch.as_tensor(indices, dtype=torch.long, device=test_features.device)
        balanced, aucs = _score_metrics(
            test_features[index_tensor], test_target[indices], weights, bias
        )
        metrics[role] = {
            "source_balanced": balanced[:n_seeds],
            "source_auc": aucs[:n_seeds],
            "oracle_balanced": balanced[n_seeds:],
            "oracle_auc": aucs[n_seeds:],
        }

    rows = []
    for index, shot_seed in enumerate(shot_seeds):
        source_id = metrics["id"]["source_balanced"][index]
        source_ood = metrics["ood"]["source_balanced"][index]
        oracle_id = metrics["id"]["oracle_balanced"][index]
        oracle_ood = metrics["ood"]["oracle_balanced"][index]
        rows.append(
            {
                "shot_seed": int(shot_seed),
                "valid": True,
                "invalid_reason": "",
                **count_fields,
                "source_id_balanced_accuracy": source_id,
                "source_ood_balanced_accuracy": source_ood,
                "source_transfer_gap": source_id - source_ood,
                "source_id_auroc": metrics["id"]["source_auc"][index],
                "source_ood_auroc": metrics["ood"]["source_auc"][index],
                "oracle_id_balanced_accuracy": oracle_id,
                "oracle_ood_balanced_accuracy": oracle_ood,
                "oracle_transfer_gap": oracle_id - oracle_ood,
                "oracle_id_auroc": metrics["id"]["oracle_auc"][index],
                "oracle_ood_auroc": metrics["ood"]["oracle_auc"][index],
            }
        )
    return rows


def _write_csv(path: str | Path, fields: Sequence[str], rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: ""
                    if row.get(field) is None
                    or (
                        isinstance(row.get(field), float)
                        and not math.isfinite(row[field])
                    )
                    else row.get(field, "")
                    for field in fields
                }
            )


def evaluate_command(args: argparse.Namespace) -> None:
    manifest = _load_manifest(args.manifest, args.manifest_sha256)
    train_cache = _load_cache(args.train_cache)
    test_cache = _load_cache(args.test_cache)
    _validate_cache_pair(train_cache, test_cache, manifest)
    if any(shot < 2 or shot % 2 for shot in args.shots):
        raise ValueError("all --shots values must be even integers of at least two")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    encoder_id = train_cache["metadata"]["encoder_id"]
    dataset = manifest["dataset"]
    train_attributes = train_cache["attributes"].numpy().astype(np.uint8, copy=False)
    test_attributes = test_cache["attributes"].numpy().astype(np.uint8, copy=False)
    folds = np.asarray(manifest["fold_assignment"], dtype=np.uint8)
    shot_seeds = [int(seed) for seed in manifest["shot_seeds"]]
    train_raw = train_cache["features"].to(args.device)
    test_raw = test_cache["features"].to(args.device)

    geometry_rows = []
    transfer_rows = []
    whitening = []
    for geometry_fold in (0, 1):
        head_fold = 1 - geometry_fold
        geometry_indices = np.flatnonzero(folds == geometry_fold)
        head_indices = np.flatnonzero(folds == head_fold)
        geometry_tensor = torch.as_tensor(geometry_indices, device=args.device)
        transform = H.fit_rewhitener(
            train_raw[geometry_tensor], rel_eig_threshold=args.whiten_rel_eig_threshold
        )
        train_features = H.apply_rewhitener(train_raw, transform)
        test_features = H.apply_rewhitener(test_raw, transform)
        whitening.append(
            {
                "geometry_fold": geometry_fold,
                "head_fold": head_fold,
                **transform.metadata(),
            }
        )
        geometry_features = train_features[geometry_tensor]
        head_tensor = torch.as_tensor(head_indices, device=args.device)
        head_features = train_features[head_tensor]
        geometry_attributes = train_attributes[geometry_indices]
        head_attributes = train_attributes[head_indices]

        for pair_number, pair in enumerate(manifest["pairs"], start=1):
            target_index = int(pair["target_index"])
            context_index = int(pair["context_index"])
            for source_context in (0, 1):
                common = {
                    "dataset": dataset,
                    "encoder_id": encoder_id,
                    "pair_id": pair["pair_id"],
                    "target": pair["target"],
                    "context": pair["context"],
                    "source_context": source_context,
                    "destination_context": 1 - source_context,
                    "geometry_fold": geometry_fold,
                    "head_fold": head_fold,
                }
                geometry = conditional_geometry(
                    geometry_features,
                    geometry_attributes[:, target_index],
                    geometry_attributes[:, context_index],
                    source_context,
                )
                geometry_rows.append(
                    {
                        **common,
                        **geometry,
                        "train_phi": pair["train_phi"],
                    }
                )
                for shot in args.shots:
                    results = evaluate_pair_transfer(
                        head_features=head_features,
                        head_target=head_attributes[:, target_index],
                        head_context=head_attributes[:, context_index],
                        test_features=test_features,
                        test_target=test_attributes[:, target_index],
                        test_context=test_attributes[:, context_index],
                        source_context=source_context,
                        shot=shot,
                        shot_seeds=shot_seeds,
                        sampling_key=(
                            dataset,
                            pair["pair_id"],
                            geometry_fold,
                            head_fold,
                        ),
                        minimum_test_cell_count=int(
                            manifest["minimum_heldout_cell_count"]
                        ),
                    )
                    transfer_rows.extend(
                        {**common, "shot": shot, **result} for result in results
                    )
            if pair_number % 25 == 0 or pair_number == len(manifest["pairs"]):
                print(
                    f"{encoder_id}: fold {geometry_fold}, "
                    f"pairs {pair_number}/{len(manifest['pairs'])}",
                    flush=True,
                )
        del train_features, test_features, geometry_features, head_features, transform
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    geometry_path = output / "geometry.csv"
    transfer_path = output / "transfer.csv"
    _write_csv(geometry_path, GEOMETRY_FIELDS, geometry_rows)
    _write_csv(transfer_path, TRANSFER_FIELDS, transfer_rows)
    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "source_commit": _git_commit(),
        "dataset": dataset,
        "encoder_id": encoder_id,
        "manifest": str(Path(args.manifest).resolve()),
        "manifest_sha256": args.manifest_sha256,
        "train_cache": str(Path(args.train_cache).resolve()),
        "train_cache_sha256": sha256_file(args.train_cache),
        "test_cache": str(Path(args.test_cache).resolve()),
        "test_cache_sha256": sha256_file(args.test_cache),
        "train_cache_metadata": train_cache["metadata"],
        "test_cache_metadata": test_cache["metadata"],
        "shots": list(args.shots),
        "shot_seeds": shot_seeds,
        "geometry_rows": len(geometry_rows),
        "transfer_rows": len(transfer_rows),
        "valid_geometry_rows": sum(bool(row["valid"]) for row in geometry_rows),
        "valid_transfer_rows": sum(bool(row["valid"]) for row in transfer_rows),
        "invalid_transfer_reason_counts": dict(
            sorted(
                Counter(
                    row["invalid_reason"]
                    for row in transfer_rows
                    if not row["valid"]
                ).items()
            )
        ),
        "whitening": whitening,
        "estimand": "frozen_representation_labeled_context_transfer",
        "head": "linear_nearest_centroid",
        "oracle": "same_total_label_budget_balanced_across_both_contexts",
        "geometry_definitions": {
            "conditional_axis_cosine": "cosine_between_target_mean_differences_at_context_0_and_1",
            "interaction_defect": "norm_of_mu11_minus_mu10_minus_mu01_plus_mu00",
            "interaction_defect_normalized": "interaction_defect_divided_by_mean_conditional_target_gap",
            "midpoint_drift": "source_normal_projection_of_context_midpoint_shift_divided_by_source_half_gap",
            "transported_margin": "minimum_destination_cell_margin_divided_by_source_centroid_margin",
            "balanced_capture": (
                "squared_balanced_four_cell_task_moment_in_a_natural_train_"
                "fold_whitener_not_a_theorem_identity_for_the_balanced_population"
            ),
        },
        "frozen_analysis_plan": manifest["analysis_plan"],
        "frozen_design_summary": {
            "eligible_pair_count": manifest["eligible_pair_count"],
            "excluded_pair_count": manifest["excluded_pair_count"],
            "fold_seed": manifest["fold_seed"],
            "fold_assignment_sha256": manifest["fold_assignment_sha256"],
            "sampling_algorithm": manifest["sampling_algorithm"],
            "minimum_heldout_cell_count": manifest[
                "minimum_heldout_cell_count"
            ],
        },
    }
    metadata_path = _write_json(output / "metadata.json", metadata)
    checksums = {
        path.name: sha256_file(path)
        for path in (geometry_path, transfer_path, metadata_path)
    }
    (output / "SHA256SUMS").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(checksums.items())),
        encoding="ascii",
    )
    print(f"Saved evaluation: {output}")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in {"", None} else float("nan")


def _mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    return float(np.nanmean(array)) if array.size and np.any(np.isfinite(array)) else float("nan")


def _percentile_interval(values: Sequence[float]) -> list[float]:
    finite = np.asarray([value for value in values if math.isfinite(value)])
    if finite.size == 0:
        return [float("nan"), float("nan")]
    return [float(value) for value in np.percentile(finite, (2.5, 97.5))]


def _cluster_bootstrap_mean(
    rows: Sequence[dict[str, Any]],
    metric: str,
    seed: int,
    repetitions: int,
) -> tuple[float, list[float]]:
    by_target: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = float(row[metric])
        if math.isfinite(value):
            by_target[row["target"]].append(value)
    target_means = {
        target: float(np.mean(values)) for target, values in by_target.items() if values
    }
    observed = _mean(target_means.values())
    targets = sorted(target_means)
    if not targets:
        return observed, [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(repetitions):
        sampled = rng.choice(targets, size=len(targets), replace=True)
        draws.append(float(np.mean([target_means[target] for target in sampled])))
    return observed, _percentile_interval(draws)


def _cluster_bootstrap_spearman(
    rows: Sequence[dict[str, Any]],
    x_key: str,
    y_key: str,
    seed: int,
    repetitions: int,
) -> tuple[float, list[float]]:
    from scipy.stats import spearmanr

    def statistic(x_values, y_values):
        x_array = np.asarray(x_values, dtype=np.float64)
        y_array = np.asarray(y_values, dtype=np.float64)
        if np.unique(x_array).size < 2 or np.unique(y_array).size < 2:
            return float("nan")
        return float(spearmanr(x_array, y_array).statistic)

    by_target: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        x_value = float(row[x_key])
        y_value = float(row[y_key])
        if math.isfinite(x_value) and math.isfinite(y_value):
            by_target[row["target"]].append((x_value, y_value))
    target_rows = [
        (
            target,
            float(np.mean([value[0] for value in values])),
            float(np.mean([value[1] for value in values])),
        )
        for target, values in sorted(by_target.items())
        if values
    ]
    if len(target_rows) < 3:
        return float("nan"), [float("nan"), float("nan")]
    observed = statistic(
        [row[1] for row in target_rows],
        [row[2] for row in target_rows],
    )
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(repetitions):
        sampled_indices = rng.integers(0, len(target_rows), size=len(target_rows))
        sampled_statistic = statistic(
            [target_rows[index][1] for index in sampled_indices],
            [target_rows[index][2] for index in sampled_indices],
        )
        if math.isfinite(sampled_statistic):
            draws.append(sampled_statistic)
    return observed, _percentile_interval(draws)


def _aggregate_evaluation(directory: Path, primary_shot: int) -> tuple[dict, list[dict]]:
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    geometry = _read_csv(directory / "geometry.csv")
    all_transfer = _read_csv(directory / "transfer.csv")
    transfer = [
        row
        for row in all_transfer
        if int(row["shot"]) == primary_shot and row["valid"] == "True"
    ]
    metadata = {
        **metadata,
        "summary_primary_shot": int(primary_shot),
        "primary_shot_transfer_rows": sum(
            int(row["shot"]) == primary_shot for row in all_transfer
        ),
        "primary_shot_valid_transfer_rows": len(transfer),
    }
    geometry_by_key = {
        (row["pair_id"], row["geometry_fold"], row["source_context"]): row
        for row in geometry
        if row["valid"] == "True"
    }
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in transfer:
        grouped[(row["pair_id"], row["geometry_fold"], row["source_context"])].append(row)
    aggregate = []
    for key, replicate_rows in grouped.items():
        geometry_row = geometry_by_key.get(key)
        if geometry_row is None:
            continue
        record: dict[str, Any] = {
            "dataset": metadata["dataset"],
            "encoder_id": metadata["encoder_id"],
            "pair_id": replicate_rows[0]["pair_id"],
            "target": replicate_rows[0]["target"],
            "context": replicate_rows[0]["context"],
            "source_context": int(replicate_rows[0]["source_context"]),
            "geometry_fold": int(replicate_rows[0]["geometry_fold"]),
        }
        for metric in (
            "source_id_balanced_accuracy",
            "source_ood_balanced_accuracy",
            "source_transfer_gap",
            "source_id_auroc",
            "source_ood_auroc",
            "oracle_id_balanced_accuracy",
            "oracle_ood_balanced_accuracy",
            "oracle_transfer_gap",
        ):
            record[metric] = _mean(_float(row, metric) for row in replicate_rows)
        record["all_context_ood_gain"] = (
            record["oracle_ood_balanced_accuracy"]
            - record["source_ood_balanced_accuracy"]
        )
        for metric in (
            "conditional_axis_cosine",
            "conditional_axis_abs_cosine",
            "interaction_defect_normalized",
            "midpoint_drift_abs",
            "transported_margin",
            "target_context_abs_cosine",
            "target_capture_balanced",
            "context_capture_balanced",
            "geometry_fold_phi",
            "train_phi",
        ):
            record[metric] = _float(geometry_row, metric)
        record["abs_train_phi"] = abs(record["train_phi"])
        aggregate.append(record)
    return metadata, aggregate


def _target_level_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    numeric_keys = (
        "source_id_balanced_accuracy",
        "source_ood_balanced_accuracy",
        "source_transfer_gap",
        "oracle_ood_balanced_accuracy",
        "all_context_ood_gain",
        "target_capture_balanced",
        "conditional_axis_cosine",
        "interaction_defect_normalized",
        "midpoint_drift_abs",
        "transported_margin",
        "target_context_abs_cosine",
        "abs_train_phi",
    )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["target"]].append(row)
    return [
        {
            "target": target,
            **{
                key: _mean(float(row.get(key, float("nan"))) for row in target_rows)
                for key in numeric_keys
            },
        }
        for target, target_rows in sorted(grouped.items())
    ]


def _cross_validated_linear_score(
    rows: Sequence[dict[str, Any]],
    predictors: Sequence[str],
    seed: int,
) -> dict[str, float]:
    return _cross_validated_target_score(_target_level_rows(rows), predictors, seed)


def _cross_validated_target_score(
    target_rows: Sequence[dict[str, Any]],
    predictors: Sequence[str],
    seed: int,
) -> dict[str, float]:
    if len(target_rows) < max(8, len(predictors) + 2):
        return {"r2": float("nan"), "mae": float("nan"), "n_targets": len(target_rows)}
    features = np.asarray(
        [[float(row[key]) for key in predictors] for row in target_rows],
        dtype=np.float64,
    )
    outcome = np.asarray(
        [float(row["source_ood_balanced_accuracy"]) for row in target_rows],
        dtype=np.float64,
    )
    valid = np.isfinite(features).all(axis=1) & np.isfinite(outcome)
    features = features[valid]
    outcome = outcome[valid]
    if outcome.size < max(8, len(predictors) + 2):
        return {"r2": float("nan"), "mae": float("nan"), "n_targets": int(outcome.size)}
    permutation = np.random.default_rng(seed).permutation(outcome.size)
    folds = np.empty(outcome.size, dtype=np.int64)
    folds[permutation] = np.arange(outcome.size) % min(5, outcome.size)
    prediction = np.full(outcome.size, np.nan, dtype=np.float64)
    for fold in np.unique(folds):
        train = folds != fold
        heldout = ~train
        mean = features[train].mean(axis=0)
        scale = features[train].std(axis=0)
        scale[scale <= 1e-12] = 1.0
        x_train = (features[train] - mean) / scale
        x_heldout = (features[heldout] - mean) / scale
        design_train = np.column_stack((np.ones(int(train.sum())), x_train))
        design_heldout = np.column_stack((np.ones(int(heldout.sum())), x_heldout))
        coefficients = np.linalg.lstsq(design_train, outcome[train], rcond=None)[0]
        prediction[heldout] = design_heldout @ coefficients
    residual = float(np.sum((outcome - prediction) ** 2))
    total = float(np.sum((outcome - outcome.mean()) ** 2))
    return {
        "r2": float(1.0 - residual / total) if total > 1e-12 else float("nan"),
        "mae": float(np.mean(np.abs(outcome - prediction))),
        "n_targets": int(outcome.size),
    }


def _predictive_increment_robustness(
    rows: Sequence[dict[str, Any]],
    base_predictors: Sequence[str],
    augmented_predictors: Sequence[str],
    *,
    seed: int,
    repetitions: int,
    permutations: int,
) -> dict[str, Any]:
    """Repeated target-fold CV plus a target-level added-predictor null.

    The split interval measures sensitivity to target-fold assignment; it is not
    an iid-image confidence interval. The permutation test is conditional on the
    first, preregistered target-fold assignment so its observed and null statistics
    use the same folds. It jointly shuffles the residualized predictor block added
    to ``base_predictors`` while leaving the base block and OOD outcome paired.
    Residualization preserves the added block's linear relation with the base
    block. This asks whether the remaining added-block variation carries
    target-level signal beyond the frozen base block.
    """
    if repetitions < 1 or permutations < 1:
        raise ValueError("predictive repetitions and permutations must be positive")
    target_rows = _target_level_rows(rows)
    added_predictors = [
        predictor for predictor in augmented_predictors if predictor not in base_predictors
    ]
    if not added_predictors:
        raise ValueError("augmented predictors must add at least one field")
    required = (*augmented_predictors, "source_ood_balanced_accuracy")
    target_rows = [
        row
        for row in target_rows
        if all(math.isfinite(float(row[key])) for key in required)
    ]

    base_scores = []
    augmented_scores = []
    increments = []
    base_mae = []
    augmented_mae = []
    for repetition in range(repetitions):
        fold_seed = seed + repetition * 104729
        base = _cross_validated_target_score(target_rows, base_predictors, fold_seed)
        augmented = _cross_validated_target_score(
            target_rows, augmented_predictors, fold_seed
        )
        if math.isfinite(base["r2"]) and math.isfinite(augmented["r2"]):
            base_scores.append(base["r2"])
            augmented_scores.append(augmented["r2"])
            increments.append(augmented["r2"] - base["r2"])
            base_mae.append(base["mae"])
            augmented_mae.append(augmented["mae"])

    observed_increment = _mean(increments)
    permutation_observed_increment = (
        increments[0] if increments else float("nan")
    )
    rng = np.random.default_rng(seed + 999983)
    base_matrix = np.asarray(
        [[float(row[key]) for key in base_predictors] for row in target_rows],
        dtype=np.float64,
    )
    added_matrix = np.asarray(
        [[float(row[key]) for key in added_predictors] for row in target_rows],
        dtype=np.float64,
    )
    residual_design = np.column_stack(
        (np.ones(len(target_rows), dtype=np.float64), base_matrix)
    )
    residual_coefficients = np.linalg.lstsq(
        residual_design, added_matrix, rcond=None
    )[0]
    added_fitted = residual_design @ residual_coefficients
    added_residual = added_matrix - added_fitted
    null_increments = []
    for _ in range(permutations):
        order = rng.permutation(len(target_rows))
        permuted_rows = []
        for row_index, row in enumerate(target_rows):
            permuted = dict(row)
            for predictor_index, predictor in enumerate(added_predictors):
                permuted[predictor] = float(
                    added_fitted[row_index, predictor_index]
                    + added_residual[int(order[row_index]), predictor_index]
                )
            permuted_rows.append(permuted)
        fold_seed = seed
        base = _cross_validated_target_score(
            permuted_rows, base_predictors, fold_seed
        )
        augmented = _cross_validated_target_score(
            permuted_rows, augmented_predictors, fold_seed
        )
        if math.isfinite(base["r2"]) and math.isfinite(augmented["r2"]):
            null_increments.append(augmented["r2"] - base["r2"])
    p_value = (
        (
            1
            + sum(
                value >= permutation_observed_increment
                for value in null_increments
            )
        )
        / (1 + len(null_increments))
        if math.isfinite(permutation_observed_increment) and null_increments
        else float("nan")
    )
    increment_interval = _percentile_interval(increments)
    return {
        "base_r2_mean": _mean(base_scores),
        "augmented_r2_mean": _mean(augmented_scores),
        "r2_increment_mean": observed_increment,
        "r2_increment_split_low": increment_interval[0],
        "r2_increment_split_high": increment_interval[1],
        "base_mae_mean": _mean(base_mae),
        "augmented_mae_mean": _mean(augmented_mae),
        "permutation_p": p_value,
        "permutation_observed_r2_increment": permutation_observed_increment,
        "permutation_fold_seed": seed,
        "null_permutations": len(null_increments),
        "split_repetitions": len(increments),
        "n_targets": len(target_rows),
    }


def _paired_target_difference(
    first_rows: Sequence[dict[str, Any]],
    second_rows: Sequence[dict[str, Any]],
    metric: str,
    seed: int,
    repetitions: int,
) -> dict[str, Any]:
    def target_means(rows):
        grouped: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            value = float(row[metric])
            if math.isfinite(value):
                grouped[row["target"]].append(value)
        return {target: float(np.mean(values)) for target, values in grouped.items()}

    first = target_means(first_rows)
    second = target_means(second_rows)
    targets = sorted(set(first).intersection(second))
    if not targets:
        return {
            "mean_difference": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_paired_targets": 0,
        }
    differences = np.asarray([first[target] - second[target] for target in targets])
    rng = np.random.default_rng(seed)
    draws = [
        float(np.mean(rng.choice(differences, size=differences.size, replace=True)))
        for _ in range(repetitions)
    ]
    interval = _percentile_interval(draws)
    return {
        "mean_difference": float(differences.mean()),
        "ci_low": interval[0],
        "ci_high": interval[1],
        "n_paired_targets": len(targets),
    }


MODEL_DISPLAY_NAMES = {
    "ijepa_celeba_epoch1000": "I-JEPA / CelebA",
    "supervised_imagenet1k_resnet50": "supervised / ImageNet",
    "vicreg_celeba_epoch1000": "VICReg / CelebA",
    "vicreg_imagenet1k_resnet50": "VICReg / ImageNet",
}


def _display_model(encoder_id: str) -> str:
    return MODEL_DISPLAY_NAMES.get(encoder_id, encoder_id.replace("_", " "))


def _bootstrap_scalar_values(
    values: Sequence[float], seed: int, repetitions: int
) -> tuple[float, list[float]]:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return float("nan"), [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    draws = [
        float(np.mean(rng.choice(finite, size=finite.size, replace=True)))
        for _ in range(repetitions)
    ]
    return float(finite.mean()), _percentile_interval(draws)


def _model_selection_rows(
    by_model: dict[str, list[dict[str, Any]]],
    *,
    seed: int,
    repetitions: int,
) -> list[dict[str, Any]]:
    """Evaluate transparent train-geometry model-selection rules by target."""
    if len(by_model) < 2:
        return []
    target_rows = {
        model: {row["target"]: row for row in _target_level_rows(rows)}
        for model, rows in by_model.items()
    }
    targets = sorted(set.intersection(*(set(rows) for rows in target_rows.values())))
    models = sorted(target_rows)
    if not targets:
        return []
    fixed_oracle_model = max(
        models,
        key=lambda model: _mean(
            target_rows[model][target]["source_ood_balanced_accuracy"]
            for target in targets
        ),
    )
    rules = (
        ("random_model_expectation", None, False),
        ("maximum_train_capture", "target_capture_balanced", False),
        ("maximum_axis_alignment", "conditional_axis_cosine", False),
        ("maximum_transported_margin", "transported_margin", False),
        ("oracle_best_single_model", None, True),
        ("oracle_best_model_per_target", None, True),
    )
    oracle_by_target = {
        target: max(
            target_rows[model][target]["source_ood_balanced_accuracy"]
            for model in models
        )
        for target in targets
    }
    output = []
    for rule_index, (rule, selection_metric, uses_heldout) in enumerate(rules):
        values = []
        regrets = []
        counts: Counter[str] = Counter()
        for target in targets:
            if rule == "random_model_expectation":
                value = _mean(
                    target_rows[model][target]["source_ood_balanced_accuracy"]
                    for model in models
                )
                counts["expected_uniform_mixture"] += 1
            elif rule == "oracle_best_single_model":
                value = target_rows[fixed_oracle_model][target][
                    "source_ood_balanced_accuracy"
                ]
                counts[fixed_oracle_model] += 1
            elif rule == "oracle_best_model_per_target":
                selected = max(
                    models,
                    key=lambda model: target_rows[model][target][
                        "source_ood_balanced_accuracy"
                    ],
                )
                value = target_rows[selected][target]["source_ood_balanced_accuracy"]
                counts[selected] += 1
            else:
                selected = max(
                    models,
                    key=lambda model: target_rows[model][target][selection_metric],
                )
                value = target_rows[selected][target]["source_ood_balanced_accuracy"]
                counts[selected] += 1
            values.append(float(value))
            regrets.append(float(oracle_by_target[target] - value))
        value_mean, value_interval = _bootstrap_scalar_values(
            values, seed + rule_index * 101, repetitions
        )
        regret_mean, regret_interval = _bootstrap_scalar_values(
            regrets, seed + rule_index * 101 + 1, repetitions
        )
        output.append(
            {
                "dataset": by_model[models[0]][0]["dataset"],
                "rule": rule,
                "selection_metric": selection_metric or "",
                "uses_heldout_outcome": uses_heldout,
                "ood_balanced_accuracy": value_mean,
                "ood_balanced_accuracy_ci_low": value_interval[0],
                "ood_balanced_accuracy_ci_high": value_interval[1],
                "regret_to_per_target_oracle": regret_mean,
                "regret_ci_low": regret_interval[0],
                "regret_ci_high": regret_interval[1],
                "n_targets": len(targets),
                "n_models": len(models),
                "selected_model_counts": json.dumps(dict(sorted(counts.items()))),
                "resampling_unit": "target_attribute",
            }
        )
    return output


DEPENDENCE_STRATA = (
    ("low", 0.0, 0.1),
    ("moderate", 0.1, 0.3),
    ("high", 0.3, float("inf")),
)


def _dependence_stratum(value: float) -> str | None:
    if not math.isfinite(value):
        return None
    for name, lower, upper in DEPENDENCE_STRATA:
        if lower <= value < upper:
            return name
    return None


def _dependence_summary_rows(
    by_model: dict[str, list[dict[str, Any]]],
    *,
    seed: int,
    repetitions: int,
) -> list[dict[str, Any]]:
    metrics = (
        "conditional_axis_cosine",
        "target_context_abs_cosine",
        "interaction_defect_normalized",
        "source_ood_balanced_accuracy",
    )
    output = []
    for model_index, encoder_id in enumerate(sorted(by_model)):
        rows = by_model[encoder_id]
        for stratum_index, (stratum, lower, upper) in enumerate(DEPENDENCE_STRATA):
            selected = [
                row
                for row in rows
                if _dependence_stratum(float(row["abs_train_phi"])) == stratum
            ]
            for metric_index, metric in enumerate(metrics):
                mean, interval = _cluster_bootstrap_mean(
                    selected,
                    metric,
                    seed=(
                        seed
                        + model_index * 1009
                        + stratum_index * 101
                        + metric_index
                    ),
                    repetitions=repetitions,
                )
                output.append(
                    {
                        "dataset": rows[0]["dataset"],
                        "encoder_id": encoder_id,
                        "stratum": stratum,
                        "stratum_order": stratum_index,
                        "abs_train_phi_lower": lower,
                        "abs_train_phi_upper": upper if math.isfinite(upper) else "",
                        "metric": metric,
                        "mean": mean,
                        "ci_low": interval[0],
                        "ci_high": interval[1],
                        "n_pair_source_fold_rows": len(selected),
                        "n_targets": len({row["target"] for row in selected}),
                        "resampling_unit": "target_attribute",
                    }
                )
    return output


def _shot_sensitivity_rows(
    evaluations: Sequence[Path],
    *,
    seed: int,
    repetitions: int,
) -> list[dict[str, Any]]:
    output = []
    for model_index, directory in enumerate(evaluations):
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        transfer_rows = _read_csv(directory / "transfer.csv")
        available_shots = sorted({int(row["shot"]) for row in transfer_rows})
        for shot_index, shot in enumerate(available_shots):
            _, rows = _aggregate_evaluation(directory, shot)
            if not rows:
                continue
            record: dict[str, Any] = {
                "dataset": metadata["dataset"],
                "encoder_id": metadata["encoder_id"],
                "shot": shot,
                "valid_transfer_replicates": sum(
                    int(row["shot"]) == shot and row["valid"] == "True"
                    for row in transfer_rows
                ),
                "total_transfer_replicates": sum(
                    int(row["shot"]) == shot for row in transfer_rows
                ),
                "target_attribute_count": len({row["target"] for row in rows}),
            }
            for metric_index, metric in enumerate(
                (
                    "source_id_balanced_accuracy",
                    "source_ood_balanced_accuracy",
                    "source_transfer_gap",
                    "oracle_ood_balanced_accuracy",
                )
            ):
                mean, interval = _cluster_bootstrap_mean(
                    rows,
                    metric,
                    seed=(
                        seed
                        + model_index * 1009
                        + shot_index * 101
                        + metric_index
                    ),
                    repetitions=repetitions,
                )
                record[metric] = mean
                record[f"{metric}_ci_low"] = interval[0]
                record[f"{metric}_ci_high"] = interval[1]
            output.append(record)
    return output


def summarize_command(args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        from .plot_style import apply_style
    except ImportError:
        from plot_style import apply_style

    apply_style()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    evaluations = [Path(path) for path in args.evaluations]
    all_rows = []
    metadata_records = []
    for directory in evaluations:
        metadata, rows = _aggregate_evaluation(directory, args.primary_shot)
        metadata_records.append(metadata)
        all_rows.extend(rows)
    if not all_rows:
        raise ValueError("no valid primary-shot rows were found")
    datasets = {metadata["dataset"] for metadata in metadata_records}
    encoder_ids = [metadata["encoder_id"] for metadata in metadata_records]
    if len(datasets) != 1:
        raise ValueError(f"summary inputs mix datasets: {sorted(datasets)}")
    if len(set(encoder_ids)) != len(encoder_ids):
        raise ValueError("summary inputs contain duplicate encoder identities")

    model_rows = []
    association_rows = []
    predictive_increment_rows = []
    paired_comparison_rows = []
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    metadata_by_model = {
        metadata["encoder_id"]: metadata for metadata in metadata_records
    }
    for row in all_rows:
        by_model[row["encoder_id"]].append(row)
    for model_index, encoder_id in enumerate(sorted(by_model)):
        rows = by_model[encoder_id]
        summary: dict[str, Any] = {
            "encoder_id": encoder_id,
            "dataset": rows[0]["dataset"],
            "target_attribute_count": len({row["target"] for row in rows}),
            "pair_source_fold_rows": len(rows),
            "valid_transfer_replicates": metadata_by_model[encoder_id][
                "primary_shot_valid_transfer_rows"
            ],
            "total_transfer_replicates": metadata_by_model[encoder_id][
                "primary_shot_transfer_rows"
            ],
            "valid_geometry_rows": metadata_by_model[encoder_id][
                "valid_geometry_rows"
            ],
            "total_geometry_rows": metadata_by_model[encoder_id][
                "geometry_rows"
            ],
        }
        for metric in (
            "source_id_balanced_accuracy",
            "source_ood_balanced_accuracy",
            "source_transfer_gap",
            "oracle_ood_balanced_accuracy",
            "all_context_ood_gain",
            "conditional_axis_cosine",
            "interaction_defect_normalized",
            "transported_margin",
        ):
            mean, interval = _cluster_bootstrap_mean(
                rows,
                metric,
                seed=args.bootstrap_seed + model_index * 101,
                repetitions=args.bootstrap_repetitions,
            )
            summary[metric] = mean
            summary[f"{metric}_ci_low"] = interval[0]
            summary[f"{metric}_ci_high"] = interval[1]
        model_rows.append(summary)

        capture_predictors = ("target_capture_balanced",)
        geometry_predictors = (
            "target_capture_balanced",
            "conditional_axis_cosine",
            "interaction_defect_normalized",
            "midpoint_drift_abs",
            "transported_margin",
            "target_context_abs_cosine",
        )
        full_predictors = (*geometry_predictors, "abs_train_phi")
        geometry_increment = _predictive_increment_robustness(
            rows,
            capture_predictors,
            geometry_predictors,
            seed=args.bootstrap_seed + model_index * 101,
            repetitions=args.predictive_cv_repetitions,
            permutations=args.predictive_null_permutations,
        )
        dependence_increment = _predictive_increment_robustness(
            rows,
            geometry_predictors,
            full_predictors,
            seed=args.bootstrap_seed + model_index * 101 + 17,
            repetitions=args.predictive_cv_repetitions,
            permutations=args.predictive_null_permutations,
        )
        predictive_increment_rows.append(
            {
                "dataset": rows[0]["dataset"],
                "encoder_id": encoder_id,
                "unit": "target_attribute_after_within_target_averaging",
                "folds": 5,
                "split_repetitions": geometry_increment["split_repetitions"],
                "capture_only_r2": geometry_increment["base_r2_mean"],
                "capture_plus_representation_geometry_r2": geometry_increment[
                    "augmented_r2_mean"
                ],
                "representation_geometry_r2_increment": geometry_increment[
                    "r2_increment_mean"
                ],
                "geometry_increment_split_low": geometry_increment[
                    "r2_increment_split_low"
                ],
                "geometry_increment_split_high": geometry_increment[
                    "r2_increment_split_high"
                ],
                "geometry_increment_permutation_p": geometry_increment[
                    "permutation_p"
                ],
                "geometry_permutation_observed_increment": geometry_increment[
                    "permutation_observed_r2_increment"
                ],
                "geometry_permutation_fold_seed": geometry_increment[
                    "permutation_fold_seed"
                ],
                "capture_plus_geometry_and_dependence_r2": dependence_increment[
                    "augmented_r2_mean"
                ],
                "dependence_r2_increment_beyond_geometry": dependence_increment[
                    "r2_increment_mean"
                ],
                "dependence_increment_split_low": dependence_increment[
                    "r2_increment_split_low"
                ],
                "dependence_increment_split_high": dependence_increment[
                    "r2_increment_split_high"
                ],
                "dependence_increment_permutation_p": dependence_increment[
                    "permutation_p"
                ],
                "dependence_permutation_observed_increment": dependence_increment[
                    "permutation_observed_r2_increment"
                ],
                "dependence_permutation_fold_seed": dependence_increment[
                    "permutation_fold_seed"
                ],
                "capture_only_mae": geometry_increment["base_mae_mean"],
                "capture_plus_representation_geometry_mae": geometry_increment[
                    "augmented_mae_mean"
                ],
                "capture_plus_geometry_and_dependence_mae": dependence_increment[
                    "augmented_mae_mean"
                ],
                "null_permutations": geometry_increment["null_permutations"],
                "n_targets": geometry_increment["n_targets"],
            }
        )

        associations = (
            ("conditional_axis_cosine", "source_ood_balanced_accuracy"),
            ("transported_margin", "source_ood_balanced_accuracy"),
            ("target_context_abs_cosine", "source_ood_balanced_accuracy"),
            ("interaction_defect_normalized", "source_transfer_gap"),
            ("midpoint_drift_abs", "source_transfer_gap"),
            ("abs_train_phi", "source_transfer_gap"),
            ("target_capture_balanced", "source_ood_balanced_accuracy"),
        )
        for association_index, (x_key, y_key) in enumerate(associations):
            statistic, interval = _cluster_bootstrap_spearman(
                rows,
                x_key,
                y_key,
                seed=args.bootstrap_seed + model_index * 101 + association_index + 1,
                repetitions=args.bootstrap_repetitions,
            )
            association_rows.append(
                {
                    "dataset": rows[0]["dataset"],
                    "encoder_id": encoder_id,
                    "x": x_key,
                    "y": y_key,
                    "spearman": statistic,
                    "ci_low": interval[0],
                    "ci_high": interval[1],
                    "cluster_unit": "target_attribute_after_within_target_averaging",
                }
            )

    comparison_specs = (
        (
            "vicreg_imagenet1k_resnet50",
            "supervised_imagenet1k_resnet50",
            "architecture_and_pretraining_dataset_matched_objective_recipe_"
            "comparison",
        ),
        (
            "vicreg_celeba_epoch1000",
            "ijepa_celeba_epoch1000",
            "descriptive_local_checkpoint_comparison",
        ),
        (
            "vicreg_celeba_epoch1000",
            "vicreg_imagenet1k_resnet50",
            "architecture_and_objective_matched_pretraining_dataset_and_recipe_"
            "comparison",
        ),
    )
    for comparison_index, (first_model, second_model, interpretation) in enumerate(
        comparison_specs
    ):
        if first_model not in by_model or second_model not in by_model:
            continue
        for metric_index, metric in enumerate(
            ("source_ood_balanced_accuracy", "source_transfer_gap")
        ):
            comparison = _paired_target_difference(
                by_model[first_model],
                by_model[second_model],
                metric,
                seed=(
                    args.bootstrap_seed
                    + 10001
                    + comparison_index * 101
                    + metric_index
                ),
                repetitions=args.bootstrap_repetitions,
            )
            paired_comparison_rows.append(
                {
                    "dataset": by_model[first_model][0]["dataset"],
                    "first_model": first_model,
                    "second_model": second_model,
                    "difference": "first_minus_second",
                    "metric": metric,
                    "interpretation": interpretation,
                    **comparison,
                    "resampling_unit": "paired_target_attribute",
                }
            )

    dependence_rows = _dependence_summary_rows(
        by_model,
        seed=args.bootstrap_seed + 20011,
        repetitions=args.bootstrap_repetitions,
    )
    model_selection_rows = _model_selection_rows(
        by_model,
        seed=args.bootstrap_seed + 30011,
        repetitions=args.bootstrap_repetitions,
    )
    shot_rows = _shot_sensitivity_rows(
        evaluations,
        seed=args.bootstrap_seed + 40011,
        repetitions=args.bootstrap_repetitions,
    )

    model_fields = list(model_rows[0])
    _write_csv(output / "model_summary.csv", model_fields, model_rows)
    _write_csv(
        output / "geometry_associations.csv",
        ("dataset", "encoder_id", "x", "y", "spearman", "ci_low", "ci_high", "cluster_unit"),
        association_rows,
    )
    _write_csv(
        output / "predictive_increment.csv",
        tuple(predictive_increment_rows[0]),
        predictive_increment_rows,
    )
    _write_csv(
        output / "paired_model_comparisons.csv",
        (
            "dataset",
            "first_model",
            "second_model",
            "difference",
            "metric",
            "interpretation",
            "mean_difference",
            "ci_low",
            "ci_high",
            "n_paired_targets",
            "resampling_unit",
        ),
        paired_comparison_rows,
    )
    _write_csv(
        output / "dependence_strata.csv",
        tuple(dependence_rows[0]) if dependence_rows else (),
        dependence_rows,
    )
    _write_csv(
        output / "model_selection.csv",
        tuple(model_selection_rows[0]) if model_selection_rows else (),
        model_selection_rows,
    )
    _write_csv(
        output / "shot_sensitivity.csv",
        tuple(shot_rows[0]) if shot_rows else (),
        shot_rows,
    )

    labels = [_display_model(row["encoder_id"]) for row in model_rows]
    positions = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(max(10.5, 2.25 * len(labels)), 4.9))
    for offset, (metric, label, color, marker) in enumerate(
        (
            (
                "source_ood_balanced_accuracy",
                "labels from one context",
                "#0072B2",
                "o",
            ),
            (
                "oracle_ood_balanced_accuracy",
                "same-budget labels from both contexts",
                "#009E73",
                "s",
            ),
        )
    ):
        values = 100.0 * (np.asarray([row[metric] for row in model_rows]) - 0.5)
        lower = 100.0 * (
            np.asarray([row[metric] for row in model_rows])
            - np.asarray([row[f"{metric}_ci_low"] for row in model_rows])
        )
        upper = 100.0 * (
            np.asarray([row[f"{metric}_ci_high"] for row in model_rows])
            - np.asarray([row[metric] for row in model_rows])
        )
        lower = np.maximum(lower, 0.0)
        upper = np.maximum(upper, 0.0)
        axes[0].errorbar(
            positions + (offset - 0.5) * 0.16,
            values,
            yerr=np.vstack((lower, upper)),
            fmt=marker,
            color=color,
            capsize=3,
            linestyle="none",
            label=label,
        )
    gap = 100.0 * np.asarray([row["source_transfer_gap"] for row in model_rows])
    gap_lower = 100.0 * (
        np.asarray([row["source_transfer_gap"] for row in model_rows])
        - np.asarray([row["source_transfer_gap_ci_low"] for row in model_rows])
    )
    gap_upper = 100.0 * (
        np.asarray([row["source_transfer_gap_ci_high"] for row in model_rows])
        - np.asarray([row["source_transfer_gap"] for row in model_rows])
    )
    gap_lower = np.maximum(gap_lower, 0.0)
    gap_upper = np.maximum(gap_upper, 0.0)
    axes[1].errorbar(
        positions,
        gap,
        yerr=np.vstack((gap_lower, gap_upper)),
        fmt="o",
        color="#D55E00",
        capsize=3,
        linestyle="none",
    )
    axes[0].axhline(0.0, color="0.25", linewidth=1, linestyle="--")
    axes[1].axhline(0.0, color="0.25", linewidth=1, linestyle="--")
    axes[0].set_ylabel("OOD gain over chance (percentage points)")
    axes[1].set_ylabel("ID - OOD drop (percentage points)")
    for axis in axes:
        axis.set_xticks(positions, labels, rotation=24, ha="right")
    axes[0].legend(frameon=False, fontsize=11)
    fig.tight_layout()
    fig.savefig(output / "context_heldout_accuracy.png", dpi=220)
    fig.savefig(output / "context_heldout_accuracy.pdf")
    plt.close(fig)

    primary_associations = (
        ("target_capture_balanced", "capture"),
        ("conditional_axis_cosine", "axis alignment"),
        ("transported_margin", "transported margin"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.3, 4.5), sharey=True)
    model_order = sorted(by_model)
    for axis, (x_key, label) in zip(axes, primary_associations, strict=True):
        selected = {
            row["encoder_id"]: row
            for row in association_rows
            if row["x"] == x_key and row["y"] == "source_ood_balanced_accuracy"
        }
        for position, encoder_id in enumerate(model_order):
            row = selected[encoder_id]
            axis.errorbar(
                row["spearman"],
                position,
                xerr=np.asarray(
                    [
                        [max(0.0, row["spearman"] - row["ci_low"])],
                        [max(0.0, row["ci_high"] - row["spearman"])],
                    ]
                ),
                fmt="o",
                color="#0072B2",
                capsize=3,
            )
        axis.axvline(0.0, color="0.25", linewidth=1, linestyle="--")
        axis.set_xlim(-1.0, 1.0)
        axis.set_xlabel(f"Spearman with OOD accuracy\n{label}")
    axes[0].set_yticks(
        np.arange(len(model_order)), [_display_model(model) for model in model_order]
    )
    fig.tight_layout()
    fig.savefig(output / "geometry_transfer_forest.png", dpi=220)
    fig.savefig(output / "geometry_transfer_forest.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), sharex=True)
    palette = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")
    stratum_names = [row[0] for row in DEPENDENCE_STRATA]
    stratum_positions = np.arange(len(stratum_names))
    for model_index, encoder_id in enumerate(model_order):
        for axis, metric in zip(
            axes,
            ("conditional_axis_cosine", "source_ood_balanced_accuracy"),
            strict=True,
        ):
            selected = [
                row
                for row in dependence_rows
                if row["encoder_id"] == encoder_id and row["metric"] == metric
            ]
            selected.sort(key=lambda row: row["stratum_order"])
            values = np.asarray([row["mean"] for row in selected])
            if metric == "source_ood_balanced_accuracy":
                values = 100.0 * (values - 0.5)
                lower = 100.0 * (
                    np.asarray([row["mean"] - row["ci_low"] for row in selected])
                )
                upper = 100.0 * (
                    np.asarray([row["ci_high"] - row["mean"] for row in selected])
                )
            else:
                lower = np.asarray([row["mean"] - row["ci_low"] for row in selected])
                upper = np.asarray([row["ci_high"] - row["mean"] for row in selected])
            lower = np.maximum(lower, 0.0)
            upper = np.maximum(upper, 0.0)
            axis.errorbar(
                stratum_positions,
                values,
                yerr=np.vstack((lower, upper)),
                marker="o",
                capsize=2,
                color=palette[model_index % len(palette)],
                label=_display_model(encoder_id),
            )
    axes[0].set_ylabel("Conditional-axis alignment")
    axes[1].set_ylabel("OOD gain over chance (percentage points)")
    for axis in axes:
        axis.set_xticks(stratum_positions, ("low", "moderate", "high"))
        axis.set_xlabel(r"train-label dependence $|\phi|$")
    axes[0].legend(frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(output / "dependence_strata.png", dpi=220)
    fig.savefig(output / "dependence_strata.pdf")
    plt.close(fig)

    if model_selection_rows:
        selection_labels = {
            "random_model_expectation": "random model",
            "maximum_train_capture": "max capture",
            "maximum_axis_alignment": "max axis alignment",
            "maximum_transported_margin": "max transported margin",
            "oracle_best_single_model": "best fixed model*",
            "oracle_best_model_per_target": "best per target*",
        }
        values = 100.0 * np.asarray(
            [row["ood_balanced_accuracy"] - 0.5 for row in model_selection_rows]
        )
        lower = 100.0 * np.asarray(
            [
                row["ood_balanced_accuracy"] - row["ood_balanced_accuracy_ci_low"]
                for row in model_selection_rows
            ]
        )
        upper = 100.0 * np.asarray(
            [
                row["ood_balanced_accuracy_ci_high"] - row["ood_balanced_accuracy"]
                for row in model_selection_rows
            ]
        )
        lower = np.maximum(lower, 0.0)
        upper = np.maximum(upper, 0.0)
        colors = [
            "0.68" if row["uses_heldout_outcome"] else "#0072B2"
            for row in model_selection_rows
        ]
        fig, axis = plt.subplots(figsize=(8.6, 4.8))
        axis.bar(
            np.arange(len(values)),
            values,
            yerr=np.vstack((lower, upper)),
            color=colors,
            capsize=3,
        )
        axis.axhline(0.0, color="0.25", linewidth=1, linestyle="--")
        axis.set_ylabel("OOD gain over chance (percentage points)")
        axis.set_xticks(
            np.arange(len(values)),
            [selection_labels[row["rule"]] for row in model_selection_rows],
            rotation=24,
            ha="right",
        )
        axis.text(
            0.99,
            0.98,
            "*held-out reference",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            color="0.35",
        )
        fig.tight_layout()
        fig.savefig(output / "train_geometry_model_selection.png", dpi=220)
        fig.savefig(output / "train_geometry_model_selection.pdf")
        plt.close(fig)

    if len({row["shot"] for row in shot_rows}) >= 2:
        fig, axis = plt.subplots(figsize=(7.6, 4.8))
        for model_index, encoder_id in enumerate(model_order):
            selected = sorted(
                (row for row in shot_rows if row["encoder_id"] == encoder_id),
                key=lambda row: row["shot"],
            )
            shots = np.asarray([row["shot"] for row in selected])
            values = 100.0 * np.asarray(
                [row["source_ood_balanced_accuracy"] - 0.5 for row in selected]
            )
            lower = 100.0 * np.asarray(
                [
                    row["source_ood_balanced_accuracy"]
                    - row["source_ood_balanced_accuracy_ci_low"]
                    for row in selected
                ]
            )
            upper = 100.0 * np.asarray(
                [
                    row["source_ood_balanced_accuracy_ci_high"]
                    - row["source_ood_balanced_accuracy"]
                    for row in selected
                ]
            )
            lower = np.maximum(lower, 0.0)
            upper = np.maximum(upper, 0.0)
            axis.errorbar(
                shots,
                values,
                yerr=np.vstack((lower, upper)),
                marker="o",
                capsize=2,
                color=palette[model_index % len(palette)],
                label=_display_model(encoder_id),
            )
        axis.axhline(0.0, color="0.25", linewidth=1, linestyle="--")
        axis.set_xscale("log", base=2)
        axis.set_xticks(
            sorted({row["shot"] for row in shot_rows}),
            [str(shot) for shot in sorted({row["shot"] for row in shot_rows})],
        )
        axis.set_xlabel("labeled examples per target class")
        axis.set_ylabel("OOD gain over chance (percentage points)")
        axis.legend(frameon=False, fontsize=10)
        fig.tight_layout()
        fig.savefig(output / "shot_sensitivity.png", dpi=220)
        fig.savefig(output / "shot_sensitivity.pdf")
        plt.close(fig)

    increments = np.asarray(
        [row["representation_geometry_r2_increment"] for row in predictive_increment_rows]
    )
    increment_lower = increments - np.asarray(
        [row["geometry_increment_split_low"] for row in predictive_increment_rows]
    )
    increment_upper = np.asarray(
        [row["geometry_increment_split_high"] for row in predictive_increment_rows]
    ) - increments
    increment_lower = np.maximum(increment_lower, 0.0)
    increment_upper = np.maximum(increment_upper, 0.0)
    fig, axis = plt.subplots(figsize=(7.4, max(3.4, 0.7 * len(labels) + 1.6)))
    axis.errorbar(
        increments,
        positions,
        xerr=np.vstack((increment_lower, increment_upper)),
        fmt="o",
        color="#0072B2",
        capsize=3,
        linestyle="none",
    )
    axis.axvline(0.0, color="0.25", linewidth=1, linestyle="--")
    axis.set_xlabel(r"cross-validated $R^2$ gain beyond capture")
    axis.set_yticks(positions, labels)
    fig.tight_layout()
    fig.savefig(output / "geometry_beyond_capture.png", dpi=220)
    fig.savefig(output / "geometry_beyond_capture.pdf")
    plt.close(fig)

    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "source_commit": _git_commit(),
        "primary_shot": args.primary_shot,
        "bootstrap": {
            "unit": "target_attribute",
            "repetitions": args.bootstrap_repetitions,
            "seed": args.bootstrap_seed,
            "interpretation": (
                "across_target_empirical_variability_not_iid_image_sampling_inference"
            ),
        },
        "predictive_increment_protocol": {
            "unit": "target_attribute_after_within_target_averaging",
            "folds": 5,
            "split_repetitions": args.predictive_cv_repetitions,
            "null_permutations": args.predictive_null_permutations,
            "geometry_block_excludes_label_dependence": True,
            "dependence_added_only_in_final_tier": True,
            "permutation_test_conditioning": (
                "fixed_first_target_fold_assignment_for_both_observed_and_null"
            ),
            "permutation_null": (
                "joint_row_permutation_of_added_predictor_residuals_after_linear_"
                "projection_on_the_base_block"
            ),
            "split_interval_interpretation": (
                "sensitivity_to_deterministic_target_fold_assignments_not_a_"
                "sampling_confidence_interval"
            ),
        },
        "evaluations": metadata_records,
        "model_summary": model_rows,
        "geometry_associations": association_rows,
        "predictive_increment": predictive_increment_rows,
        "paired_model_comparisons": paired_comparison_rows,
        "dependence_strata": dependence_rows,
        "model_selection": model_selection_rows,
        "shot_sensitivity": shot_rows,
    }
    _write_json(output / "summary.json", summary)
    files = sorted(path for path in output.rglob("*") if path.is_file())
    (output / "SHA256SUMS").write_text(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(output).as_posix()}\n"
            for path in files
            if path.name != "SHA256SUMS"
        ),
        encoding="ascii",
    )
    print(f"Saved cross-model summary: {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="freeze a train-only manifest")
    prepare.add_argument("--dataset", choices=("celeba", "cub200"), required=True)
    prepare.add_argument("--config", help="CelebA analysis config")
    prepare.add_argument("--data-root", help="official CUB_200_2011 root")
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--fold-seed", type=int, default=20260811)
    prepare.add_argument("--primary-shot", type=int, default=32)
    prepare.add_argument("--minimum-heldout-cell-count", type=int, default=20)
    prepare.add_argument("--shot-seeds", nargs="+", type=int, default=DEFAULT_SHOT_SEEDS)
    prepare.set_defaults(function=prepare_command)

    cache = subparsers.add_parser("cache", help="extract one frozen feature cache")
    cache.add_argument("--dataset", choices=("celeba", "cub200"), required=True)
    cache.add_argument("--split", choices=("train", "test"), required=True)
    cache.add_argument(
        "--encoder-kind",
        choices=("checkpoint", "vicreg_imagenet1k", "supervised_imagenet1k"),
        required=True,
    )
    cache.add_argument("--encoder-id", required=True)
    cache.add_argument("--checkpoint")
    cache.add_argument("--weights-path")
    cache.add_argument("--config", help="CelebA analysis config")
    cache.add_argument("--data-root", help="official CUB_200_2011 root")
    cache.add_argument("--output", required=True)
    cache.add_argument("--device", default="cuda:0")
    cache.add_argument("--batch-size", type=int, default=256)
    cache.add_argument("--num-workers", type=int, default=16)
    cache.add_argument("--image-size", type=int, default=224)
    cache.add_argument("--no-crop-to-bbox", action="store_true")
    cache.set_defaults(function=cache_command)

    evaluate = subparsers.add_parser("evaluate", help="run a frozen held-out evaluation")
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--manifest-sha256", required=True)
    evaluate.add_argument("--train-cache", required=True)
    evaluate.add_argument("--test-cache", required=True)
    evaluate.add_argument("--output-dir", required=True)
    evaluate.add_argument("--device", default="cuda:0")
    evaluate.add_argument("--shots", nargs="+", type=int, default=[32])
    evaluate.add_argument("--whiten-rel-eig-threshold", type=float, default=1e-3)
    evaluate.set_defaults(function=evaluate_command)

    summarize = subparsers.add_parser("summarize", help="aggregate model evaluations")
    summarize.add_argument("--evaluations", nargs="+", required=True)
    summarize.add_argument("--output-dir", required=True)
    summarize.add_argument("--primary-shot", type=int, default=32)
    summarize.add_argument("--bootstrap-repetitions", type=int, default=2000)
    summarize.add_argument("--bootstrap-seed", type=int, default=20260811)
    summarize.add_argument("--predictive-cv-repetitions", type=int, default=200)
    summarize.add_argument("--predictive-null-permutations", type=int, default=999)
    summarize.set_defaults(function=summarize_command)
    return parser


def _validate_command_paths(args: argparse.Namespace) -> None:
    if args.command in {"prepare", "cache"}:
        if args.dataset == "celeba" and not args.config:
            raise ValueError("--config is required for CelebA")
        if args.dataset == "cub200" and not args.data_root:
            raise ValueError("--data-root is required for CUB-200")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _validate_command_paths(args)
    args.function(args)


if __name__ == "__main__":
    main()
