"""Official CUB-200-2011 images, split, boxes, and binary attributes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class CUB200Metadata:
    image_ids: np.ndarray
    relative_paths: list[str]
    class_labels: np.ndarray
    is_train: np.ndarray
    bounding_boxes: np.ndarray
    attributes: np.ndarray
    attribute_names: list[str]


def _read_indexed_text(path: Path) -> tuple[np.ndarray, list[str]]:
    ids = []
    values = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            identifier, value = line.rstrip("\n").split(maxsplit=1)
            ids.append(int(identifier))
            values.append(value)
    return np.asarray(ids, dtype=np.int64), values


def _require_consecutive(ids: np.ndarray, name: str) -> None:
    expected = np.arange(1, len(ids) + 1, dtype=np.int64)
    if not np.array_equal(ids, expected):
        raise ValueError(f"{name} identifiers must be consecutive and one-indexed")


def load_cub200_metadata(root: str | Path) -> CUB200Metadata:
    """Parse metadata distributed in the official CUB_200_2011 archive."""
    root = Path(root).expanduser().resolve()
    # Caltech's tarball stores the attribute-name table as a top-level archive
    # member, next to the ``CUB_200_2011`` directory, while the per-image
    # annotations live inside that directory.  Repackaged copies commonly move
    # the name table into the dataset root or its ``attributes`` subdirectory,
    # so accept all three layouts in a deterministic order.
    attribute_name_candidates = (
        root / "attributes.txt",
        root / "attributes" / "attributes.txt",
        root.parent / "attributes.txt",
    )
    attribute_names_path = next(
        (path for path in attribute_name_candidates if path.is_file()),
        attribute_name_candidates[0],
    )
    required = (
        "images.txt",
        "image_class_labels.txt",
        "train_test_split.txt",
        "bounding_boxes.txt",
        "attributes/image_attribute_labels.txt",
    )
    missing = [name for name in required if not (root / name).is_file()]
    if not attribute_names_path.is_file():
        missing.append("attributes.txt")
    if missing:
        raise FileNotFoundError(
            f"CUB-200 root {root} is missing required files: {missing}"
        )

    image_ids, relative_paths = _read_indexed_text(root / "images.txt")
    _require_consecutive(image_ids, "image")
    n_images = len(image_ids)

    labels_table = np.loadtxt(root / "image_class_labels.txt", dtype=np.int64)
    split_table = np.loadtxt(root / "train_test_split.txt", dtype=np.int64)
    boxes_table = np.loadtxt(root / "bounding_boxes.txt", dtype=np.float64)
    if (
        labels_table.shape != (n_images, 2)
        or split_table.shape != (n_images, 2)
        or boxes_table.shape != (n_images, 5)
    ):
        raise ValueError("CUB image metadata tables have inconsistent shapes")
    for table, name in (
        (labels_table, "class label"),
        (split_table, "split"),
        (boxes_table, "bounding box"),
    ):
        _require_consecutive(table[:, 0].astype(np.int64), name)

    attribute_ids, raw_attribute_names = _read_indexed_text(attribute_names_path)
    _require_consecutive(attribute_ids, "attribute")
    attribute_names = [
        value.removeprefix("has_").replace("::", "=")
        for value in raw_attribute_names
    ]
    n_attributes = len(attribute_names)

    # Official rows are:
    # image_id attribute_id is_present certainty_id annotation_time
    attribute_rows = np.loadtxt(
        root / "attributes" / "image_attribute_labels.txt",
        dtype=np.float64,
        usecols=(0, 1, 2),
    )
    if attribute_rows.ndim != 2 or attribute_rows.shape[1] != 3:
        raise ValueError("Unexpected CUB image-attribute table shape")
    row_image = attribute_rows[:, 0].astype(np.int64) - 1
    row_attribute = attribute_rows[:, 1].astype(np.int64) - 1
    row_present = attribute_rows[:, 2]
    if (
        row_image.min(initial=0) < 0
        or row_image.max(initial=-1) >= n_images
        or row_attribute.min(initial=0) < 0
        or row_attribute.max(initial=-1) >= n_attributes
    ):
        raise ValueError("CUB image-attribute identifiers are out of range")
    if not np.all(np.isin(row_present, (0.0, 1.0))):
        raise ValueError("CUB image-attribute presence values must be binary")
    flat_indices = row_image * n_attributes + row_attribute
    expected_attributes = n_images * n_attributes
    if (
        len(flat_indices) != expected_attributes
        or np.unique(flat_indices).size != expected_attributes
    ):
        raise ValueError(
            "CUB image-attribute table must contain each image/attribute pair once"
        )
    attributes = np.zeros((n_images, n_attributes), dtype=np.uint8)
    attributes[row_image, row_attribute] = row_present.astype(np.uint8)

    return CUB200Metadata(
        image_ids=image_ids,
        relative_paths=relative_paths,
        class_labels=labels_table[:, 1].astype(np.int64) - 1,
        is_train=split_table[:, 1].astype(bool),
        bounding_boxes=boxes_table[:, 1:].astype(np.float32),
        attributes=attributes,
        attribute_names=attribute_names,
    )


class CUB200AttributeDataset(Dataset):
    """PyTorch view over an official CUB split with all 312 attributes."""

    def __init__(
        self,
        root: str | Path,
        metadata: CUB200Metadata,
        split: str,
        transform: Optional[Callable] = None,
        crop_to_bbox: bool = True,
    ):
        self.root = Path(root).expanduser().resolve()
        self.metadata = metadata
        self.transform = transform
        self.crop_to_bbox = crop_to_bbox
        if split not in {"train", "test"}:
            raise ValueError(f"split must be train or test, got {split}")
        keep = metadata.is_train if split == "train" else ~metadata.is_train
        self.indices = np.flatnonzero(keep)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, item: int):
        index = int(self.indices[item])
        image_path = self.root / "images" / self.metadata.relative_paths[index]
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
        if self.crop_to_bbox:
            x, y, width, height = self.metadata.bounding_boxes[index]
            left = max(0, int(np.floor(x)))
            top = max(0, int(np.floor(y)))
            right = min(image.width, int(np.ceil(x + width)))
            bottom = min(image.height, int(np.ceil(y + height)))
            if right > left and bottom > top:
                image = image.crop((left, top, right, bottom))
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "attributes": torch.from_numpy(
                self.metadata.attributes[index].astype(np.int64, copy=False)
            ),
            "class_label": int(self.metadata.class_labels[index]),
            "image_id": int(self.metadata.image_ids[index]),
        }
