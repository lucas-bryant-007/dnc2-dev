from pathlib import Path

import numpy as np
from PIL import Image

from data_utils.cub200_dataset import (
    CUB200AttributeDataset,
    load_cub200_metadata,
)


def _write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _tiny_cub(root: Path):
    _write(root / "images.txt", "1 a/one.jpg\n2 b/two.jpg\n")
    _write(root / "image_class_labels.txt", "1 2\n2 1\n")
    _write(root / "train_test_split.txt", "1 1\n2 0\n")
    _write(root / "bounding_boxes.txt", "1 1 1 4 3\n2 0 0 6 5\n")
    _write(
        root / "attributes" / "attributes.txt",
        "1 has_bill_shape::curved\n2 has_wing_color::blue\n3 has_size::small\n",
    )
    rows = []
    values = ((1, 0, 1), (0, 1, 1))
    for image_id, present in enumerate(values, start=1):
        for attribute_id, value in enumerate(present, start=1):
            rows.append(f"{image_id} {attribute_id} {value} 4 0.1\n")
    _write(root / "attributes" / "image_attribute_labels.txt", "".join(rows))
    for relative, color in (("a/one.jpg", "red"), ("b/two.jpg", "blue")):
        path = root / "images" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (6, 5), color=color).save(path)


def test_load_cub200_metadata_parses_official_tables(tmp_path):
    _tiny_cub(tmp_path)

    metadata = load_cub200_metadata(tmp_path)

    assert metadata.relative_paths == ["a/one.jpg", "b/two.jpg"]
    assert metadata.attribute_names == [
        "bill_shape=curved",
        "wing_color=blue",
        "size=small",
    ]
    assert metadata.class_labels.tolist() == [1, 0]
    assert metadata.is_train.tolist() == [True, False]
    assert metadata.attributes.tolist() == [[1, 0, 1], [0, 1, 1]]


def test_cub200_dataset_uses_official_split_attributes_and_bbox(tmp_path):
    _tiny_cub(tmp_path)
    metadata = load_cub200_metadata(tmp_path)
    train = CUB200AttributeDataset(
        tmp_path,
        metadata,
        "train",
        crop_to_bbox=True,
    )
    test = CUB200AttributeDataset(
        tmp_path,
        metadata,
        "test",
        crop_to_bbox=False,
    )

    assert len(train) == len(test) == 1
    assert train[0]["image"].size == (4, 3)
    assert test[0]["image"].size == (6, 5)
    assert train[0]["attributes"].numpy().tolist() == [1, 0, 1]
    assert test[0]["class_label"] == 0


def test_cub200_metadata_rejects_missing_annotation_files(tmp_path):
    with np.testing.assert_raises(FileNotFoundError):
        load_cub200_metadata(tmp_path)


def test_cub200_metadata_rejects_incomplete_attribute_grid(tmp_path):
    _tiny_cub(tmp_path)
    labels = tmp_path / "attributes" / "image_attribute_labels.txt"
    labels.write_text(
        "\n".join(labels.read_text(encoding="utf-8").splitlines()[:-1]) + "\n",
        encoding="utf-8",
    )

    with np.testing.assert_raises_regex(
        ValueError,
        "each image/attribute pair once",
    ):
        load_cub200_metadata(tmp_path)
