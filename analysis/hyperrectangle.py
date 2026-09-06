"""Minimal, paper-faithful CelebA hyperrectangle experiment.

Fit on CelebA train, freeze everything, then evaluate on CelebA test. For a
centered whitened F and balanced Y in {-1,+1}:

    w = E[YF],  B = ||w||^2,  u = w/sqrt(B),  corner_t = Y_t sqrt(B_t).

sqrt(B_t) is a half-side; the full edge length is 2 sqrt(B_t).
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F


ATTRIBUTES = (
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes",
    "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair",
    "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
    "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
    "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
    "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks",
    "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings",
    "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie",
    "Young",
)
MODEL_SPECS = {
    "vicreg_celeba": {
        "repo_id": "dlf-ssl/vicreg-resnet50-celeba",
        "filename": "checkpoints/epoch_1000/model.safetensors",
    },
    "ijepa_celeba": {
        "repo_id": "dlf-ssl/ijepa-resnet50-celeba",
        "filename": "checkpoints/epoch_1000/model.safetensors",
    },
    "vicreg_imagenet": {
        "url": "https://dl.fbaipublicfiles.com/vicreg/resnet50.pth",
        "filename": "resnet50.pth",
        "sha256": "c843e7652491ff2f712734619fd74d85e5e92ac902615665ad3fd50dc6ada591",
    },
    "ijepa_imagenet": {
        "url": "https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar",
        "filename": "IN1K-vit.h.14-300e.pth.tar",
        "sha256": "0382013c481743e9ccea89f970bc6c6aa126aa19a62127500d6e672a641aae22",
    },
}
MODELS = tuple(MODEL_SPECS)
CELLS = tuple(itertools.product((-1, 1), repeat=3))

# Frozen paper protocol. These are not CLI knobs because they must not be tuned
# after looking at the held-out result.
SELECTION_SEED, TEST_SEEDS = 6, tuple(range(7, 27))
CANDIDATE_POOL, MAX_EXACT_CANDIDATES = 12, 10
MIN_CLASS_FRACTION, CANDIDATE_MIN_CAPTURE = 0.20, 0.05
MIN_TRAIN_CELL, MAX_TRAIN_CELL = 1000, 5000
PROXY_MAX_COSINE, MIN_TRAIN_CAPTURE, MAX_TRAIN_COSINE = 0.25, 0.10, 0.12
MAX_TEST_CELL, MIN_TEST_CELL = 500, 100
MIN_TEST_CAPTURE, MAX_TEST_COSINE, MAX_TEST_RMSE = 0.10, 0.15, 0.25
COVARIANCE_EIGENVALUE_CUTOFF, SSL_EIGENVALUE_CUTOFF, SSL_RIDGE = 1e-3, 1e-3, 1e-3


@dataclass(frozen=True)
class LinearMap:
    mean: torch.Tensor
    matrix: torch.Tensor

    def transform(self, x):
        return (x - self.mean) @ self.matrix


@dataclass(frozen=True)
class Whitener(LinearMap):
    eigenvalues: torch.Tensor
    relative_threshold: float


@dataclass(frozen=True)
class EncoderAdapter:
    model: torch.nn.Module
    is_vit: bool
    provenance: dict

    @torch.inference_mode()
    def encode(self, images):
        if not self.is_vit:
            return self.model(images).flatten(1)
        tokens = self.model.forward_features(images)
        patches = tokens[:, int(getattr(self.model, "num_prefix_tokens", 0)):]
        if not patches.shape[1]:
            raise ValueError("I-JEPA produced no patch tokens")
        return patches.mean(1)  # I-JEPA paper convention; CLS is untrained here.


class SelectionFailure(RuntimeError):
    def __init__(self, attempts):
        super().__init__("no train triple met the frozen criteria")
        self.attempts = attempts


# --------------------------------------------------------- download/checkpoint

def _snapshot_revision(path):
    parts = path.parts
    return parts[parts.index("snapshots") + 1] if "snapshots" in parts else None


def resolve_weights(model_name, weights=None, cache_dir=None):
    """Return a local checkpoint, downloading the published default if needed."""
    if model_name not in MODELS:
        raise ValueError(f"model must be one of {MODELS}")
    if weights:
        path = Path(weights).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path, {"source": "user_supplied", "downloaded_automatically": False}

    spec = MODEL_SPECS[model_name]
    if "repo_id" in spec:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as error:
            raise RuntimeError(
                "Automatic model download requires `pip install huggingface_hub`."
            ) from error
        kwargs = {"repo_id": spec["repo_id"], "revision": "main"}
        if cache_dir:
            kwargs["cache_dir"] = str(Path(cache_dir).expanduser().resolve())
        # Keep the snapshot path instead of resolving its blob symlink: the
        # commit hash and adjacent config.json are part of model provenance.
        path = Path(hf_hub_download(filename=spec["filename"], **kwargs)).absolute()
        # Local CelebA checkpoints need their companion training metadata so the
        # resolved architecture and any metadata mismatch remain auditable.
        hf_hub_download(filename="config.json", **kwargs)
        return path, {
            "source": "huggingface_hub",
            "repo_id": spec["repo_id"],
            "filename": spec["filename"],
            "revision": _snapshot_revision(path),
            "downloaded_automatically": True,
        }

    root = (Path(cache_dir).expanduser().resolve() if cache_dir
            else Path(torch.hub.get_dir()).resolve() / "hyperrectangle")
    root.mkdir(parents=True, exist_ok=True)
    path = root / spec["filename"]
    if not path.is_file() or _sha256(path) != spec["sha256"]:
        partial_path = path.with_suffix(path.suffix + ".part")
        print(f"Download {spec['url']} -> {path}")
        torch.hub.download_url_to_file(spec["url"], str(partial_path), progress=True)
        observed = _sha256(partial_path)
        if observed != spec["sha256"]:
            partial_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"downloaded {model_name} SHA-256 mismatch: {observed}"
            )
        partial_path.replace(path)
    return path, {
        "source": "official_url",
        "url": spec["url"],
        "expected_sha256": spec["sha256"],
        "downloaded_automatically": True,
    }

def _checkpoint_value(checkpoint, section, key):
    config = checkpoint.get("hyper_parameters", checkpoint)
    try:
        return str(config[section][key])
    except (KeyError, TypeError) as error:
        raise ValueError(f"checkpoint is missing {section}.{key}") from error


def _state_with_prefix(state, prefix):
    selected = {key[len(prefix):]: value for key, value in state.items()
                if key.startswith(prefix)}
    if not selected:
        raise ValueError(f"checkpoint contains no {prefix} weights")
    return selected


def _ijepa_encoder_prefix(state):
    if any(key.startswith("teacher.vit.") for key in state):
        return "teacher.vit."
    raise ValueError(
        "I-JEPA checkpoint has no EMA teacher (teacher.vit.*). "
        "Paper evaluation must not silently substitute the student."
    )


def _load_local_vicreg(checkpoint, state_dict=None):
    from torchvision.models import resnet18, resnet50

    architecture = _checkpoint_value(checkpoint, "model", "resnet_name").lower()
    if architecture not in {"resnet18", "resnet50"}:
        raise ValueError(f"unsupported VICReg backbone: {architecture}")
    network = {"resnet18": resnet18, "resnet50": resnet50}[architecture](weights=None)
    backbone = torch.nn.Sequential(*list(network.children())[:-1])
    state = checkpoint["state_dict"] if state_dict is None else state_dict
    backbone.load_state_dict(_state_with_prefix(state, "backbone."), strict=True)
    return backbone, {"architecture": architecture, "embedding_dimension": 2048 if architecture == "resnet50" else 512,
                      "image_size": 128, "patch_size": None, "metadata_warnings": []}


def _infer_vit_metadata(state):
    patch = state["patch_embed.proj.weight"]
    embed_dim, patch_size = int(patch.shape[0]), int(patch.shape[-1])
    blocks = [int(key.split(".")[1]) for key in state if key.startswith("blocks.")]
    depth = max(blocks) + 1
    has_class_token = "cls_token" in state
    position_tokens = int(state["pos_embed"].shape[1]) - int(has_class_token)
    grid_size = math.isqrt(position_tokens)
    if grid_size * grid_size != position_tokens:
        raise ValueError(f"I-JEPA has non-square positional grid: {position_tokens} tokens")
    sizes = {768: ("base", 12), 1024: ("large", 16), 1280: ("huge", 16)}
    if embed_dim not in sizes:
        raise ValueError(f"unsupported I-JEPA embedding dimension: {embed_dim}")
    size, heads = sizes[embed_dim]
    return {
        "architecture": f"vit_{size}_patch{patch_size}_{grid_size * patch_size}",
        "embedding_dimension": embed_dim,
        "depth": depth,
        "num_heads": heads,
        "image_size": grid_size * patch_size,
        "patch_size": patch_size,
        "has_class_token": has_class_token,
    }


def _build_ijepa(state):
    import timm

    metadata = _infer_vit_metadata(state)
    num_classes = int(state["head.weight"].shape[0]) if "head.weight" in state else 0
    backbone = timm.models.vision_transformer.VisionTransformer(
        img_size=metadata["image_size"], patch_size=metadata["patch_size"],
        embed_dim=metadata["embedding_dimension"], depth=metadata["depth"],
        num_heads=metadata["num_heads"], mlp_ratio=4.0, qkv_bias=True,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        class_token=metadata["has_class_token"], num_classes=num_classes,
        global_pool="token" if metadata["has_class_token"] else "",
        weight_init="skip",
    )
    backbone.load_state_dict(state, strict=True)
    return backbone, metadata


def _declared_ijepa_metadata(checkpoint):
    model = checkpoint.get("model", {}) if isinstance(checkpoint, dict) else {}
    data = checkpoint.get("data", {}) if isinstance(checkpoint, dict) else {}
    return {"architecture": model.get("encoder_type"),
            "image_size": model.get("image_size", data.get("img_size")),
            "patch_size": model.get("patch_size")}


def _load_local_ijepa(checkpoint, state_dict=None):
    state = checkpoint["state_dict"] if state_dict is None else state_dict
    selected = _state_with_prefix(state, _ijepa_encoder_prefix(state))
    backbone, metadata = _build_ijepa(selected)
    declared = _declared_ijepa_metadata(checkpoint)
    warnings = [f"declared {key}={value}, inferred {metadata[key]} from checkpoint tensors"
                for key, value in declared.items()
                if value is not None and str(value).lower() != str(metadata[key]).lower()]
    return backbone, {**metadata, "declared_metadata": declared,
                      "metadata_warnings": warnings}


def _load_imagenet_vicreg(state):
    from torchvision.models import resnet50

    backbone = resnet50(weights=None)
    backbone.fc = torch.nn.Identity()
    backbone.load_state_dict(state, strict=True)
    return backbone, {"architecture": "resnet50", "embedding_dimension": 2048,
                      "image_size": 224, "patch_size": None, "metadata_warnings": []}


def _load_imagenet_ijepa(path, checkpoint, state):
    if "target_encoder" in checkpoint:
        selected = {key.removeprefix("module."): value
                    for key, value in checkpoint["target_encoder"].items()}
        format_name = "official_meta_full_checkpoint_target_encoder"
    else:
        raise ValueError(f"unrecognized official I-JEPA checkpoint format: {path}")
    backbone, metadata = _build_ijepa(selected)
    return backbone, {**metadata, "checkpoint_format": format_name,
                      "metadata_warnings": []}


def _read_checkpoint(path):
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        config_paths = [parent / "config.json" for parent in (path.parent, *path.parents)]
        config_path = next((candidate for candidate in config_paths if candidate.is_file()), None)
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path else {}
        return config, load_file(path)
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except RuntimeError as error:
        if "mmap" not in str(error).lower():
            raise
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    return checkpoint, state


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_encoder(model_name, weights, device, model_cache_dir=None):
    path, source = resolve_weights(model_name, weights, model_cache_dir)
    checkpoint, state = _read_checkpoint(path)
    method = model_name.split("_", 1)[0]
    if model_name.endswith("_celeba"):
        declared_method = _checkpoint_value(checkpoint, "method", "name").lower()
        if method != declared_method:
            raise ValueError(f"requested {model_name}, but checkpoint says {declared_method}")
    if model_name == "vicreg_celeba":
        model, metadata = _load_local_vicreg(checkpoint, state)
        encoder_name = "backbone"
    elif model_name == "ijepa_celeba":
        model, metadata = _load_local_ijepa(checkpoint, state)
        encoder_name = "EMA teacher, mean patch pooling"
    elif model_name == "vicreg_imagenet":
        model, metadata = _load_imagenet_vicreg(state)
        encoder_name = "official backbone"
    else:
        model, metadata = _load_imagenet_ijepa(path, checkpoint, state)
        encoder_name = "official target encoder, mean patch pooling"
    model = model.to(device).eval().requires_grad_(False)
    return EncoderAdapter(model, method == "ijepa", {
        "name": model_name, "method": method, "pretraining_dataset": model_name.split("_", 1)[1],
        "encoder": encoder_name, **metadata, **source,
        "weights": str(path), "sha256": _sha256(path),
    })


# ------------------------------------------------------------------------ data

def build_transforms(model_name):
    """The same method-specific CelebA transforms used during training."""
    from torchvision import transforms

    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    tensor = [transforms.ToTensor(), normalize]
    if model_name == "vicreg_celeba":
        train = [transforms.RandomCrop(160), transforms.Resize((128, 128)),
                 transforms.RandomHorizontalFlip(),
                 transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.05)], p=0.8),
                 transforms.RandomGrayscale(p=0.1),
                 transforms.RandomApply([transforms.GaussianBlur(7, (0.1, 2.0))], p=0.5)]
        evaluate = [transforms.CenterCrop(160), transforms.Resize((128, 128))]
    elif model_name == "ijepa_celeba":
        train = [transforms.RandomCrop(160), transforms.Resize((224, 224)),
                 transforms.RandomHorizontalFlip()]
        evaluate = [transforms.Resize((224, 224))]
    elif model_name == "vicreg_imagenet":
        train = [transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
                 transforms.RandomHorizontalFlip(),
                 transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                 transforms.RandomGrayscale(p=0.2),
                 transforms.RandomApply([transforms.GaussianBlur(23)], p=0.5)]
        evaluate = [transforms.Resize(256), transforms.CenterCrop(224)]
    else:
        # Official I-JEPA ViT-H/14 uses a 224 crop, scale [0.3,1.0], and no
        # color distortion, blur, or horizontal flip.
        train = [transforms.RandomResizedCrop(224, scale=(0.3, 1.0))]
        evaluate = [transforms.Resize((224, 224))]
    return transforms.Compose(train + tensor), transforms.Compose(evaluate + tensor)


def build_eval_transform(model_name):
    return build_transforms(model_name)[1]


def load_celeba_splits(cache_dir):
    from datasets import load_dataset

    options = {"cache_dir": cache_dir}
    return (load_dataset("flwrlabs/celeba", split="train", **options),
            load_dataset("flwrlabs/celeba", split="test", **options))


def resolve_celeba_attributes(dataset):
    columns = set(dataset.column_names)
    names = [name for name in ATTRIBUTES if name in columns]
    if not names:
        raise ValueError("CelebA attribute columns were not found")
    return names


def _extract(dataset, encode, transform, names, device, batch_size, max_samples, views):
    """One extractor serves paired augmented views and deterministic features."""
    from torch.utils.data import DataLoader

    def collate(batch):
        images = [row["image"].convert("RGB") for row in batch]
        tensors = [torch.stack([transform(image) for image in images]) for _ in range(views)]
        if names is None:
            return tensors
        labels = torch.tensor([[int(row[name]) for name in names] for row in batch],
                              dtype=torch.float32)
        return tensors, labels

    outputs = [[] for _ in range(views)]
    labels, seen = [], 0
    for batch in DataLoader(dataset, batch_size=batch_size, collate_fn=collate):
        tensors, batch_labels = (batch, None) if names is None else batch
        for index, images in enumerate(tensors):
            outputs[index].append(F.normalize(encode(images.to(device)), dim=1).cpu())
        if batch_labels is not None:
            labels.append(batch_labels)
        seen += len(tensors[0])
        if max_samples is not None and seen >= max_samples:
            break
    limit = max_samples or seen
    features = [torch.cat(values)[:limit] for values in outputs]
    return features if names is None else (features[0], torch.cat(labels)[:limit])


def extract_dataset_features(dataset, encode, transform, attribute_names, *, device,
                             batch_size, max_samples=None):
    return _extract(dataset, encode, transform, attribute_names, device,
                    batch_size, max_samples, views=1)


def extract_paired_features(dataset, encode, transform, *, device, batch_size,
                            max_samples=None):
    return tuple(_extract(dataset, encode, transform, None, device,
                          batch_size, max_samples, views=2))


def fit_ssl_map(first, second, covariance_threshold=COVARIANCE_EIGENVALUE_CUTOFF,
                ssl_threshold=SSL_EIGENVALUE_CUTOFF):
    """Fit the train-only paired-view map and retain genuinely stable directions."""
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("paired SSL views must have the same [N,D] shape")
    joined = torch.cat((first, second))
    mean = joined.mean(0)
    first_centered, second_centered = first - mean, second - mean
    joined_centered = joined - mean
    covariance = joined_centered.T @ joined_centered / len(joined_centered)
    covariance_values, covariance_vectors = torch.linalg.eigh(covariance)
    order = torch.argsort(covariance_values, descending=True)
    covariance_values, covariance_vectors = covariance_values[order], covariance_vectors[:, order]
    if covariance_values[0] <= 0:
        raise ValueError("paired features have no positive covariance directions")
    covariance_keep = covariance_values >= covariance_values[0] * covariance_threshold
    retained_covariance_values = covariance_values[covariance_keep]
    ridge = SSL_RIDGE * covariance_values[0]
    whitener = covariance_vectors[:, covariance_keep] / (retained_covariance_values + ridge).sqrt()

    first_white, second_white = first_centered @ whitener, second_centered @ whitener
    cross = first_white.T @ second_white / len(first)
    cross = (cross + cross.T) / 2
    ssl_values, ssl_vectors = torch.linalg.eigh(cross)
    order = torch.argsort(ssl_values, descending=True)
    ssl_values, ssl_vectors = ssl_values[order], ssl_vectors[:, order]
    if ssl_values[0] <= 0:
        raise ValueError("paired views have no positively stable SSL directions")
    ssl_keep = ssl_values >= ssl_values[0] * ssl_threshold
    ssl_keep &= ssl_values > 0
    retained_ssl_values = ssl_values[ssl_keep]
    mapping = LinearMap(mean, whitener @ ssl_vectors[:, ssl_keep])
    return mapping, {
        "fit_split": "train",
        "fit_population": "all train instances; two augmented views each",
        "input_dimension": first.shape[1],
        "covariance_retained_dimension": int(covariance_keep.sum()),
        "retained_dimension": int(ssl_keep.sum()),
        "covariance_relative_eigenvalue_cutoff": covariance_threshold,
        "ssl_positive_relative_eigenvalue_cutoff": ssl_threshold,
        "whitening_ridge_relative_to_top_eigenvalue": SSL_RIDGE,
        "top_ssl_eigenvalue": float(ssl_values[0]),
        "minimum_retained_ssl_eigenvalue": float(retained_ssl_values[-1]),
        "nonpositive_ssl_directions_dropped": int((ssl_values <= 0).sum()),
        "ssl_retention_rule": "positive cross-view eigenvalue >= cutoff * top eigenvalue",
        "frozen_for_test": True,
    }


@torch.inference_mode()
def transform_in_chunks(features, mapping, device, batch_size):
    mean, matrix = mapping.mean.to(device), mapping.matrix.to(device)
    return torch.cat([(part.to(device) - mean) @ matrix
                      for part in features.split(batch_size)])


# ------------------------------------------------------------------------ math

def fit_whitener(features, relative_threshold=COVARIANCE_EIGENVALUE_CUTOFF):
    mean = features.mean(0)
    centered = features - mean
    covariance = centered.T @ centered / len(features)
    values, vectors = torch.linalg.eigh(covariance)
    order = torch.argsort(values, descending=True)
    values, vectors = values[order], vectors[:, order]
    keep = values > values[0] * relative_threshold
    if not keep.any():
        raise ValueError("whitening retained no directions")
    return Whitener(mean, vectors[:, keep] / values[keep].sqrt(),
                    values[keep], relative_threshold)


def as_pm_one(labels):
    values = set(torch.unique(labels).cpu().tolist())
    if values <= {0.0, 1.0}:
        return 2 * labels - 1
    if values <= {-1.0, 1.0}:
        return labels.float()
    raise ValueError("labels must be binary")


def task_probe(features, labels):
    """Empirical E[YF]; on every balanced fit fold this is (mu+ - mu-)/2."""
    labels = as_pm_one(labels)
    if not bool((labels > 0).any()) or not bool((labels < 0).any()):
        raise ValueError("both classes are required")
    vector = (labels[:, None] * features).mean(0)
    return vector, vector.square().sum().item()


def _balanced_rows(labels, seed, maximum):
    labels = as_pm_one(labels)
    groups = [torch.where((labels == labels.new_tensor(cell)).all(1))[0] for cell in CELLS]
    counts = [len(group) for group in groups]
    if min(counts) == 0:
        raise ValueError("all eight joint cells are required")
    per_cell = min(counts) if maximum is None else min(min(counts), maximum)
    generator = torch.Generator().manual_seed(seed)
    chosen = [group[torch.randperm(len(group), generator=generator)[:per_cell].to(group.device)]
              for group in groups]
    return torch.cat(chosen), counts, per_cell


def _split_cells(rows, per_cell, sizes):
    folds = [[] for _ in sizes]
    for cell in range(8):
        values, start = rows[cell * per_cell:(cell + 1) * per_cell], 0
        for fold, size in zip(folds, sizes, strict=True):
            fold.append(values[start:start + size])
            start += size
    return tuple(torch.cat(fold) for fold in folds)


def _geometry_from_gram(gram):
    capture = torch.diagonal(gram)
    if not bool(torch.all(capture > 0)):
        return {"valid": False, "capture_B": capture.tolist(),
                "gram_matrix": gram.tolist(), "cosine_matrix": None,
                "maximum_absolute_cosine": None}
    cosine = gram / torch.sqrt(capture[:, None] * capture[None, :])
    cosine.fill_diagonal_(1)
    triangle = torch.triu_indices(3, 3, 1, device=gram.device)
    return {"valid": True, "capture_B": capture.tolist(),
            "gram_matrix": gram.tolist(), "cosine_matrix": cosine.tolist(),
            "maximum_absolute_cosine": cosine.abs()[triangle.unbind()].max().item()}


def _crossfit_geometry(features_a, labels_a, features_b, labels_b):
    probes_a = torch.stack([task_probe(features_a, labels_a[:, i])[0] for i in range(3)])
    probes_b = torch.stack([task_probe(features_b, labels_b[:, i])[0] for i in range(3)])
    cross = probes_a @ probes_b.T
    return _geometry_from_gram((cross + cross.T) / 2)


def _named_geometry(geometry, names, status):
    return {"estimator": "symmetrized_split_half_cross_gram",
            "valid_positive_diagonal": geometry["valid"],
            "capture_B": dict(zip(names, geometry["capture_B"], strict=True)),
            "gram_matrix": geometry["gram_matrix"],
            "cosine_matrix": geometry["cosine_matrix"],
            "max_abs_cos": geometry["maximum_absolute_cosine"],
            "task_selection_status": status,
            "post_selection_unbiasedness_claimed": False}


def _balanced_proxy(features, labels):
    labels = as_pm_one(labels)
    signs = torch.tensor(CELLS, dtype=features.dtype, device=features.device)
    groups = [features[(labels == cell).all(1)] for cell in signs]
    counts = [len(group) for group in groups]
    if min(counts) == 0:
        return {"counts": counts}
    probes = signs.T @ torch.stack([group.mean(0) for group in groups]) / 8
    capture, axes = probes.square().sum(1), F.normalize(probes, dim=1)
    triangle = torch.triu_indices(3, 3, 1, device=features.device)
    return {"counts": counts, "capture_B": capture.tolist(),
            "maximum_absolute_cosine": (axes @ axes.T).abs()[triangle.unbind()].max().item()}


def fit_task_axes(features, labels, names, capture_B=None):
    vectors, plug_in = zip(*[task_probe(features, labels[:, i]) for i in range(3)])
    axes = F.normalize(torch.stack(vectors), dim=1)
    capture = list(plug_in if capture_B is None else capture_B)
    if min(capture) <= 0:
        raise ValueError("positive capture is required")
    predicted = [{"combo": [(s + 1) // 2 for s in signs], "signs": list(signs),
                  "center": [signs[i] * math.sqrt(capture[i]) for i in range(3)]}
                 for signs in CELLS]
    return {"names": list(names), "axes": axes, "capture_B": capture,
            "plug_in_capture_B": list(plug_in),
            "cosine_matrix": (axes @ axes.T).cpu().tolist(),
            "predicted_corners": predicted}


def select_train_triple(features, attributes, names, *, seed=SELECTION_SEED):
    """Screen and fit only on train; test is never passed to this function."""
    screened, labels = fit_whitener(features).transform(features), as_pm_one(attributes)
    eligible = []
    for index in range(labels.shape[1]):
        fraction = min((labels[:, index] > 0).float().mean().item(),
                       (labels[:, index] < 0).float().mean().item())
        capture = task_probe(screened, labels[:, index])[1]
        if fraction >= MIN_CLASS_FRACTION and capture >= CANDIDATE_MIN_CAPTURE:
            eligible.append((index, capture))
    eligible = [index for index, _ in sorted(eligible, key=lambda row: -row[1])[:CANDIDATE_POOL]]

    ranked = []
    for triple in itertools.combinations(eligible, 3):
        proxy = _balanced_proxy(screened, labels[:, triple])
        if (min(proxy["counts"]) >= MIN_TRAIN_CELL
                and proxy["maximum_absolute_cosine"] <= PROXY_MAX_COSINE):
            ranked.append((min(proxy["capture_B"]), proxy["maximum_absolute_cosine"],
                           -sum(proxy["capture_B"]) / 3, triple, proxy))
    ranked.sort(key=lambda row: (-row[0], row[1], row[2], row[3]))

    attempts = []
    for rank, (_min_b, _cos, _mean_b, triple, proxy) in enumerate(
            ranked[:MAX_EXACT_CANDIDATES], 1):
        rows, counts, n = _balanced_rows(labels[:, triple], seed, MAX_TRAIN_CELL)
        n_white, n_a = n // 3, (n - n // 3) // 2
        white_rows, fold_a, fold_b = _split_cells(rows, n, (n_white, n_a, n - n_white - n_a))
        whitener = fit_whitener(features[white_rows])
        geometry = _crossfit_geometry(
            whitener.transform(features[fold_a]), labels[fold_a][:, triple],
            whitener.transform(features[fold_b]), labels[fold_b][:, triple])
        passed = bool(geometry["valid"] and min(geometry["capture_B"]) >= MIN_TRAIN_CAPTURE
                      and geometry["maximum_absolute_cosine"] <= MAX_TRAIN_COSINE)
        attempt = {"rank": rank, "triple": [names[i] for i in triple], "proxy": proxy,
                   "original_cell_counts": counts, "samples_per_cell": n,
                   "capture_B": geometry["capture_B"],
                   "maximum_absolute_cosine": geometry["maximum_absolute_cosine"],
                   "passed": passed}
        attempts.append(attempt)
        cosine_text = ("invalid" if geometry["maximum_absolute_cosine"] is None
                       else f"{geometry['maximum_absolute_cosine']:.3f}")
        print(f"  candidate {rank}: {attempt['triple']} max|cos|={cosine_text} passed={passed}")
        if passed:
            balanced = whitener.transform(features[rows])
            box = fit_task_axes(balanced, labels[rows][:, triple], attempt["triple"],
                                geometry["capture_B"])
            return {"indices": list(triple), "names": attempt["triple"],
                    "selected_rows": rows, "whitener": whitener, "box": box,
                    "exact_attempts": attempts, "crossfit_probe_geometry": geometry,
                    "balance": {"original_cell_counts": counts, "samples_per_cell": n,
                                "whitening_samples_per_cell": n_white,
                                "probe_samples_per_cell_a": n_a,
                                "probe_samples_per_cell_b": n - n_white - n_a}}
    raise SelectionFailure(attempts)


def measure_cell_centroids(features, labels, axes):
    coordinates, labels = features @ axes.T, as_pm_one(labels)
    cells = []
    for signs in CELLS:
        values = coordinates[(labels == labels.new_tensor(signs)).all(1)]
        cells.append({"combo": [(s + 1) // 2 for s in signs], "signs": list(signs),
                      "count": len(values),
                      "center": values.mean(0).cpu().tolist() if len(values) else None,
                      "standard_error": (values.std(0).div(math.sqrt(len(values))).cpu().tolist()
                                         if len(values) > 1 else None)})
    return coordinates, cells


def corner_diagnostics(observed, predicted):
    target = {tuple(row["combo"]): torch.tensor(row["center"], dtype=torch.float64)
              for row in predicted}
    errors, radii, counts, combos = [], [], [], []
    for row in observed:
        if row["center"] is None:
            continue
        corner = target[tuple(row["combo"])]
        errors.append(torch.linalg.vector_norm(
            torch.tensor(row["center"], dtype=torch.float64) - corner).item())
        radii.append(float(corner @ corner)); counts.append(row["count"]); combos.append(row["combo"])
    if len(errors) != 8:
        raise ValueError("all eight held-out cells are required")
    rmse, radius = math.sqrt(sum(x * x for x in errors) / 8), math.sqrt(sum(radii) / 8)
    return {"n_corners": 8, "centroid_rmse": rmse, "predicted_rms_radius": radius,
            "normalized_centroid_rmse": rmse / radius, "max_centroid_error": max(errors),
            "min_cell_count": min(counts),
            "per_corner_error": [{"combo": combo, "l2_error": error}
                                 for combo, error in zip(combos, errors, strict=True)]}


def side_length_diagnostics(observed, predicted, names):
    """Compare all twelve observed centroid edges with the predicted box edges."""
    observed_by_sign = {tuple(row["signs"]): row for row in observed}
    predicted_by_sign = {tuple(row["signs"]): row for row in predicted}
    axes, all_observed, all_predicted = [], [], []
    for axis, name in enumerate(names):
        observed_lengths, axial_lengths, predicted_lengths, edges = [], [], [], []
        for signs in CELLS:
            if signs[axis] != -1:
                continue
            opposite = list(signs); opposite[axis] = 1; opposite = tuple(opposite)
            low, high = observed_by_sign[signs], observed_by_sign[opposite]
            if low["center"] is None or high["center"] is None:
                raise ValueError("all eight centroids are required for side lengths")
            low_center = torch.tensor(low["center"], dtype=torch.float64)
            high_center = torch.tensor(high["center"], dtype=torch.float64)
            predicted_low = torch.tensor(predicted_by_sign[signs]["center"], dtype=torch.float64)
            predicted_high = torch.tensor(predicted_by_sign[opposite]["center"], dtype=torch.float64)
            delta = high_center - low_center
            observed_length = float(torch.linalg.vector_norm(delta))
            predicted_length = float(torch.linalg.vector_norm(predicted_high - predicted_low))
            observed_lengths.append(observed_length)
            axial_lengths.append(abs(float(delta[axis])))
            predicted_lengths.append(predicted_length)
            edges.append({"from_signs": list(signs), "to_signs": list(opposite),
                          "observed_euclidean_length": observed_length,
                          "observed_axis_coordinate_length": axial_lengths[-1],
                          "predicted_length": predicted_length})
        empirical_mean = sum(observed_lengths) / len(observed_lengths)
        predicted_mean = sum(predicted_lengths) / len(predicted_lengths)
        axes.append({
            "attribute": name, "n_parallel_edges": len(edges),
            "empirical_euclidean_lengths": observed_lengths,
            "empirical_axis_coordinate_lengths": axial_lengths,
            "mean_empirical_euclidean_length": empirical_mean,
            "mean_empirical_axis_coordinate_length": sum(axial_lengths) / len(axial_lengths),
            "predicted_full_length_2sqrtB": predicted_mean,
            "empirical_to_predicted_ratio": empirical_mean / predicted_mean,
            "relative_error": (empirical_mean - predicted_mean) / predicted_mean,
            "edges": edges,
        })
        all_observed.extend(observed_lengths); all_predicted.extend(predicted_lengths)
    differences = [observed - predicted
                   for observed, predicted in zip(all_observed, all_predicted, strict=True)]
    return {
        "definition": "Euclidean distances between held-out centroids differing in one label",
        "per_axis": axes,
        "mean_empirical_edge_length": sum(all_observed) / len(all_observed),
        "mean_predicted_edge_length": sum(all_predicted) / len(all_predicted),
        "edge_length_rmse": math.sqrt(sum(value * value for value in differences) / len(differences)),
        "n_edges": len(all_observed),
    }


def evaluate_test_seed(features, labels, selection, seed):
    rows, counts, n = _balanced_rows(labels, seed, MAX_TEST_CELL)
    _, cells = measure_cell_centroids(features[rows], labels[rows], selection["box"]["axes"])
    fold_a, fold_b = _split_cells(rows, n, (n // 2, n - n // 2))
    geometry = _crossfit_geometry(features[fold_a], labels[fold_a],
                                  features[fold_b], labels[fold_b])
    diagnostics = corner_diagnostics(cells, selection["box"]["predicted_corners"])
    side_lengths = side_length_diagnostics(
        cells, selection["box"]["predicted_corners"], selection["names"])
    passed = bool(geometry["valid"] and geometry["maximum_absolute_cosine"] <= MAX_TEST_COSINE
                  and min(geometry["capture_B"]) >= MIN_TEST_CAPTURE
                  and diagnostics["normalized_centroid_rmse"] <= MAX_TEST_RMSE
                  and n >= MIN_TEST_CELL)
    return {"seed": seed, "original_cell_counts": counts, "samples_per_cell": n,
            "cell_centroids": cells, "crossfit_probe_geometry": geometry,
            "corner_diagnostics": diagnostics, "side_length_diagnostics": side_lengths,
            "headline_criteria_passed": passed}


def summarize_stability(records, names):
    aggregate = _geometry_from_gram(torch.tensor([
        row["crossfit_probe_geometry"]["gram_matrix"] for row in records]).mean(0))
    rmses = [row["corner_diagnostics"]["normalized_centroid_rmse"] for row in records]
    cosines = [row["crossfit_probe_geometry"]["maximum_absolute_cosine"] for row in records]
    valid_cosines = [value for value in cosines if value is not None]
    stats = lambda values: {"mean": sum(values) / len(values),
                            "min": min(values), "max": max(values)}
    return {"n_resamples": len(records), "test_balance_seeds": [r["seed"] for r in records],
            "pass_count": sum(r["headline_criteria_passed"] for r in records),
            "all_resamples_passed": all(r["headline_criteria_passed"] for r in records),
            "normalized_centroid_rmse": stats(rmses),
            "maximum_absolute_cosine": stats(valid_cosines) if valid_cosines else None,
            "invalid_crossfit_resamples": len(cosines) - len(valid_cosines),
            "aggregate_crossfit_probe_geometry": _named_geometry(
                aggregate, names, "frozen_from_independent_training_split")}


# ---------------------------------------------------------------------- output

def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def plot_hyperrectangle(path, names, observed, predicted, *, subtitle="Frozen encoder",
                        diagnostics=None, maximum_cosine=None, side_lengths=None, passed=True):
    """Paper Figure 4 styling: observed solid, train prediction dashed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    ink, amber, slate = "#1F2A37", "#D97706", "#6B7280"
    colors = ("#0D9488", "#2563EB", "#D97706", "#BE185D",
              "#0F766E", "#1D4ED8", "#B45309", "#9D174D")
    edges = ((0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
             (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7))
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5,
                                "text.color": ink, "pdf.fonttype": 42, "ps.fonttype": 42})
    observed = torch.tensor([row["center"] for row in observed])
    predicted = torch.tensor([row["center"] for row in predicted])
    figure = plt.figure(figsize=(4.9, 4.4)); axis = figure.add_subplot(111, projection="3d")
    for first, second in edges:
        axis.plot(*observed[[first, second]].T, color=ink, lw=1.7)
        axis.plot(*predicted[[first, second]].T, color=amber, lw=1.25, ls=(0, (3, 2)))
    for index, color in enumerate(colors):
        axis.scatter(*observed[index], s=30, color=color, edgecolor=ink, lw=0.6,
                     depthshade=False)
        axis.scatter(*predicted[index], s=19, marker="D", facecolor="white",
                     edgecolor=amber, lw=0.7, depthshade=False)
    points, center = torch.cat((observed, predicted)), torch.cat((observed, predicted)).mean(0)
    span = max(0.70 * (points.max(0).values - points.min(0).values).max().item(), 1e-6)
    axis.set_xlim(center[0] - span, center[0] + span)
    axis.set_ylim(center[1] - span, center[1] + span)
    axis.set_zlim(center[2] - span, center[2] + span)
    axis.set_box_aspect((1, 1, 1)); axis.view_init(elev=18, azim=-56); axis.set_axis_off()
    axis.text2D(0.5, 1.05, "CelebA", transform=axis.transAxes, ha="center", va="top",
                fontsize=10.8, fontweight="bold")
    axis.text2D(0.5, 0.975, subtitle, transform=axis.transAxes, ha="center", va="top",
                fontsize=8.8, color=slate)
    axis.text2D(0.5, 0.115, "\n".join(n.replace("_", " ").lower() for n in names),
                transform=axis.transAxes, ha="center", va="bottom", fontsize=8.6,
                color=slate, linespacing=1.35)
    if diagnostics is not None and maximum_cosine is not None:
        caption = (f"RMSE {diagnostics['normalized_centroid_rmse']:.3f}    "
                   f"max|cos| {maximum_cosine:.3f}")
        if side_lengths is not None:
            ratio = (side_lengths["mean_empirical_edge_length"] /
                     side_lengths["mean_predicted_edge_length"])
            caption += f"    edge ratio {ratio:.3f}"
        axis.text2D(0.5, -0.055, caption + ("" if passed else "\nmisses fixed criteria"),
                    transform=axis.transAxes, ha="center", va="bottom", fontsize=9.6)
    handles = (Line2D([0], [0], color=ink, lw=1.7, marker="o", ms=6.5,
                      markerfacecolor=colors[1], label="observed held-out centroids"),
               Line2D([0], [0], color=amber, lw=1.25, ls=(0, (3, 2)), marker="D",
                      ms=5.4, markerfacecolor="white", label="train-predicted capture box"))
    figure.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0),
                  frameon=False, fontsize=8.8)
    figure.text(0.5, 0.006, "Geometry fitted on train; evaluated on held-out test images.",
                ha="center", color=slate, fontsize=7.8)
    figure.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.11)
    path.parent.mkdir(parents=True, exist_ok=True); stem = path.with_suffix("")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02,
                   metadata={"Creator": "minimal hyperrectangle", "CreationDate": None})
    figure.savefig(stem.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02,
                   dpi=320, metadata={"Software": "minimal hyperrectangle"})
    plt.close(figure)


def run_experiment(args):
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    torch.manual_seed(SELECTION_SEED); random.seed(SELECTION_SEED)
    output = Path(args.out_dir).expanduser().resolve()
    json_path = output / f"hyperrectangle_{args.model}.json"
    figure_path = output / f"hyperrectangle_{args.model}.png"

    print(f"1/6  Load {args.model} on {device}")
    encoder = load_encoder(args.model, args.weights, device, args.model_cache_dir)
    train_transform, eval_transform = build_transforms(args.model)
    train, test = load_celeba_splits(args.cache_dir)
    names = resolve_celeba_attributes(train)
    batch_size = args.batch_size or (8 if args.model == "ijepa_imagenet"
                                     else 32 if args.model == "ijepa_celeba" else 128)

    print("2/6  Fit paired-view SSL map on train")
    view_a, view_b = extract_paired_features(
        train, encoder.encode, train_transform, device=device,
        batch_size=batch_size, max_samples=args.max_samples)
    ssl_map, ssl_record = fit_ssl_map(view_a, view_b); del view_a, view_b

    print("3/6  Select and fit three train tasks")
    train_features, train_labels = extract_dataset_features(
        train, encoder.encode, eval_transform, names, device=device,
        batch_size=batch_size, max_samples=args.max_samples)
    raw_dimension = train_features.shape[1]
    train_features = transform_in_chunks(train_features, ssl_map, device,
                                         args.transform_batch_size)
    train_labels = train_labels.to(device)
    try:
        selection = select_train_triple(train_features, train_labels, names)
    except SelectionFailure as error:
        write_json(json_path, {"method": encoder.provenance["method"], "dataset": "celeba",
                               "model": encoder.provenance, "selection_succeeded": False,
                               "failure_reason": str(error),
                               "exact_train_candidate_attempts": error.attempts,
                               "ssl_subspace": ssl_record})
        print(f"Selection failed honestly; saved {json_path}")
        return json_path, None
    indices, rows = selection["indices"], selection["selected_rows"]
    _, train_cells = measure_cell_centroids(
        selection["whitener"].transform(train_features[rows]),
        train_labels[rows][:, indices], selection["box"]["axes"])
    del train_features, train_labels

    print("4/6  Evaluate 20 held-out balanced resamples")
    test_features, test_labels = extract_dataset_features(
        test, encoder.encode, eval_transform, names, device=device,
        batch_size=batch_size, max_samples=args.max_samples)
    test_features = selection["whitener"].transform(transform_in_chunks(
        test_features, ssl_map, device, args.transform_batch_size))
    test_labels = test_labels.to(device)[:, indices]
    _, all_test_cells = measure_cell_centroids(
        test_features, test_labels, selection["box"]["axes"])
    all_test_diagnostics = corner_diagnostics(
        all_test_cells, selection["box"]["predicted_corners"])
    all_test_side_lengths = side_length_diagnostics(
        all_test_cells, selection["box"]["predicted_corners"], selection["names"])
    records = [evaluate_test_seed(test_features, test_labels, selection, seed)
               for seed in TEST_SEEDS]
    primary, stability = records[0], summarize_stability(records, selection["names"])

    print("5/6  Save paper-compatible numbers")
    train_geometry = _named_geometry(selection["crossfit_probe_geometry"], selection["names"],
                                     "selected_using_same_probe_observations")
    test_geometry = _named_geometry(primary["crossfit_probe_geometry"], selection["names"],
                                    "frozen_from_independent_training_split")
    capture = selection["box"]["capture_B"]
    payload = {
        "method": encoder.provenance["method"], "dataset": "celeba",
        "model": encoder.provenance, "selection_succeeded": True,
        "selected_triple": selection["names"],
        "protocol": {
            "analysis_protocol": "minimal_paper_accurate_celeba_v2",
            "selection_split": "train", "evaluation_split": "test",
            "population": "uniform_over_selected_eight_attribute_cells",
            "triple_and_geometry_frozen_before_test": True,
            "capture_estimator": "symmetrized_split_half_cross_gram",
            "whitening_fit_independent_of_train_probe_folds": True,
            "test_exact_whiteness_claimed": False,
            "post_selection_unbiasedness_claimed": False,
            "test_resamples_are_correlated_not_independent_replications": True,
            "figure_centroids_use_all_held_out_samples_in_each_cell": True,
            "fixed_test_criteria": {"max_pairwise_abs_cos": MAX_TEST_COSINE,
                                    "min_capture_B": MIN_TEST_CAPTURE,
                                    "max_normalized_centroid_rmse": MAX_TEST_RMSE,
                                    "min_cell_count": MIN_TEST_CELL}},
        "samples": {"train": min(len(train), args.max_samples or len(train)),
                    "test": min(len(test), args.max_samples or len(test)),
                    "raw_feature_dimension": raw_dimension,
                    "feature_extraction_batch_size": batch_size,
                    "max_samples_diagnostic_cap": args.max_samples,
                    "train_fingerprint": getattr(train, "_fingerprint", None),
                    "test_fingerprint": getattr(test, "_fingerprint", None)},
        "ssl_subspace": ssl_record,
        "train_selection": {
            "triple_names": selection["names"],
            "exact_train_candidate_attempts": selection["exact_attempts"],
            "balance": selection["balance"],
            "whitened_dimension": selection["whitener"].matrix.shape[1],
            "crossfit_probe_geometry": train_geometry,
            "capture_B": capture,
            "sqrt_capture_B_half_sides": [math.sqrt(value) for value in capture],
            "full_edge_lengths": [2 * math.sqrt(value) for value in capture],
            "mean_predicted_full_edge_length": sum(2 * math.sqrt(value) for value in capture) / 3,
            "same_sample_plug_in_capture_B": selection["box"]["plug_in_capture_B"],
            "fitted_task_axis_cosine_matrix": selection["box"]["cosine_matrix"],
            "predicted_box": selection["box"]["predicted_corners"],
            "train_cell_centroids": train_cells},
        "test_evaluation": {"triple_names": selection["names"],
                            "box": all_test_cells,
                            "predicted_box": selection["box"]["predicted_corners"],
                            "crossfit_probe_geometry": test_geometry,
                            "all_held_out_corner_counts": [row["count"] for row in all_test_cells],
                            "all_held_out_total_samples": sum(row["count"] for row in all_test_cells)},
        "test_box_diagnostics": all_test_diagnostics,
        "test_side_length_diagnostics": all_test_side_lengths,
        "headline_criteria_passed": primary["headline_criteria_passed"],
        "test_stability": {**stability, "records": [
            {"seed": row["seed"], "samples_per_cell": row["samples_per_cell"],
             "capture_B": row["crossfit_probe_geometry"]["capture_B"],
             "maximum_absolute_cosine": row["crossfit_probe_geometry"]["maximum_absolute_cosine"],
             "normalized_centroid_rmse": row["corner_diagnostics"]["normalized_centroid_rmse"],
             "mean_empirical_edge_length": row["side_length_diagnostics"]["mean_empirical_edge_length"],
             "mean_predicted_edge_length": row["side_length_diagnostics"]["mean_predicted_edge_length"],
             "headline_criteria_passed": row["headline_criteria_passed"]}
            for row in records]}}
    write_json(json_path, payload)

    print("6/6  Save paper-style figure")
    subtitles = {"vicreg_celeba": "VICReg, pretrained on CelebA",
                 "ijepa_celeba": "I-JEPA, pretrained on CelebA",
                 "vicreg_imagenet": "VICReg, pretrained on ImageNet-1K",
                 "ijepa_imagenet": "I-JEPA, pretrained on ImageNet-1K"}
    plot_hyperrectangle(figure_path, selection["names"], all_test_cells,
                        selection["box"]["predicted_corners"], subtitle=subtitles[args.model],
                        diagnostics=all_test_diagnostics,
                        maximum_cosine=test_geometry["max_abs_cos"],
                        side_lengths=all_test_side_lengths,
                        passed=primary["headline_criteria_passed"])
    print(f"Selected: {selection['names']}")
    print(f"Held-out normalized RMSE: {all_test_diagnostics['normalized_centroid_rmse']:.4f}")
    print(f"Mean edges empirical/predicted: {all_test_side_lengths['mean_empirical_edge_length']:.4f}/"
          f"{all_test_side_lengths['mean_predicted_edge_length']:.4f}")
    print(f"Stability: {stability['pass_count']}/{stability['n_resamples']} passed")
    print(json_path); print(figure_path.with_suffix(".pdf")); print(figure_path)
    return json_path, figure_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--weights", default=None,
                        help="optional local override; otherwise download the published checkpoint")
    parser.add_argument("--model-cache-dir", default=None,
                        help="optional model-download cache directory")
    parser.add_argument("--out-dir", default="hyperrectangle_output")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="override safe per-model default")
    parser.add_argument("--transform-batch-size", type=int, default=4096)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="diagnostic subset only; omit for a paper run")
    run_experiment(parser.parse_args())


if __name__ == "__main__":
    main()
