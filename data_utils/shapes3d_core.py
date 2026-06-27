"""3DShapes (Shapes3D) two-view core -- pure torch / numpy, Lightning-free.

Shapes3D is a Cartesian product of six independent factors

    [floor_hue(10), wall_hue(10), object_hue(10), scale(8), shape(4), orientation(15)]

over 480000 RGB 64x64 images. We pick three salient, nameable binary tasks --
default object_hue (object colour), shape, scale -- and treat the two background
hues + orientation as the nuisance the two views differ on. A positive pair shares
the task content and resamples the nuisance, exactly the two-view operator whose
SSL optimum lands the 2^3 granular-task centroids on the corners of an axis-aligned
box. Mirrors ``dsprites_core`` so the same analysis drivers apply.

Needs ``h5py`` to read the standard ``3dshapes.h5`` (keys: 'images', 'labels').
"""
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Column order of the Shapes3D label grid.
COL_FLOOR, COL_WALL, COL_OBJECT, COL_SCALE, COL_SHAPE, COL_ORIENT = range(6)
FACTOR_COL = {"floor_hue": COL_FLOOR, "wall_hue": COL_WALL, "object_hue": COL_OBJECT,
              "scale": COL_SCALE, "shape": COL_SHAPE, "orientation": COL_ORIENT}
FACTOR_SIZES = {"floor_hue": 10, "wall_hue": 10, "object_hue": 10,
                "scale": 8, "shape": 4, "orientation": 15}
# Default tasks: object colour / shape / size -- all salient and independent.
TASK_NAMES = ["object_hue", "shape", "scale"]


@dataclass
class Shapes3DCfg:
    name: str = "shapes3d"
    h5_path: str = "./data/shapes3d/3dshapes.h5"

    method: str = "vicreg"

    img_size: int = 64
    batch_size: int = 512
    num_workers: int = 8
    num_views: int = 2

    # Task definition: 3 binary tasks (content); the rest is nuisance.
    task_factors: Sequence[str] = ("object_hue", "shape", "scale")
    shapes: Sequence[int] = (0, 1, 2, 3)        # which shape categories to keep
    # ">= threshold" splits per factor (indices, not raw values).
    floor_hue_threshold: int = 5
    wall_hue_threshold: int = 5
    object_hue_threshold: int = 5
    scale_threshold: int = 4
    shape_threshold: int = 2                     # {cube,cylinder} vs {sphere,capsule}
    orient_threshold: int = 8
    # Optionally keep only extreme level indices per factor (crisp binary tasks).
    keep_levels: Optional[dict] = None

    pair_mode: str = "exact"                     # "granular" | "exact"
    noise_std: float = 0.0

    normalize_mean: float = 0.5
    normalize_std: float = 0.5

    max_samples: Optional[int] = 200000
    seed: int = 6


# ---------------------------------------------------------------------------
# Loading / task derivation / grouping
# ---------------------------------------------------------------------------
def load_shapes3d(h5_path: str, shapes: Sequence[int], max_samples: Optional[int],
                  seed: int, keep_levels: Optional[dict] = None
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Return (imgs[N,64,64,3] uint8, latent_indices[N,6] int) filtered + subsampled."""
    try:
        import h5py
    except ImportError as e:  # pragma: no cover
        raise ImportError("Reading Shapes3D needs h5py: pip install h5py") from e

    with h5py.File(h5_path, "r") as f:
        labels = np.asarray(f["labels"])                      # [N,6] float values
        # Map each factor's float values to 0-based indices (sorted order).
        latents = np.empty(labels.shape, dtype=np.int64)
        for c in range(labels.shape[1]):
            _, inv = np.unique(labels[:, c], return_inverse=True)
            latents[:, c] = inv

        keep = np.isin(latents[:, COL_SHAPE], np.asarray(list(shapes)))
        if keep_levels:
            for fname, levels in keep_levels.items():
                keep &= np.isin(latents[:, FACTOR_COL[fname]], np.asarray(list(levels)))
        idx = np.where(keep)[0]
        if max_samples is not None and idx.size > max_samples:
            rng = np.random.default_rng(seed)
            idx = np.sort(rng.choice(idx, size=max_samples, replace=False))
        imgs_sel = np.asarray(f["images"][idx])               # idx is sorted (h5py req.)
        latents_sel = np.ascontiguousarray(latents[idx])
    return imgs_sel, latents_sel


def _factor_bit(latents: np.ndarray, name: str, cfg: Shapes3DCfg) -> np.ndarray:
    """Binary {0,1} label for one named factor (>= its threshold)."""
    thresh = {"floor_hue": cfg.floor_hue_threshold, "wall_hue": cfg.wall_hue_threshold,
              "object_hue": cfg.object_hue_threshold, "scale": cfg.scale_threshold,
              "shape": cfg.shape_threshold, "orientation": cfg.orient_threshold}[name]
    return (latents[:, FACTOR_COL[name]] >= thresh).astype(np.int64)


def derive_task_bits(latents: np.ndarray, cfg: Shapes3DCfg) -> np.ndarray:
    """[N,6] latents -> [N,3] binary task labels for cfg.task_factors, in {0,1}."""
    if len(cfg.task_factors) != 3:
        raise ValueError(f"task_factors must name exactly 3 factors, got {cfg.task_factors}")
    return np.stack([_factor_bit(latents, f, cfg) for f in cfg.task_factors], axis=1)


def build_groups(latents: np.ndarray, bits: np.ndarray, cfg: Shapes3DCfg
                 ) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Return (group_of[N], groups). 'granular' keys on the 3 task bits; 'exact'
    keys on the exact values of the task factors (rest = nuisance)."""
    if cfg.pair_mode == "granular":
        key = bits
    elif cfg.pair_mode == "exact":
        key = latents[:, [FACTOR_COL[f] for f in cfg.task_factors]]
    else:
        raise ValueError(f"Unknown pair_mode={cfg.pair_mode!r} (use 'granular' or 'exact')")
    _, group_of = np.unique(key, axis=0, return_inverse=True)
    group_of = group_of.astype(np.int64).reshape(-1)
    n_groups = int(group_of.max()) + 1 if group_of.size else 0
    groups = [np.where(group_of == g)[0] for g in range(n_groups)]
    return group_of, groups


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------
def _to_tensor(img_u8: np.ndarray, cfg: Shapes3DCfg,
               rng: Optional[np.random.Generator] = None) -> torch.Tensor:
    """uint8 [64,64,3] -> float [3,H,W] normalized (optional noise)."""
    x = torch.from_numpy(np.ascontiguousarray(img_u8)).float() / 255.0  # [64,64,3]
    x = x.permute(2, 0, 1).contiguous()                                 # [3,64,64]
    if cfg.img_size != x.shape[-1]:
        x = torch.nn.functional.interpolate(
            x.unsqueeze(0), size=(cfg.img_size, cfg.img_size),
            mode="bilinear", align_corners=False).squeeze(0)
    if cfg.noise_std > 0:
        x = x + torch.randn_like(x) * cfg.noise_std
    return (x - cfg.normalize_mean) / cfg.normalize_std


# ---------------------------------------------------------------------------
# Datasets / collates (match the project's (views, labels, idx_enc, idx_pred))
# ---------------------------------------------------------------------------
class Shapes3DPairDataset(Dataset):
    """Two-view positives sharing task content, differing in nuisance."""

    def __init__(self, imgs, bits, group_of, groups, cfg: Shapes3DCfg):
        self.imgs, self.bits = imgs, bits
        self.group_of, self.groups, self.cfg = group_of, groups, cfg

    def __len__(self) -> int:
        return self.imgs.shape[0]

    def __getitem__(self, i: int):
        members = self.groups[self.group_of[i]]
        j = int(members[np.random.randint(members.shape[0])])
        v0 = _to_tensor(self.imgs[i], self.cfg)
        v1 = _to_tensor(self.imgs[j], self.cfg)
        bits = torch.from_numpy(np.ascontiguousarray(self.bits[i])).long()
        return v0, v1, bits


class Shapes3DEvalDataset(Dataset):
    """Single view + the 3 binary task labels (for feature extraction)."""

    def __init__(self, imgs, bits, cfg: Shapes3DCfg):
        self.imgs, self.bits, self.cfg = imgs, bits, cfg

    def __len__(self) -> int:
        return self.imgs.shape[0]

    def __getitem__(self, i: int):
        v = _to_tensor(self.imgs[i], self.cfg)
        bits = torch.from_numpy(np.ascontiguousarray(self.bits[i])).long()
        return v, bits


def collate_pairs(batch):
    v0 = torch.stack([b[0] for b in batch])
    v1 = torch.stack([b[1] for b in batch])
    bits = torch.stack([b[2] for b in batch])
    return [v0, v1], bits, None, None


def collate_eval(batch):
    v = torch.stack([b[0] for b in batch])
    bits = torch.stack([b[1] for b in batch])
    return v, bits


# ---------------------------------------------------------------------------
# Convenience builders (mirror dsprites_core; reused by analysis drivers)
# ---------------------------------------------------------------------------
def build_arrays(cfg: Shapes3DCfg):
    imgs, latents = load_shapes3d(cfg.h5_path, cfg.shapes, cfg.max_samples, cfg.seed,
                                  keep_levels=cfg.keep_levels)
    bits = derive_task_bits(latents, cfg)
    group_of, groups = build_groups(latents, bits, cfg)
    return imgs, latents, bits, group_of, groups


def make_eval_loader(cfg: Shapes3DCfg, imgs=None, bits=None, shuffle=False,
                     num_workers: Optional[int] = None) -> DataLoader:
    if imgs is None:
        imgs, _, bits, _, _ = build_arrays(cfg)
    ds = Shapes3DEvalDataset(imgs, bits, cfg)
    return DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=shuffle,
        num_workers=cfg.num_workers if num_workers is None else num_workers,
        pin_memory=True, drop_last=False, collate_fn=collate_eval)


def make_paired_loader(cfg: Shapes3DCfg, imgs=None, bits=None, group_of=None,
                       groups=None, shuffle=True,
                       num_workers: Optional[int] = None) -> DataLoader:
    if imgs is None:
        imgs, _, bits, group_of, groups = build_arrays(cfg)
    ds = Shapes3DPairDataset(imgs, bits, group_of, groups, cfg)
    return DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=shuffle,
        num_workers=cfg.num_workers if num_workers is None else num_workers,
        pin_memory=True, drop_last=True, collate_fn=collate_pairs)
