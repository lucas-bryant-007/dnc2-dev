"""DSprites two-view core (pure torch / numpy -- no Lightning).

This module builds the controlled "two-view" distribution for the hyper-rectangle
experiment. DSprites is a Cartesian product of independent factors

    [color(1), shape(3), scale(6), orientation(40), posX(32), posY(32)]

We pick three *recoverable* binary tasks -- shape, scale, posX -- and treat
orientation + posY (and within-bin variation) as the *nuisance* that the two
views differ on. A positive pair therefore shares the task content and differs
only in nuisance, which is exactly the two-view operator whose SSL optimum lands
the eight granular-task centroids on the corners of an axis-aligned box
(Thm. 4.4). No pixel augmentation is needed: the pairing *is* the augmentation.

Kept Lightning-free so the analysis driver can reuse the loaders on CPU.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Column order of DSprites ``latents_classes``.
COL_COLOR, COL_SHAPE, COL_SCALE, COL_ORIENT, COL_POSX, COL_POSY = range(6)
TASK_NAMES = ["shape", "scale", "posX"]


@dataclass
class DSpritesCfg:
    name: str = "dsprites"
    npz_path: str = "./data/dsprites/dsprites.npz"

    # Logic switch (set by train.py; only "vicreg" is wired for training).
    method: str = "vicreg"

    # Image / loader params.
    img_size: int = 64
    batch_size: int = 512
    num_workers: int = 8
    num_views: int = 2

    # Task definition.
    shapes: Sequence[int] = (0, 1)          # which DSprites shapes to keep (0=sq,1=ell,2=heart)
    scale_threshold: int = 3                # scale index >= this -> "large"
    posx_threshold: int = 16                # posX index >= this -> "right"

    # Two-view pairing.
    #   "granular": pair shares the 3 task *bits* (tightest clusters, cleanest box)
    #   "exact":    pair shares exact (shape, scale, posX) (more continuous spread)
    pair_mode: str = "granular"
    noise_std: float = 0.0                  # optional Gaussian pixel noise per view

    # Normalization ({0,1} pixels -> ~[-1,1]).
    normalize_mean: float = 0.5
    normalize_std: float = 0.5

    # Optional subsample for speed (None = use all).
    max_samples: Optional[int] = 150000
    seed: int = 6


# ---------------------------------------------------------------------------
# Loading / task derivation / grouping
# ---------------------------------------------------------------------------
def load_dsprites(npz_path: str, shapes: Sequence[int], max_samples: Optional[int],
                  seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (imgs[N,64,64] uint8, latents_classes[N,6] int) filtered + subsampled."""
    with np.load(npz_path, allow_pickle=True, mmap_mode="r") as data:
        imgs = data["imgs"]            # [737280, 64, 64] uint8
        latents = np.asarray(data["latents_classes"])  # [737280, 6] int

    keep = np.isin(latents[:, COL_SHAPE], np.asarray(list(shapes)))
    idx = np.where(keep)[0]
    if max_samples is not None and idx.size > max_samples:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(idx, size=max_samples, replace=False))
    # Materialize the selected images into RAM (mmap slice -> array).
    imgs_sel = np.ascontiguousarray(imgs[idx])
    latents_sel = np.ascontiguousarray(latents[idx])
    return imgs_sel, latents_sel


def derive_task_bits(latents: np.ndarray, cfg: DSpritesCfg) -> np.ndarray:
    """[N,6] latents -> [N,3] binary task labels (shape, scale, posX) in {0,1}."""
    shapes = list(cfg.shapes)
    pos_shape = shapes[-1]  # last listed shape is the "positive" class
    shape_bit = (latents[:, COL_SHAPE] == pos_shape).astype(np.int64)
    scale_bit = (latents[:, COL_SCALE] >= cfg.scale_threshold).astype(np.int64)
    posx_bit = (latents[:, COL_POSX] >= cfg.posx_threshold).astype(np.int64)
    return np.stack([shape_bit, scale_bit, posx_bit], axis=1)


def build_groups(latents: np.ndarray, bits: np.ndarray, pair_mode: str
                 ) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Return (group_of[N], groups) where groups[g] = indices sharing the key.

    ``granular`` keys on the 3 task bits (8 groups); ``exact`` keys on the exact
    (shape, scale, posX) tuple. The nuisance factors (orientation, posY, and --
    in granular mode -- within-bin scale/posX) vary freely within a group.
    """
    if pair_mode == "granular":
        key = bits
    elif pair_mode == "exact":
        key = latents[:, [COL_SHAPE, COL_SCALE, COL_POSX]]
    else:
        raise ValueError(f"Unknown pair_mode={pair_mode!r} (use 'granular' or 'exact')")
    _, group_of = np.unique(key, axis=0, return_inverse=True)
    group_of = group_of.astype(np.int64).reshape(-1)
    n_groups = int(group_of.max()) + 1 if group_of.size else 0
    groups = [np.where(group_of == g)[0] for g in range(n_groups)]
    return group_of, groups


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------
def _to_tensor(img_u8: np.ndarray, cfg: DSpritesCfg,
               rng: Optional[np.random.Generator] = None) -> torch.Tensor:
    """uint8 [64,64] {0,1} -> float [3,H,W] normalized (optional noise)."""
    x = torch.from_numpy(np.ascontiguousarray(img_u8)).float()  # [64,64]
    x = x.unsqueeze(0).repeat(3, 1, 1)                           # [3,64,64]
    if cfg.img_size != x.shape[-1]:
        x = torch.nn.functional.interpolate(
            x.unsqueeze(0), size=(cfg.img_size, cfg.img_size),
            mode="bilinear", align_corners=False).squeeze(0)
    if cfg.noise_std > 0:
        noise = torch.randn_like(x) * cfg.noise_std
        x = x + noise
    return (x - cfg.normalize_mean) / cfg.normalize_std


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
class DSpritesPairDataset(Dataset):
    """Two-view positives sharing task content, differing in nuisance."""

    def __init__(self, imgs: np.ndarray, bits: np.ndarray,
                 group_of: np.ndarray, groups: List[np.ndarray], cfg: DSpritesCfg):
        self.imgs = imgs
        self.bits = bits
        self.group_of = group_of
        self.groups = groups
        self.cfg = cfg

    def __len__(self) -> int:
        return self.imgs.shape[0]

    def __getitem__(self, i: int):
        members = self.groups[self.group_of[i]]
        j = int(members[np.random.randint(members.shape[0])])
        v0 = _to_tensor(self.imgs[i], self.cfg)
        v1 = _to_tensor(self.imgs[j], self.cfg)
        bits = torch.from_numpy(np.ascontiguousarray(self.bits[i])).long()
        return v0, v1, bits


class DSpritesEvalDataset(Dataset):
    """Single view + the 3 binary task labels (for feature extraction)."""

    def __init__(self, imgs: np.ndarray, bits: np.ndarray, cfg: DSpritesCfg):
        self.imgs = imgs
        self.bits = bits
        self.cfg = cfg

    def __len__(self) -> int:
        return self.imgs.shape[0]

    def __getitem__(self, i: int):
        v = _to_tensor(self.imgs[i], self.cfg)
        bits = torch.from_numpy(np.ascontiguousarray(self.bits[i])).long()
        return v, bits


# ---------------------------------------------------------------------------
# Collates (match the project's (views, labels, idx_enc, idx_pred) convention)
# ---------------------------------------------------------------------------
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
# Convenience builders (used by the analysis driver; no Lightning needed)
# ---------------------------------------------------------------------------
def build_arrays(cfg: DSpritesCfg):
    imgs, latents = load_dsprites(cfg.npz_path, cfg.shapes, cfg.max_samples, cfg.seed)
    bits = derive_task_bits(latents, cfg)
    group_of, groups = build_groups(latents, bits, cfg.pair_mode)
    return imgs, latents, bits, group_of, groups


def make_eval_loader(cfg: DSpritesCfg, imgs=None, bits=None, shuffle=False,
                     num_workers: Optional[int] = None) -> DataLoader:
    if imgs is None:
        imgs, _, bits, _, _ = build_arrays(cfg)
    ds = DSpritesEvalDataset(imgs, bits, cfg)
    return DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=shuffle,
        num_workers=cfg.num_workers if num_workers is None else num_workers,
        pin_memory=True, drop_last=False, collate_fn=collate_eval)


def make_paired_loader(cfg: DSpritesCfg, imgs=None, bits=None, group_of=None,
                       groups=None, shuffle=True,
                       num_workers: Optional[int] = None) -> DataLoader:
    if imgs is None:
        imgs, _, bits, group_of, groups = build_arrays(cfg)
    ds = DSpritesPairDataset(imgs, bits, group_of, groups, cfg)
    return DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=shuffle,
        num_workers=cfg.num_workers if num_workers is None else num_workers,
        pin_memory=True, drop_last=True, collate_fn=collate_pairs)
