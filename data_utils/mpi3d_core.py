"""MPI3D two-view core -- pure torch / numpy, Lightning-free. Mirrors shapes3d_core.

MPI3D is a Cartesian product of seven factors over 1,036,800 RGB 64x64 images:

    [object_color(6), object_shape(6), object_size(2), camera_height(3),
     background_color(3), horizontal_axis(40), vertical_axis(40)]

Crucially it has TWO position axes (horizontal/vertical = where the held object is)
-- the dSprites-style factors that SSL captures cleanest -- plus size, camera height,
shape and colour. So a diverse task family of the *distinct* factors can need 5-6
independent bottleneck dimensions, more than 3DShapes' ~4.

The npz stores only 'images' [N,64,64,3] uint8 in canonical factor order (last factor
varies fastest); the factor labels are derived from the flat index.
"""
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Canonical MPI3D factor order + sizes (product = 1,036,800).
FACTOR_NAMES = ["obj_color", "obj_shape", "obj_size", "camera",
                "bg_color", "posX", "posY"]
FACTOR_SIZES = [6, 6, 2, 3, 3, 40, 40]
FACTOR_COL = {n: i for i, n in enumerate(FACTOR_NAMES)}
COL_SHAPE = FACTOR_COL["obj_shape"]          # for analysis-driver compatibility
TASK_NAMES = ["posX", "posY", "obj_size"]


@dataclass
class Mpi3dCfg:
    name: str = "mpi3d"
    npz_path: str = "./data/mpi3d/mpi3d_toy.npz"

    method: str = "vicreg"

    img_size: int = 64
    batch_size: int = 512
    num_workers: int = 8
    num_views: int = 2

    task_factors: Sequence[str] = ("posX", "posY", "obj_size")
    content_factors: Optional[Sequence[str]] = None  # shared in a positive pair
    # ">= threshold" splits per factor (indices).
    thresholds: Optional[dict] = None                # {factor: index}; else midpoints
    keep_levels: Optional[dict] = None

    pair_mode: str = "exact"
    noise_std: float = 0.0

    normalize_mean: float = 0.5
    normalize_std: float = 0.5

    max_samples: Optional[int] = 200000
    seed: int = 6


_DEFAULT_THRESH = {"obj_color": 3, "obj_shape": 3, "obj_size": 1, "camera": 2,
                   "bg_color": 2, "posX": 20, "posY": 20}


def _threshold(cfg: Mpi3dCfg, name: str) -> int:
    if cfg.thresholds and name in cfg.thresholds:
        return int(cfg.thresholds[name])
    return _DEFAULT_THRESH[name]


# ---------------------------------------------------------------------------
def _latents_from_index(idx: np.ndarray) -> np.ndarray:
    """Flat index -> [.,7] factor indices (last factor varies fastest)."""
    lat = np.empty((idx.shape[0], len(FACTOR_SIZES)), dtype=np.int64)
    rem = idx.astype(np.int64).copy()
    for c in range(len(FACTOR_SIZES) - 1, -1, -1):
        lat[:, c] = rem % FACTOR_SIZES[c]
        rem //= FACTOR_SIZES[c]
    return lat


def load_mpi3d(npz_path: str, max_samples: Optional[int], seed: int,
               keep_levels: Optional[dict] = None) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)
    key = "images" if "images" in data else data.files[0]
    images = data[key]                                   # [N,64,64,3] uint8
    n = images.shape[0]
    latents = _latents_from_index(np.arange(n))
    keep = np.ones(n, dtype=bool)
    if keep_levels:
        for fname, levels in keep_levels.items():
            keep &= np.isin(latents[:, FACTOR_COL[fname]], np.asarray(list(levels)))
    idx = np.where(keep)[0]
    if max_samples is not None and idx.size > max_samples:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(idx, size=max_samples, replace=False))
    imgs_sel = np.ascontiguousarray(images[idx])
    return imgs_sel, np.ascontiguousarray(latents[idx])


def _factor_bit(latents: np.ndarray, name: str, cfg: Mpi3dCfg) -> np.ndarray:
    return (latents[:, FACTOR_COL[name]] >= _threshold(cfg, name)).astype(np.int64)


def derive_task_bits(latents: np.ndarray, cfg: Mpi3dCfg) -> np.ndarray:
    if len(cfg.task_factors) != 3:
        raise ValueError(f"task_factors must name exactly 3, got {cfg.task_factors}")
    return np.stack([_factor_bit(latents, f, cfg) for f in cfg.task_factors], axis=1)


def build_groups(latents: np.ndarray, bits: np.ndarray, cfg: Mpi3dCfg
                 ) -> Tuple[np.ndarray, List[np.ndarray]]:
    if cfg.pair_mode == "granular":
        key = bits
    elif cfg.pair_mode == "exact":
        factors = cfg.content_factors if cfg.content_factors else cfg.task_factors
        key = latents[:, [FACTOR_COL[f] for f in factors]]
    else:
        raise ValueError(f"Unknown pair_mode={cfg.pair_mode!r}")
    _, group_of = np.unique(key, axis=0, return_inverse=True)
    group_of = group_of.astype(np.int64).reshape(-1)
    n_groups = int(group_of.max()) + 1 if group_of.size else 0
    groups = [np.where(group_of == g)[0] for g in range(n_groups)]
    return group_of, groups


# ---------------------------------------------------------------------------
def _to_tensor(img_u8: np.ndarray, cfg: Mpi3dCfg, rng=None) -> torch.Tensor:
    x = torch.from_numpy(np.ascontiguousarray(img_u8)).float() / 255.0   # [64,64,3]
    x = x.permute(2, 0, 1).contiguous()
    if cfg.img_size != x.shape[-1]:
        x = torch.nn.functional.interpolate(
            x.unsqueeze(0), size=(cfg.img_size, cfg.img_size),
            mode="bilinear", align_corners=False).squeeze(0)
    if cfg.noise_std > 0:
        x = x + torch.randn_like(x) * cfg.noise_std
    return (x - cfg.normalize_mean) / cfg.normalize_std


class Mpi3dPairDataset(Dataset):
    def __init__(self, imgs, bits, group_of, groups, cfg):
        self.imgs, self.bits = imgs, bits
        self.group_of, self.groups, self.cfg = group_of, groups, cfg

    def __len__(self): return self.imgs.shape[0]

    def __getitem__(self, i):
        members = self.groups[self.group_of[i]]
        j = int(members[np.random.randint(members.shape[0])])
        v0 = _to_tensor(self.imgs[i], self.cfg)
        v1 = _to_tensor(self.imgs[j], self.cfg)
        bits = torch.from_numpy(np.ascontiguousarray(self.bits[i])).long()
        return v0, v1, bits


class Mpi3dEvalDataset(Dataset):
    def __init__(self, imgs, bits, cfg):
        self.imgs, self.bits, self.cfg = imgs, bits, cfg

    def __len__(self): return self.imgs.shape[0]

    def __getitem__(self, i):
        v = _to_tensor(self.imgs[i], self.cfg)
        bits = torch.from_numpy(np.ascontiguousarray(self.bits[i])).long()
        return v, bits


def collate_pairs(batch):
    v0 = torch.stack([b[0] for b in batch]); v1 = torch.stack([b[1] for b in batch])
    return [v0, v1], torch.stack([b[2] for b in batch]), None, None


def collate_eval(batch):
    return torch.stack([b[0] for b in batch]), torch.stack([b[1] for b in batch])


def build_arrays(cfg: Mpi3dCfg):
    imgs, latents = load_mpi3d(cfg.npz_path, cfg.max_samples, cfg.seed,
                               keep_levels=cfg.keep_levels)
    bits = derive_task_bits(latents, cfg)
    group_of, groups = build_groups(latents, bits, cfg)
    return imgs, latents, bits, group_of, groups


def make_eval_loader(cfg, imgs=None, bits=None, shuffle=False, num_workers=None):
    if imgs is None:
        imgs, _, bits, _, _ = build_arrays(cfg)
    ds = Mpi3dEvalDataset(imgs, bits, cfg)
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle,
                      num_workers=cfg.num_workers if num_workers is None else num_workers,
                      pin_memory=True, drop_last=False, collate_fn=collate_eval)


def make_paired_loader(cfg, imgs=None, bits=None, group_of=None, groups=None,
                       shuffle=True, num_workers=None):
    if imgs is None:
        imgs, _, bits, group_of, groups = build_arrays(cfg)
    ds = Mpi3dPairDataset(imgs, bits, group_of, groups, cfg)
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle,
                      num_workers=cfg.num_workers if num_workers is None else num_workers,
                      pin_memory=True, drop_last=True, collate_fn=collate_pairs)
