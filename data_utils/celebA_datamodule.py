from dataclasses import dataclass
from typing import Optional
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, DistributedSampler
from datasets import load_dataset
from data_utils import get_transforms


@dataclass
class CelebACfg:
    # General HF params
    name: str = "celeba"
    hf_repo: str = "flwrlabs/celeba"
    hf_cache_dir: str = "./data/hf_cache"
    hf_name: Optional[str] = None
    
    # Logic Switch
    method: str = "vicreg"  # options: "vicreg", "ijepa"
    
    # Image params
    img_size: int = 224
    patch_size: int = 16  # For I-JEPA
    batch_size: int = 128
    num_workers: int = 8
    
    # Split names
    train_split: str = "train"
    test_split: str = "test"
    image_key: str = "image"
    label_key: str = "label"
    
    # VICReg specific
    num_views: int = 2
    
    # I-JEPA specific
    n_target_blocks: int = 4


class CelebADataModule(pl.LightningDataModule):
    def __init__(self, cfg: CelebACfg):
        super().__init__()
        self.cfg = cfg
        self.ds_train = None
        self.ds_test = None
        
        # Setup Transforms
        self.train_tfms, self.test_tfms = get_transforms(
            method=self.cfg.method, 
            dataset=self.cfg.name,
            img_size=self.cfg.img_size,
        )

    def prepare_data(self):
        # Download datasets if not already present
        load_dataset(
            self.cfg.hf_repo, name=self.cfg.hf_name, 
            split=self.cfg.train_split, cache_dir=self.cfg.hf_cache_dir
        )
        load_dataset(
            self.cfg.hf_repo, name=self.cfg.hf_name, 
            split=self.cfg.test_split, cache_dir=self.cfg.hf_cache_dir
        )

    def setup(self, stage: Optional[str] = None):
        self.ds_train = load_dataset(
            self.cfg.hf_repo, name=self.cfg.hf_name,
            split=self.cfg.train_split, cache_dir=self.cfg.hf_cache_dir,
        )
        self.ds_test = load_dataset(
            self.cfg.hf_repo, name=self.cfg.hf_name,
            split=self.cfg.test_split, cache_dir=self.cfg.hf_cache_dir,
        )

        # Grid logic for I-JEPA
        if self.cfg.img_size % self.cfg.patch_size != 0:
            raise ValueError(
                f"img_size={self.cfg.img_size} must be divisible by "
                f"patch_size={self.cfg.patch_size}"
            )
        self.grid_h = self.cfg.img_size // self.cfg.patch_size
        self.grid_w = self.cfg.img_size // self.cfg.patch_size
        self.num_patches = self.grid_h * self.grid_w

    # ──────────────────────────────────────────────────────────────
    #  Collate functions
    # ──────────────────────────────────────────────────────────────
    def _collate_vicreg(self, batch):
        """Standard multi-view collate. Returns (views, labels, None, None)."""
        views = [[] for _ in range(self.cfg.num_views)]
        labels = []
        for ex in batch:
            img = ex[self.cfg.image_key].convert("RGB")
            for v in range(self.cfg.num_views):
                views[v].append(self.train_tfms(img))
            labels.append(ex.get(self.cfg.label_key, -1))

        return [torch.stack(v) for v in views], torch.tensor(labels, dtype=torch.long), None, None

    def _collate_ijepa(self, batch):
        """Block-masking collate. Returns ([images], labels, idx_enc, idx_pred)."""
        images, labels = [], []
        for ex in batch:
            img = ex[self.cfg.image_key].convert("RGB")
            images.append(self.train_tfms(img))
            labels.append(ex.get(self.cfg.label_key, -1))

        images = torch.stack(images)
        idx_enc, idx_pred = self._generate_jepa_masks(len(batch))

        return [images], torch.tensor(labels, dtype=torch.long), idx_enc, idx_pred

    def _collate_eval(self, batch):
        """Single-view eval collate. Returns ([images], labels, None, None)."""
        images = [self.test_tfms(ex[self.cfg.image_key].convert("RGB")) for ex in batch]
        labels = [ex.get(self.cfg.label_key, -1) for ex in batch]
        return [torch.stack(images)], torch.tensor(labels, dtype=torch.long), None, None

    # ──────────────────────────────────────────────────────────────
    #  I-JEPA mask generation (placeholder — wire in your real logic)
    # ──────────────────────────────────────────────────────────────
    def _generate_jepa_masks(self, batch_size: int):
        """
        Generate context encoder indices and target block indices.

        Returns:
            idx_enc:  [B, N_ctx]              — patch indices kept by encoder (+1 for CLS offset)
            idx_pred: [B, n_target_blocks, N_tgt] — patch indices for each target block (+1 for CLS offset)
        """
        B = batch_size
        gh, gw = self.grid_h, self.grid_w
        num_patches = self.num_patches
        npred = self.cfg.n_target_blocks

        # Sample block shapes ONCE per batch (Meta-style)
        ctx_h, ctx_w = sample_block_shape(
            gh, gw, scale_range=(0.85, 1.0), aspect_ratio_range=(1.0, 1.0),
            num_patches=num_patches,
        )
        tgt_h, tgt_w = sample_block_shape(
            gh, gw, scale_range=(0.15, 0.2), aspect_ratio_range=(0.75, 1.5),
            num_patches=num_patches,
        )

        idx_enc_list = []
        idx_pred_list = []

        for _ in range(B):
            # ── Context mask ──────────────────────────────────────
            mask_ctx = torch.zeros((gh, gw), dtype=torch.int32)
            top_c, left_c = sample_block_position(gh, gw, ctx_h, ctx_w)
            mask_ctx[top_c : top_c + ctx_h, left_c : left_c + ctx_w] = 1

            # ── Target masks (separate blocks) ────────────────────
            tgt_masks = []
            tgt_indices = []
            for _ in range(npred):
                mask_t = torch.zeros((gh, gw), dtype=torch.int32)
                top_t, left_t = sample_block_position(gh, gw, tgt_h, tgt_w)
                mask_t[top_t : top_t + tgt_h, left_t : left_t + tgt_w] = 1
                tgt_masks.append(mask_t)
                # +1 offset to account for CLS token at position 0
                pred_tok = torch.nonzero(mask_t.flatten()).squeeze(-1) + 1
                tgt_indices.append(pred_tok)

            # ── Encoder mask = context minus union of all targets ──
            mask_tgt_union = torch.stack(tgt_masks).sum(dim=0).clamp(max=1)
            mask_enc = mask_ctx * (1 - mask_tgt_union)
            enc_tok = torch.nonzero(mask_enc.flatten()).squeeze(-1) + 1

            idx_enc_list.append(enc_tok)
            idx_pred_list.append(tgt_indices)

        # ── Batch via truncation ──────────────────────────────────
        # Truncate encoder indices to the shortest across the batch
        min_enc_len = min(x.numel() for x in idx_enc_list)
        idx_enc = torch.stack(
            [x[:min_enc_len] for x in idx_enc_list], dim=0
        )  # [B, N_ctx]

        # Truncate each target block separately to the shortest
        min_pred_len = min(
            idx_pred_list[b][k].numel()
            for b in range(B)
            for k in range(npred)
        )
        idx_pred = torch.stack([
            torch.stack(
                [idx_pred_list[b][k][:min_pred_len] for k in range(npred)],
                dim=0,
            )
            for b in range(B)
        ], dim=0)  # [B, npred, N_tgt]

        return idx_enc, idx_pred

    # ──────────────────────────────────────────────────────────────
    #  Shared DataLoader factory
    # ──────────────────────────────────────────────────────────────
    def _make_loader(self, ds, train: bool, collate_fn=None, shuffle: bool = False,
                     num_workers: int = 0, distributed: bool = True):
        is_ddp = (
            distributed
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )
        sampler = DistributedSampler(ds, shuffle=train) if is_ddp else None
        return DataLoader(
            ds,
            batch_size=self.cfg.batch_size,
            shuffle=(sampler is None and shuffle),
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
            drop_last=train,
            collate_fn=collate_fn,
            prefetch_factor=4 if num_workers > 0 else None,
        )

    # ──────────────────────────────────────────────────────────────
    #  DataLoaders
    # ──────────────────────────────────────────────────────────────
    def train_dataloader(self):
        collate_fn = self._collate_ijepa if self.cfg.method == "ijepa" else self._collate_vicreg
        return self._make_loader(self.ds_train, train=True, collate_fn=collate_fn, shuffle=True, num_workers=self.cfg.num_workers)

    def paired_train_dataloader(self):
        """Analysis loader for SSL subspace estimation with two augmented views."""
        return self._make_loader(self.ds_train, train=True, collate_fn=self._collate_vicreg, shuffle=True, num_workers=self.cfg.num_workers)

    def val_dataloader(self):
        return self._make_loader(self.ds_test, train=False, collate_fn=self._collate_eval, shuffle=False)

    def test_dataloader(self):
        return self._make_loader(self.ds_test, train=False, collate_fn=self._collate_eval, shuffle=False, num_workers=self.cfg.num_workers)

    def probe_train_dataloader(self):
        return self._make_loader(
            self.ds_train, train=False, collate_fn=self._collate_eval,
            shuffle=True, num_workers=4, distributed=False,
        )

    def probe_test_dataloader(self):
        return self._make_loader(
            self.ds_test, train=False, collate_fn=self._collate_eval,
            shuffle=False, num_workers=4, distributed=False,
        )

import math
import random
# ──────────────────────────────────────────────────────────────
#  Block-masking helpers (module-level)
# ──────────────────────────────────────────────────────────────
def sample_block_shape(gh, gw, scale_range, aspect_ratio_range, num_patches):
    """Sample a random block shape (h, w) on the patch grid."""
    s_min, s_max = scale_range
    ar_min, ar_max = aspect_ratio_range

    # target area in patches
    area = random.uniform(s_min, s_max) * num_patches
    log_ar = random.uniform(math.log(ar_min), math.log(ar_max))
    ar = math.exp(log_ar)

    h = int(round(math.sqrt(area / ar)))
    w = int(round(math.sqrt(area * ar)))
    h = max(1, min(h, gh))
    w = max(1, min(w, gw))
    return h, w


def sample_block_position(gh, gw, bh, bw):
    """Sample a random top-left corner for a block of size (bh, bw)."""
    top = random.randint(0, gh - bh)
    left = random.randint(0, gw - bw)
    return top, left
