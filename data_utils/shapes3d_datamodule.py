"""Lightning datamodule for 3DShapes two-view SSL training.

Thin wrapper over ``shapes3d_core`` (Lightning-free). Yields the project-standard
``([view0, view1], labels, idx_enc, idx_pred)`` batches that ``LightlyVICReg``
consumes. Only VICReg is wired (no I-JEPA masking on these 64x64 images).
"""
from typing import Optional

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, DistributedSampler

from data_utils.shapes3d_core import (
    Shapes3DCfg,
    Shapes3DEvalDataset,
    Shapes3DPairDataset,
    build_arrays,
    collate_eval,
    collate_pairs,
)


class Shapes3DDataModule(pl.LightningDataModule):
    def __init__(self, cfg: Shapes3DCfg):
        super().__init__()
        self.cfg = cfg
        self._arrays = None  # (imgs, latents, bits, group_of, groups)

    def setup(self, stage: Optional[str] = None):
        if self._arrays is None:
            self._arrays = build_arrays(self.cfg)
        imgs, _, bits, group_of, groups = self._arrays
        self.ds_train = Shapes3DPairDataset(imgs, bits, group_of, groups, self.cfg)
        self.ds_eval = Shapes3DEvalDataset(imgs, bits, self.cfg)
        print(f"Shapes3D: {imgs.shape[0]} images, {len(groups)} pair-groups "
              f"(pair_mode={self.cfg.pair_mode}, tasks={list(self.cfg.task_factors)}, "
              f"shapes={list(self.cfg.shapes)})")

    def _make_loader(self, ds, train: bool, collate_fn, shuffle: bool,
                     num_workers: int, distributed: bool = True):
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

    def train_dataloader(self):
        return self._make_loader(self.ds_train, train=True, collate_fn=collate_pairs,
                                 shuffle=True, num_workers=self.cfg.num_workers)

    def paired_train_dataloader(self):
        return self._make_loader(self.ds_train, train=True, collate_fn=collate_pairs,
                                 shuffle=True, num_workers=self.cfg.num_workers)

    def probe_train_dataloader(self):
        return self._make_loader(
            self.ds_eval, train=False, collate_fn=collate_eval,
            shuffle=True, num_workers=4, distributed=False,
        )

    def probe_test_dataloader(self):
        return self._make_loader(
            self.ds_eval, train=False, collate_fn=collate_eval,
            shuffle=False, num_workers=4, distributed=False,
        )
