"""Lightning datamodule for MPI3D two-view SSL training (thin wrapper over mpi3d_core)."""
from typing import Optional

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, DistributedSampler

from data_utils.mpi3d_core import (
    Mpi3dCfg, Mpi3dEvalDataset, Mpi3dPairDataset, build_arrays,
    collate_eval, collate_pairs,
)


class Mpi3dDataModule(pl.LightningDataModule):
    def __init__(self, cfg: Mpi3dCfg):
        super().__init__()
        self.cfg = cfg
        self._arrays = None

    def setup(self, stage: Optional[str] = None):
        if self._arrays is None:
            self._arrays = build_arrays(self.cfg)
        imgs, _, bits, group_of, groups = self._arrays
        self.ds_train = Mpi3dPairDataset(imgs, bits, group_of, groups, self.cfg)
        self.ds_eval = Mpi3dEvalDataset(imgs, bits, self.cfg)
        print(f"MPI3D: {imgs.shape[0]} images, {len(groups)} pair-groups "
              f"(pair_mode={self.cfg.pair_mode}, content="
              f"{list(self.cfg.content_factors or self.cfg.task_factors)})")

    def _make_loader(self, ds, train, collate_fn, shuffle, num_workers,
                     distributed=True):
        is_ddp = (
            distributed
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )
        sampler = DistributedSampler(ds, shuffle=train) if is_ddp else None
        return DataLoader(
            ds, batch_size=self.cfg.batch_size,
            shuffle=(sampler is None and shuffle), sampler=sampler,
            num_workers=num_workers, pin_memory=True,
            persistent_workers=(num_workers > 0), drop_last=train,
            collate_fn=collate_fn, prefetch_factor=4 if num_workers > 0 else None)

    def train_dataloader(self):
        return self._make_loader(self.ds_train, True, collate_pairs, True, self.cfg.num_workers)

    def paired_train_dataloader(self):
        return self._make_loader(self.ds_train, True, collate_pairs, True, self.cfg.num_workers)

    def probe_train_dataloader(self):
        return self._make_loader(
            self.ds_eval, False, collate_eval, True, 4, distributed=False
        )

    def probe_test_dataloader(self):
        return self._make_loader(
            self.ds_eval, False, collate_eval, False, 4, distributed=False
        )
