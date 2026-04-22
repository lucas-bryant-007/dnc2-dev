from dataclasses import dataclass
from typing import Optional, Union, List
from pathlib import Path

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.datasets import CelebA

from data_utils import get_transforms


@dataclass
class CelebACfg:
    name: str = "celeba"
    root: str = "./data"
    method: str = "vicreg"
    img_size: int = 224
    batch_size: int = 128
    num_workers: int = 8

    train_split: str = "train"
    val_split: str = "valid"
    test_split: str = "test"

    num_views: int = 2
    target_type: Union[str, List[str]] = "attr"
    download: bool = True


class CelebADataModule(pl.LightningDataModule):
    """
    Torchvision-backed CelebA DataModule.

    CelebA returns:
      image: PIL.Image
      target: depends on target_type
        - "attr" -> Tensor of shape [40]
        - "identity" -> int / tensor
        - "bbox" -> Tensor of shape [4]
        - "landmarks" -> Tensor of shape [10]
        - list[...] -> tuple of the above
    """

    def __init__(self, cfg: CelebACfg):
        super().__init__()
        self.cfg = cfg

        self.ds_train = None
        self.ds_val = None
        self.ds_test = None

        self.train_tfms, self.test_tfms = self._get_transforms(self.cfg.method, self.cfg.name)
        self.is_ddp = torch.distributed.is_available() and torch.distributed.is_initialized()

    def _get_transforms(self, method: str, dataset_name: str):
        return get_transforms(
            method=method,
            dataset="cifar" if "cifar" in (dataset_name or "").lower() else dataset_name,
        )

    def prepare_data(self):
        # Lightning runs this on rank 0 only
        CelebA(
            root=self.cfg.root,
            split=self.cfg.train_split,
            target_type=self.cfg.target_type,
            download=self.cfg.download,
        )
        CelebA(
            root=self.cfg.root,
            split=self.cfg.val_split,
            target_type=self.cfg.target_type,
            download=self.cfg.download,
        )
        CelebA(
            root=self.cfg.root,
            split=self.cfg.test_split,
            target_type=self.cfg.target_type,
            download=self.cfg.download,
        )

    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit"):
            self.ds_train = CelebA(
                root=self.cfg.root,
                split=self.cfg.train_split,
                target_type=self.cfg.target_type,
                download=False,
            )
            self.ds_val = CelebA(
                root=self.cfg.root,
                split=self.cfg.val_split,
                target_type=self.cfg.target_type,
                download=False,
            )

        if stage in (None, "test"):
            self.ds_test = CelebA(
                root=self.cfg.root,
                split=self.cfg.test_split,
                target_type=self.cfg.target_type,
                download=False,
            )

    def _stack_targets(self, targets):
        """
        Robust stacking for CelebA targets.

        Cases:
        - target_type="attr": each target is Tensor[40]
        - target_type="identity": scalar-like
        - target_type=["attr", "identity"]: each target is tuple(...)
        """
        first = targets[0]

        if isinstance(first, tuple):
            # Multi-target case: stack each component independently
            out = []
            for i in range(len(first)):
                elems = [t[i] for t in targets]
                if torch.is_tensor(elems[0]):
                    out.append(torch.stack(elems, dim=0))
                else:
                    out.append(torch.tensor(elems))
            return tuple(out)

        if torch.is_tensor(first):
            return torch.stack(targets, dim=0)

        return torch.tensor(targets)

    def _collate(self, batch, train: bool):
        images_raw, targets_raw = zip(*batch)

        if train:
            views = [[] for _ in range(self.cfg.num_views)]
            for img in images_raw:
                img = img.convert("RGB")
                for v in range(self.cfg.num_views):
                    views[v].append(self.train_tfms(img))
            views = [torch.stack(v, dim=0) for v in views]
        else:
            images = [self.test_tfms(img.convert("RGB")) for img in images_raw]
            views = [torch.stack(images, dim=0)]

        targets = self._stack_targets(list(targets_raw))
        return views, targets

    def train_collate(self, batch):
        return self._collate(batch, train=True)

    def eval_collate(self, batch):
        return self._collate(batch, train=False)

    def train_dataloader(self):
        kwargs = dict(
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=(self.cfg.num_workers > 0),
            drop_last=True,
            collate_fn=self.train_collate,
        )
        if self.cfg.num_workers > 0:
            kwargs["prefetch_factor"] = 2
        return DataLoader(self.ds_train, **kwargs)

    def val_dataloader(self):
        kwargs = dict(
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=(self.cfg.num_workers > 0),
            drop_last=False,
            collate_fn=self.eval_collate,
        )
        if self.cfg.num_workers > 0:
            kwargs["prefetch_factor"] = 2
        return DataLoader(self.ds_val, **kwargs)

    def test_dataloader(self):
        kwargs = dict(
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=(self.cfg.num_workers > 0),
            drop_last=False,
            collate_fn=self.eval_collate,
        )
        if self.cfg.num_workers > 0:
            kwargs["prefetch_factor"] = 2
        return DataLoader(self.ds_test, **kwargs)

    def probe_train_dataloader(self):
        kwargs = dict(
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=(self.cfg.num_workers > 0),
            drop_last=False,
            collate_fn=self.eval_collate,
        )
        if self.cfg.num_workers > 0:
            kwargs["prefetch_factor"] = 2
        return DataLoader(self.ds_train, **kwargs)

    def probe_test_dataloader(self):
        kwargs = dict(
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=(self.cfg.num_workers > 0),
            drop_last=False,
            collate_fn=self.eval_collate,
        )
        if self.cfg.num_workers > 0:
            kwargs["prefetch_factor"] = 2
        return DataLoader(self.ds_test, **kwargs)