from dataclasses import dataclass
from typing import Optional
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datasets import load_dataset
from data_utils import get_transforms


@dataclass
class MiniImageNetCfg:
    name: str
    hf_repo: str
    hf_cache_dir: str
    method: str = "vicreg"
    img_size: int = 224
    batch_size: int = 128
    num_workers: int = 8
    train_split: str = "train"
    test_split: str = "test"
    num_views: int = 2


class MiniImageNetDataModule(pl.LightningDataModule):
    """
    HuggingFace Datasets-backed mini-ImageNet.

    - prepare_data(): downloads/caches dataset (runs on rank 0 only in Lightning)
    - setup(): creates split objects
    - dataloaders: return torch DataLoaders

    HF example items typically look like:
      {"image": PIL.Image, "label": int, ...}
    """

    def __init__(self, cfg: MiniImageNetCfg):
        super().__init__()
        self.cfg = cfg
        self.ds_train = None
        self.ds_test = None
        self.train_tfms, self.test_tfms = self._get_transforms(self.cfg.method, self.cfg.name)
        self.is_ddp = torch.distributed.is_available() and torch.distributed.is_initialized()

    def _get_transforms(self, method: str, dataset_name: str):
        """Method-aware transforms factory (mirrors data_utils/dataloaders.py)."""
        return get_transforms(
            method=method,
            dataset="cifar" if "cifar" in (dataset_name or "").lower() else dataset_name,
        )

    def prepare_data(self):
        # Download/cache the dataset. Lightning calls this only on rank 0.
        # We load both splits once so the cache is warm.
        load_dataset(self.cfg.hf_repo, split=self.cfg.train_split, cache_dir=self.cfg.hf_cache_dir)
        load_dataset(self.cfg.hf_repo, split=self.cfg.test_split, cache_dir=self.cfg.hf_cache_dir)

    def setup(self, stage: Optional[str] = None):
        self.ds_train = load_dataset(
            self.cfg.hf_repo,
            split=self.cfg.train_split,
            cache_dir=self.cfg.hf_cache_dir,
        )
        self.ds_test = load_dataset(
            self.cfg.hf_repo,
            split=self.cfg.test_split,
            cache_dir=self.cfg.hf_cache_dir,
        )

    def _collate(self, batch, train: bool):
        """
        Collate function that returns multiple views for SSL methods.
        """
        labels = [ex.get("label", -1) for ex in batch]
        
        if train:
            # Create 2 augmented views of each image
            views = [[] for _ in range(self.cfg.num_views)]
            for ex in batch:
                img = ex["image"].convert("RGB")
                for v in range(self.cfg.num_views):
                    views[v].append(self.train_tfms(img))
            views = [torch.stack(v, dim=0) for v in views]
            return views, torch.tensor(labels, dtype=torch.long)
        else:
            # Single view (for validation)
            images = [self.test_tfms(ex["image"].convert("RGB")) for ex in batch]
            return [torch.stack(images, dim=0)], torch.tensor(labels, dtype=torch.long)

    def train_collate(self, batch):
        """Collate function for training loader (with augmentations)."""
        return self._collate(batch, train=True)

    def eval_collate(self, batch):
        """Collate function for eval/probe loaders (no augmentations)."""
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
            shuffle=not self.is_ddp,  # shuffle val only if not using DDP (Lightning handles shuffling in DDP)
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=(self.cfg.num_workers > 0),
            drop_last=False,
            collate_fn=self.eval_collate,
        )
        if self.cfg.num_workers > 0:
            kwargs["prefetch_factor"] = 2

        return DataLoader(self.ds_test, **kwargs)

    def test_dataloader(self):
        return DataLoader(
            self.ds_test,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=(self.cfg.num_workers > 0),
            collate_fn=self.eval_collate,
        )

    
    def probe_train_dataloader(self):
        return DataLoader(
            self.ds_train,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=(self.cfg.num_workers > 0),
            collate_fn=self.eval_collate,
        )
    
    def probe_test_dataloader(self):
        return DataLoader(
            self.ds_test,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=(self.cfg.num_workers > 0),
            collate_fn=self.eval_collate,
        )