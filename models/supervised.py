"""Single-task supervised control with the same ResNet backbone as VICReg."""

from __future__ import annotations

import math

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from models.vicreg import _namespace_to_dict, build_resnet


class SupervisedAttributeModel(pl.LightningModule):
    """Train one binary task while retaining the penultimate representation."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(_namespace_to_dict(cfg))
        self.backbone, feature_dim = build_resnet(
            cfg.model.resnet_name,
            pretrained=cfg.model.get("pretrained", False),
        )
        task_factors = list(cfg.data.task_factors)
        self.target_name = str(cfg.model.target_name)
        if self.target_name not in task_factors:
            raise ValueError(
                f"supervised target {self.target_name!r} is not in {task_factors}"
            )
        pair_factors = (
            task_factors
            if cfg.data.get("pair_factors") is None
            else list(cfg.data.pair_factors)
        )
        if self.target_name not in pair_factors:
            raise ValueError(
                f"supervised target {self.target_name!r} must be fixed across "
                f"paired views; pair_factors={pair_factors}"
            )
        self.target_index = task_factors.index(self.target_name)
        self.classifier = torch.nn.Linear(feature_dim, 2)

    def _features(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images).flatten(start_dim=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self._features(images))

    def training_step(self, batch, batch_idx):
        views, labels, _, _ = batch
        target = labels[:, self.target_index].long()
        if not isinstance(views, (list, tuple)) or len(views) != 2:
            raise ValueError("the matched supervised control requires exactly two views")
        logits = torch.cat((self(views[0]), self(views[1])), dim=0)
        repeated_target = target.repeat(2)
        loss = F.cross_entropy(logits, repeated_target)
        accuracy = (logits.argmax(dim=1) == repeated_target).float().mean()
        self.log(
            "train/supervised_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "train/supervised_accuracy",
            accuracy,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def configure_optimizers(self):
        scaled_lr = (
            float(self.cfg.model.lr)
            * self.trainer.world_size
            * self.cfg.data.batch_size
            / 256.0
        )
        minimum_lr = float(self.cfg.model.min_lr)
        if scaled_lr <= 0 or minimum_lr < 0 or minimum_lr > scaled_lr:
            raise ValueError(
                f"Expected 0 <= min_lr <= scaled_lr, got {minimum_lr:g} and "
                f"{scaled_lr:g}"
            )
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=scaled_lr,
            weight_decay=float(self.cfg.model.weight_decay),
        )
        warmup_epochs = int(self.cfg.model.warmup_epochs)
        maximum_epochs = int(self.cfg.trainer.max_epochs)

        def multiplier(epoch: int) -> float:
            if epoch < warmup_epochs:
                return (epoch + 1) / max(1, warmup_epochs)
            progress = (epoch - warmup_epochs) / max(1, maximum_epochs - warmup_epochs)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            ratio = minimum_lr / scaled_lr
            return ratio + (1.0 - ratio) * cosine

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": LambdaLR(optimizer, multiplier),
                "interval": "epoch",
                "frequency": 1,
            },
        }
