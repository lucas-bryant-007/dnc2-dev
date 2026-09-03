import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

import pytorch_lightning as pl
from lightly.loss import VICRegLoss
from lightly.models.modules.heads import VICRegProjectionHead
from torchvision.models import resnet18, resnet50
from timm.optim.lars import Lars


def _namespace_to_dict(ns) -> dict:
    """
    Convert a Namespace object back to a dictionary recursively.
    Used for saving hyperparameters to checkpoints.
    """
    if isinstance(ns, dict):
        return {k: _namespace_to_dict(v) for k, v in ns.items()}
    elif hasattr(ns, '__dict__'):
        return {k: _namespace_to_dict(v) for k, v in ns.__dict__.items()}
    else:
        return ns


def build_resnet(resnet_name: str = "resnet50", pretrained: bool = False):
    resnet_name = resnet_name.lower()
    
    if resnet_name == "resnet18":
        model = resnet18(pretrained=pretrained)
        feature_dim = 512
    elif resnet_name == "resnet50":
        model = resnet50(pretrained=pretrained)
        feature_dim = 2048
    else:
        raise ValueError(f"Unknown resnet_name={resnet_name}. Supported: resnet18, resnet50")
    
    backbone = nn.Sequential(*list(model.children())[:-1])
    
    return backbone, feature_dim


class LightlyVICReg(pl.LightningModule):
    """
    Lightly VICReg implementation (ResNet backbone) compatible with our Trainer/DataModule.
    Expects batches shaped like:
        batch = (views, labels)
    where views is a list/tuple and views[0], views[1] are two augmented views of images [B,3,H,W].
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # Convert Namespace to dict for hyperparameter saving
        self.save_hyperparameters(_namespace_to_dict(cfg))

        self.backbone, feature_dim = build_resnet(
            cfg.model.resnet_name,
            pretrained=cfg.model.get('pretrained', False)
        )
        
        self.projection_head = VICRegProjectionHead(
            input_dim=feature_dim,
            hidden_dim=cfg.model.hidden_dim,
            output_dim=cfg.model.output_dim,
            num_layers=cfg.model.num_layers,
        )
        
        self.criterion = VICRegLoss(
            lambda_param=cfg.model.lambda_param,
            mu_param=cfg.model.mu_param,
            nu_param=cfg.model.nu_param,
            gather_distributed=(cfg.trainer.devices > 1),
        )

    def forward(self, x):
        """
        Forward pass through backbone and projection head.
        Args:
            x: input images [B, 3, H, W]
        Returns:
            z: projected embeddings [B, output_dim]
        """
        features = self.backbone(x)
        features = features.flatten(start_dim=1)

        z = self.projection_head(features)
        return z

    def training_step(self, batch, batch_idx):
        if self.global_step == 0 and self.global_rank == 0:
            self.print(f"world_size={self.trainer.world_size}, per_gpu_bs={self.cfg.data.batch_size}")

        views, _, _, _ = batch
        images_0 = views[0] 
        images_1 = views[1]  

        z0 = self.forward(images_0)
        z1 = self.forward(images_1)

        loss = self.criterion(z0, z1)
        
        self.log("train/vicreg_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss
    
    def on_train_epoch_end(self):
        opt = self.optimizers()
        lr = opt.param_groups[0]["lr"]
        self.log("train/lr", lr, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

    def configure_optimizers(self):
        effective_batch_size = (
            self.trainer.world_size
            * self.cfg.data.batch_size
            * self.cfg.trainer.accumulate_grad_batches
        )
        scaled_lr = float(self.cfg.model.lr) * effective_batch_size / 256.0
        
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=scaled_lr,
            weight_decay=float(self.cfg.model.weight_decay),
        )

        # optimizer = Lars(
        #     self.parameters(),
        #     lr=scaled_lr,
        #     weight_decay=float(self.cfg.model.weight_decay),
        # )

        # Warmup + cosine decay scheduler
        warmup_epochs = self.cfg.model.warmup_epochs
        max_epochs = self.cfg.trainer.max_epochs
        min_lr = float(self.cfg.model.min_lr)
        if scaled_lr <= 0 or min_lr < 0 or min_lr > scaled_lr:
            raise ValueError(
                f"Expected 0 <= min_lr <= scaled_lr, got min_lr={min_lr:g} "
                f"and scaled_lr={scaled_lr:g}."
            )

        def lr_lambda(epoch):
            # epoch is 0-indexed
            if epoch < warmup_epochs:
                # Linear warmup
                return (epoch + 1) / max(1, warmup_epochs)
            
            # Cosine decay after warmup
            t = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
            import math
            cosine = 0.5 * (1.0 + math.cos(math.pi * t))
            return (min_lr / scaled_lr) + (1 - min_lr / scaled_lr) * cosine
        
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            }
        }
