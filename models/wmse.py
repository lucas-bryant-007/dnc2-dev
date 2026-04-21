import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

import pytorch_lightning as pl
from lightly.models.modules.heads import VICRegProjectionHead, ProjectionHead
from torchvision.models import resnet18, resnet50
from timm.optim.lars import Lars #TODO: make this stable

# WMSELoss-specific imports
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from lightly.utils.dist import gather

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


def build_resnet(resnet_name: str = "resnet50"):
    resnet_name = resnet_name.lower()
    
    if resnet_name == "resnet18":
        model = resnet18()
        feature_dim = 512
    elif resnet_name == "resnet50":
        model = resnet50()
        feature_dim = 2048
    else:
        raise ValueError(f"Unknown resnet_name={resnet_name}. Supported: resnet18, resnet50")
    
    backbone = nn.Sequential(*list(model.children())[:-1])
    
    return backbone, feature_dim


class LightlyWMSE(pl.LightningModule):
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
        )
        
        self.projection_head = WMSEProjectionHead(
            input_dim=feature_dim,
            hidden_dim=cfg.model.hidden_dim,
            output_dim=cfg.model.output_dim,
            # num_layers=cfg.model.num_layers,
        )
        
        self.criterion = WhiteningMSELoss(
            slice_size=cfg.model.get('slice_size', 256),
            gather_distributed=cfg.model.get('gather_distributed', False),
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

        views, _ = batch
        images_0 = views[0] 
        images_1 = views[1]  

        z0 = self.forward(images_0)
        z1 = self.forward(images_1)

        loss = self.criterion(z0, z1)
        
        self.log("train/wmse_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss
    
    def on_train_epoch_end(self):
        opt = self.optimizers()
        lr = opt.param_groups[0]["lr"]
        self.log("train/lr", lr, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

    def configure_optimizers(self):
        scaled_lr = float(self.cfg.model.lr) * self.trainer.world_size * self.cfg.data.batch_size / 256.0
        
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
        min_lr = self.cfg.model.min_lr

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
    

class WMSEProjectionHead(ProjectionHead):
    """
    Projection head for W-MSE.

    Smaller and simpler than VICReg since whitening handles
    variance + covariance regularization.
    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 2048,
        output_dim: int = 128,
    ):
        super().__init__([
            (input_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()),
            (hidden_dim, output_dim, None, None),
        ])
    
class WhiteningMSELoss(torch.nn.Module):
    """
    Whitening MSE Loss (W-MSE) with batch slicing.

    Supports multi-view inputs and distributed training.

    Args:
        slice_size:
            Size of each sub-batch for whitening.
        gather_distributed:
            If True, gathers embeddings across GPUs before loss computation.
        eps:
            Numerical stability for covariance.
    """

    def __init__(
        self,
        slice_size: int = 256,
        gather_distributed: bool = False,
        eps: float = 1e-4,
    ):
        super().__init__()

        if gather_distributed and not dist.is_available():
            raise ValueError(
                "gather_distributed=True but torch.distributed not available."
            )

        self.slice_size = slice_size
        self.gather_distributed = gather_distributed
        self.eps = eps

    def forward(self, *zs: Tensor) -> Tensor:
        """
        Args:
            zs:
                List of tensors [z1, z2, ..., zK], each shape (B, D)

        Returns:
            Scalar W-MSE loss
        """

        assert len(zs) >= 2, "Need at least 2 views for W-MSE"
        B, D = zs[0].shape

        for z in zs:
            assert z.shape == (B, D), "All views must have same shape"

        # --------------------------------------------------
        # Gather across GPUs if needed
        # --------------------------------------------------
        if self.gather_distributed and dist.is_initialized():
            world_size = dist.get_world_size()
            if world_size > 1:
                zs = [torch.cat(gather(z), dim=0) for z in zs]
                B = zs[0].shape[0]

        # --------------------------------------------------
        # Stack views → [K, B, D]
        # --------------------------------------------------
        zs = torch.stack(zs, dim=0)
        K = zs.shape[0]

        # --------------------------------------------------
        # Same permutation across views
        # --------------------------------------------------
        perm = torch.randperm(B, device=zs.device)
        zs = zs[:, perm]

        # --------------------------------------------------
        # Batch slicing
        # --------------------------------------------------
        slice_size = self.slice_size
        num_slices = B // slice_size

        if num_slices == 0:
            raise ValueError(
                f"Batch size ({B}) must be >= slice_size ({slice_size})"
            )

        total_loss = 0.0
        count = 0

        for i in range(num_slices):
            start = i * slice_size
            end = start + slice_size

            # [K, slice_size, D]
            z_slice = zs[:, start:end]

            # --------------------------------------------------
            # Whitening (joint across views)
            # --------------------------------------------------
            z_flat = z_slice.reshape(-1, D)  # [K * slice_size, D]

            with torch.amp.autocast(device_type="cuda",enabled=False):
                z_whiten = whitening(z_flat)
            z_whiten = F.normalize(z_whiten, dim=1)

            # back to [K, slice_size, D]
            z_whiten = z_whiten.view(K, slice_size, D)

            # --------------------------------------------------
            # Pairwise MSE
            # --------------------------------------------------
            for a in range(K):
                for b in range(a + 1, K):
                    total_loss += F.mse_loss(z_whiten[a], z_whiten[b])
                    count += 1

        loss = total_loss / max(count, 1)
        return loss


# --------------------------------------------------
# Whitening helper (same as before)
# --------------------------------------------------
def whitening(x: Tensor, eps: float = 1e-4) -> Tensor:
    orig_dtype = x.dtype

    # ---- force float32 ----
    x = x.float()

    B, D = x.shape

    # center
    x = x - x.mean(dim=0, keepdim=True)

    # 🔥 reshape like official implementation
    T = x.T.contiguous()  # [D, B]

    # covariance
    cov = T @ T.T / (B - 1)

    # enforce symmetry
    cov = 0.5 * (cov + cov.T)
    # shrinkage for numerical stability
    eye = torch.eye(D, device=x.device, dtype=x.dtype)
    cov = (1 - eps) * cov + eps * eye

    L = torch.linalg.cholesky(cov)
    # inv_L = torch.inverse(L)
    inv_sqrt = torch.linalg.solve_triangular(L, eye, upper=False)

    # x_whiten = x @ inv_L.T
    x_whiten = x @ inv_sqrt.T
    # ---- cast back ----
    return x_whiten.to(orig_dtype)