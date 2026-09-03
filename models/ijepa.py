import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from copy import deepcopy

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.ijepa_utils import IJEPAPredictorViT

import pytorch_lightning as pl
from lightly.models import utils
from lightly.models.modules import MaskedVisionTransformerTIMM

# timm vit builders
from timm.models.vision_transformer import vit_base_patch32_224, vit_base_patch16_224, vit_large_patch16_224

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
    
def build_vit(vit_name: str):
    vit_name = vit_name.lower()
    if vit_name == "vit_base_patch32_224":
        return vit_base_patch32_224()
    if vit_name == "vit_base_patch16_224":
        return vit_base_patch16_224()
    if vit_name == "vit_large_patch16_224":
        return vit_large_patch16_224()
    raise ValueError(f"Unknown vit_name={vit_name}. Add it to build_vit().")

class LightlyIJEPA(pl.LightningModule):
    """
    IJEPA implementation.
    Uses timm ViT backbones via lightly's MaskedVisionTransformerTIMM,
    and a custom Transformer predictor.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # Convert Namespace to dict for hyperparameter saving
        self.save_hyperparameters(_namespace_to_dict(cfg))

        # ── student backbone ──────────────────────────────────────────
        vit_student = build_vit(cfg.model.encoder_type)
        self.backbone = MaskedVisionTransformerTIMM(vit=vit_student)
        self.sequence_length = self.backbone.sequence_length  # includes CLS

        patch_size = vit_student.patch_embed.patch_size
        patch_size = patch_size[0] if isinstance(patch_size, tuple) else patch_size
        image_size = vit_student.patch_embed.img_size
        image_size = image_size[0] if isinstance(image_size, tuple) else image_size
        if int(cfg.model.patch_size) != patch_size:
            raise ValueError(
                f"Configured patch_size={cfg.model.patch_size} does not match "
                f"{cfg.model.encoder_type} patch size {patch_size}."
            )
        if int(cfg.data.img_size) != image_size:
            raise ValueError(
                f"Configured data.img_size={cfg.data.img_size} does not match "
                f"{cfg.model.encoder_type} image size {image_size}."
            )
        self.expected_image_size = image_size

        D = vit_student.embed_dim

        # ── predictor ─────────────────────────────────────────────────
        pred_depth = cfg.model.predictor_depth
        pred_heads = cfg.model.predictor_heads
        self.predictor = IJEPAPredictorViT(
            num_patches=self.sequence_length - 1,
            embed_dim=D,
            predictor_embed_dim=384,
            depth=pred_depth,
            num_heads=pred_heads,
        )

        # ── teacher backbone (EMA of student, no gradients) ──────────
        self.teacher = deepcopy(self.backbone)
        for p in self.teacher.parameters():
            p.requires_grad = False

        # ── EMA schedule parameters ──────────────────────────────────
        self.ema_momentum_start = getattr(cfg.model, "ema_momentum_start", 0.996)
        self.ema_momentum_end = getattr(cfg.model, "ema_momentum_end", 1.0)

        # ── mask token & misc ─────────────────────────────────────────
        self.mask_token = nn.Parameter(torch.zeros(1, 1, D))
        nn.init.normal_(self.mask_token, std=0.02)

        self.criterion = nn.MSELoss()

    def train(self, mode: bool = True):
        """Train the student/predictor while keeping the EMA teacher deterministic."""
        super().train(mode)
        self.teacher.eval()
        return self

    # ──────────────────────────────────────────────────────────────────
    #  EMA helpers
    # ──────────────────────────────────────────────────────────────────
    def _ema_momentum(self) -> float:
        """Cosine schedule from ema_momentum_start → ema_momentum_end."""
        if self.trainer is None or self.trainer.max_steps is None:
            return self.ema_momentum_end
        # use global_step for smooth per-step schedule
        max_steps = self.trainer.estimated_stepping_batches
        progress = min(self.global_step / max(1, max_steps), 1.0)
        m_start = self.ema_momentum_start
        m_end = self.ema_momentum_end
        return m_end - (m_end - m_start) * (1.0 + math.cos(math.pi * progress)) / 2.0

    @torch.no_grad()
    def _update_teacher(self):
        """Exponential moving average update of teacher from student."""
        momentum = self._ema_momentum()
        for teacher_p, student_p in zip(
            self.teacher.parameters(), self.backbone.parameters()
        ):
            teacher_p.data.mul_(momentum).add_(student_p.data, alpha=1.0 - momentum)
        self.log("train/ema_momentum", momentum, on_step=True, on_epoch=False)

    # ──────────────────────────────────────────────────────────────────
    #  Forward helpers
    # ──────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def extract_teacher_features(self, images, idx_mask):
        teacher_full = self.teacher.encode(
            images=images, idx_keep=None
        )  # [B, N+1, D]

        B, num_blocks, N_targ = idx_mask.shape
        idx_mask_flat = idx_mask.flatten(1)
        teacher_targets_flat = utils.get_at_index(
            teacher_full, idx_mask_flat
        )  # [B, 4*N_targ, D]
        teacher_targets = teacher_targets_flat.view(B, num_blocks, N_targ, -1)

        # ── normalize targets (as in the I-JEPA paper) ────────────────
        teacher_targets = F.layer_norm(teacher_targets, (teacher_targets.shape[-1],))

        return teacher_targets

    # ──────────────────────────────────────────────────────────────────
    #  Training
    # ──────────────────────────────────────────────────────────────────
    def training_step(self, batch, batch_idx):
        if self.global_step == 0 and self.global_rank == 0:
            self.print(
                f"world_size={self.trainer.world_size}, "
                f"per_gpu_bs={self.cfg.data.batch_size}"
            )

        views, _, idx_keep, idx_mask = batch
        images = views[0]  # [B, 3, H, W]

        if tuple(images.shape[-2:]) != (self.expected_image_size,) * 2:
            raise ValueError(
                f"Expected {self.expected_image_size}x{self.expected_image_size} images, "
                f"got {tuple(images.shape[-2:])}."
            )
        if idx_keep.numel() == 0 or idx_mask.numel() == 0:
            raise ValueError("I-JEPA masks need at least one context and target token")
        if idx_keep.min() <= 0 or idx_mask.min() <= 0:
            raise ValueError("CLS token at index 0 must not appear in I-JEPA masks")
        if max(int(idx_keep.max()), int(idx_mask.max())) >= self.sequence_length:
            raise ValueError("I-JEPA mask index exceeds the encoder token grid")

        # teacher: full image, no grad, normalized targets
        teacher_mask = self.extract_teacher_features(
            images=images, idx_mask=idx_mask
        )  # [B, 4, N_targ, D]

        # student: visible patches only
        student_vis = self.backbone.encode(
            images=images, idx_keep=idx_keep
        )  # [B, N_vis, D]

        # predictor: predict target representations from context
        student_pred_mask = self.predictor(
            student_vis, context_masks=idx_keep, target_masks=idx_mask
        )  # [B, 4, N_targ, D]

        if self.global_rank == 0 and self.global_step < 5:
            self.print(
                "teacher_mask mean/std:",
                teacher_mask.mean().item(),
                teacher_mask.std().item(),
            )
            self.print(
                "student_pred mean/std:",
                student_pred_mask.mean().item(),
                student_pred_mask.std().item(),
            )
            self.print(
                "student_vis shape:", student_vis.shape,
                "idx_keep shape:", idx_keep.shape,
                "idx_mask shape:", idx_mask.shape,
            )

        loss = self.criterion(student_pred_mask.float(), teacher_mask.float())
        self.log(
            "train/ijepa_loss", loss,
            prog_bar=True, on_step=True, on_epoch=True, sync_dist=True,
        )
        return loss

    def on_before_zero_grad(self, optimizer):
        """Update the EMA teacher once after each optimiser step."""
        self._update_teacher()

    # ──────────────────────────────────────────────────────────────────
    #  LR schedule & optimizer
    # ──────────────────────────────────────────────────────────────────
    def on_train_epoch_end(self):
        opt = self.optimizers()
        lr = opt.param_groups[0]["lr"]
        self.log(
            "train/lr", lr,
            on_step=False, on_epoch=True, prog_bar=False, sync_dist=True,
        )

    def configure_optimizers(self):
        scaled_lr = float(self.cfg.model.lr)
        wd = float(self.cfg.model.weight_decay)
        warmup_epochs = int(self.cfg.model.warmup_epochs)
        max_epochs = int(self.cfg.trainer.max_epochs)
        min_lr = float(self.cfg.model.min_lr)
        if scaled_lr <= 0 or min_lr < 0 or min_lr > scaled_lr:
            raise ValueError(
                f"Expected 0 <= min_lr <= lr, got min_lr={min_lr:g} "
                f"and lr={scaled_lr:g}."
            )

        # ── Collect parameters explicitly ─────────────────────────
        no_wd_keywords = {"pos_embed", "cls_token", "mask_token", "bias"}

        decay_params = []
        no_decay_params = []

        # Only optimize student backbone + predictor (NOT teacher)
        for name, param in self.backbone.named_parameters():
            if not param.requires_grad:
                continue
            if any(kw in name for kw in no_wd_keywords) or param.ndim <= 1:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        for name, param in self.predictor.named_parameters():
            if not param.requires_grad:
                continue
            if any(kw in name for kw in no_wd_keywords) or param.ndim <= 1:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        # mask_token is a top-level Parameter (1D-ish, no weight decay)
        no_decay_params.append(self.mask_token)

        param_groups = [
            {"params": decay_params, "weight_decay": wd},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(
            param_groups,
            lr=scaled_lr,
            betas=(0.9, 0.95),
        )

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / max(1, warmup_epochs)
            t = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
            cosine = 0.5 * (1.0 + math.cos(math.pi * t))
            return (min_lr / scaled_lr) + (1 - min_lr / scaled_lr) * cosine

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    # ──────────────────────────────────────────────────────────────────
    #  Checkpointing
    # ──────────────────────────────────────────────────────────────────
    def load_state_dict(self, state_dict, strict: bool = True):
        """Load exact new checkpoints; identify legacy student-only checkpoints."""
        if any(key.startswith("teacher.") for key in state_dict):
            return super().load_state_dict(state_dict, strict=strict)

        result = super().load_state_dict(state_dict, strict=False)
        non_teacher_missing = [
            key for key in result.missing_keys if not key.startswith("teacher.")
        ]
        if strict and (non_teacher_missing or result.unexpected_keys):
            raise RuntimeError(
                f"Invalid I-JEPA checkpoint: missing={non_teacher_missing}, "
                f"unexpected={result.unexpected_keys}"
            )
        self.teacher.load_state_dict(self.backbone.state_dict(), strict=True)
        warnings.warn(
            "Legacy I-JEPA checkpoint has no EMA teacher; reconstructed it from "
            "the student. New checkpoints preserve the exact EMA teacher.",
            UserWarning,
            stacklevel=2,
        )
        return result
