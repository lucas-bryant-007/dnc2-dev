"""Shared pieces for the RO3 Push-T JEPA pipeline: frozen visual embeddings,
episode-level splits, and (state, candidate) row flattening."""
import os
import random
import warnings

import numpy as np
import torch
import torch.nn.functional as F

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# Goal-overlap progress is a spatial quantity, so we keep a coarse
# SPATIAL_GRID x SPATIAL_GRID layout of a patch/feature grid instead of a global
# average pool -- otherwise object position, and thus goal progress, is not
# linearly recoverable from the frozen embedding.
#
# Encoder is selectable via env var RO3_ENCODER:
#   "dinov2"     (default) DINOv2 ViT-S/14 patch tokens -- self-supervised ViT
#                whose patch features localize objects far better than an
#                ImageNet-classification CNN, so progress becomes recoverable.
#   "resnet18sp" frozen ImageNet ResNet-18 layer4 feature map (fallback).
# ENC_TAG keys the embedding cache so switching encoders never reuses a stale
# cache.
SPATIAL_GRID = 3
ENC_KIND = os.environ.get("RO3_ENCODER", "dinov2")
DINO_REF = os.environ.get(
    "RO3_DINOV2_REF", "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
)
ENC_TAG = f"{ENC_KIND}_{DINO_REF[:8]}_sp{SPATIAL_GRID}" if ENC_KIND == "dinov2" \
    else f"{ENC_KIND}_sp{SPATIAL_GRID}"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def _embed_resnet18sp(imgs_uint8, device, bs=256):
    """(N,H,W,3) uint8 -> (N, 512*S^2): ImageNet ResNet-18 layer4 map pooled to
    a SPATIAL_GRID x SPATIAL_GRID grid (keeps coarse position)."""
    from torchvision.models import resnet18, ResNet18_Weights
    net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    backbone = torch.nn.Sequential(*list(net.children())[:-2])  # up to layer4
    backbone.eval().to(device)
    mean, std = IMAGENET_MEAN.to(device), IMAGENET_STD.to(device)
    out = []
    for i in range(0, len(imgs_uint8), bs):
        x = torch.as_tensor(imgs_uint8[i:i + bs], device=device)
        x = x.permute(0, 3, 1, 2).float() / 255.0
        x = F.interpolate(x, size=224, mode="bilinear", align_corners=False)
        fmap = backbone((x - mean) / std)                       # (b,512,7,7)
        g = F.adaptive_avg_pool2d(fmap, SPATIAL_GRID)
        out.append(g.flatten(1).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


@torch.no_grad()
def _embed_dinov2(imgs_uint8, device, bs=128):
    """(N,H,W,3) uint8 -> (N, 384*S^2): DINOv2 ViT-S/14 patch tokens reshaped to
    their spatial grid and pooled to SPATIAL_GRID x SPATIAL_GRID. Patch tokens
    localize objects, so goal progress is linearly recoverable when preserved."""
    model = torch.hub.load(f"facebookresearch/dinov2:{DINO_REF}", "dinov2_vits14",
                           trust_repo=True)
    model.eval().to(device)
    mean, std = IMAGENET_MEAN.to(device), IMAGENET_STD.to(device)
    out = []
    for i in range(0, len(imgs_uint8), bs):
        x = torch.as_tensor(imgs_uint8[i:i + bs], device=device)
        x = x.permute(0, 3, 1, 2).float() / 255.0
        x = F.interpolate(x, size=224, mode="bilinear", align_corners=False)
        feats = model.forward_features((x - mean) / std)["x_norm_patchtokens"]
        b, npatch, cdim = feats.shape                           # (b,256,384)
        p = int(round(npatch ** 0.5))                           # 16
        grid = feats.transpose(1, 2).reshape(b, cdim, p, p)     # (b,384,16,16)
        g = F.adaptive_avg_pool2d(grid, SPATIAL_GRID)           # (b,384,S,S)
        out.append(g.flatten(1).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def _embed_images(imgs_uint8, device, bs=None):
    if ENC_KIND == "dinov2":
        return _embed_dinov2(imgs_uint8, device, bs or 128)
    if ENC_KIND == "resnet18sp":
        return _embed_resnet18sp(imgs_uint8, device, bs or 256)
    raise ValueError(f"unknown RO3_ENCODER={ENC_KIND!r}")


def frozen_embeddings(d, data_path, device):
    """Embed X_t and all six X_{t+H} once; cache next to the data file.
    Cache key includes ENC_TAG so changing the encoder invalidates old caches."""
    cache = os.path.splitext(data_path)[0] + f"_emb_{ENC_TAG}.npz"
    # Reuse the cache only if it is at least as new as the data file, so a
    # regenerated data file (same path, new contents) is always re-embedded.
    if os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(data_path):
        z = np.load(cache)
        return dict(e_t=z["e_t"], e_f=z["e_f"])
    n, c = d["x_f"].shape[:2]
    e_t = _embed_images(d["x_t"], device)
    e_f = _embed_images(d["x_f"].reshape(n * c, *d["x_f"].shape[2:]),
                        device).reshape(n, c, -1)
    np.savez(cache, e_t=e_t, e_f=e_f)
    print("cached embeddings ->", cache)
    return dict(e_t=e_t, e_f=e_f)


def _standardize(x, eps=1e-6):
    """Per-dim z-score; constant dims left unscaled (no divide-by-~0)."""
    mu = x.mean(0, keepdims=True)
    sd = x.std(0, keepdims=True)
    sd = np.where(sd < eps, 1.0, sd)
    return ((x - mu) / sd).astype(np.float32)


def flat_rows(d, emb):
    """One training row per (state, candidate): repeat e_t across candidates,
    flatten the H-step action sequence, normalize positions to [-1, 1].

    The returned image embedding is raw. Call ``standardize_rows`` with training
    indices before fitting a model so held-out rows do not affect preprocessing.
    """
    n, c = d["c_f"].shape
    acts = d["actions"].astype(np.float32) / 256.0 - 1.0        # (n,c,H,2)
    e_t = np.asarray(emb["e_t"], dtype=np.float32)              # (n, D)
    return dict(
        e_t=np.repeat(e_t, c, axis=0),                          # (n*c, D)
        act=acts.reshape(n * c, -1),                            # (n*c, H*2)
        e_f=emb["e_f"].reshape(n * c, -1),                      # (n*c, D)
        progress=d["progress"].reshape(n * c).astype(np.float32),
        state_id=np.repeat(np.arange(n), c),
        cand_id=np.tile(np.arange(c), n))


def standardize_rows(rows, fit_indices, mean=None, std=None, eps=1e-6):
    """Copy rows and z-score ``e_t`` using training rows or supplied stats."""
    result = dict(rows)
    e_t = np.asarray(rows["e_t"], dtype=np.float32)
    if mean is None or std is None:
        fit = e_t[np.asarray(fit_indices)]
        mean = fit.mean(0, keepdims=True)
        std = fit.std(0, keepdims=True)
    mean = np.asarray(mean, dtype=np.float32).reshape(1, -1)
    std = np.asarray(std, dtype=np.float32).reshape(1, -1)
    std = np.where(std < eps, 1.0, std)
    result["e_t"] = ((e_t - mean) / std).astype(np.float32)
    return result, {"input_mean": mean, "input_std": std}


def rows_for_checkpoint(rows, split, checkpoint):
    """Apply saved train-only preprocessing, with explicit legacy fallback."""
    if "input_mean" in checkpoint and "input_std" in checkpoint:
        return standardize_rows(
            rows,
            split["train"],
            mean=checkpoint["input_mean"],
            std=checkpoint["input_std"],
        )[0]
    warnings.warn(
        "Legacy Push-T checkpoint has no input normalization statistics; "
        "using the historical all-row normalization for compatibility.",
        UserWarning,
        stacklevel=2,
    )
    legacy = dict(rows)
    legacy["e_t"] = _standardize(rows["e_t"])
    return legacy


def episode_split(episode_ids, n_cand, seed=0, frac=(0.7, 0.15, 0.15)):
    """Split ROWS by demo episode so no initial state leaks across splits.
    Returns flat row indices (state-major, candidate-minor)."""
    eps = np.unique(episode_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(eps)
    n1 = int(frac[0] * len(eps)); n2 = n1 + int(frac[1] * len(eps))
    groups = dict(train=set(eps[:n1]), val=set(eps[n1:n2]),
                  test=set(eps[n2:]))
    out = {}
    for k, g in groups.items():
        states = np.flatnonzero(np.isin(episode_ids, list(g)))
        out[k] = (states[:, None] * n_cand + np.arange(n_cand)).ravel()
    out["test_states"] = np.flatnonzero(np.isin(episode_ids,
                                                list(groups["test"])))
    return out
