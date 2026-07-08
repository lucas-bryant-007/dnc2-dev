"""Shared pieces for the RO3 Push-T JEPA pipeline: frozen visual embeddings,
episode-level splits, and (state, candidate) row flattening."""
import os
import random

import numpy as np
import torch

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def _embed_images(imgs_uint8, device, bs=256):
    """(N,H,W,3) uint8 -> (N,512) float32 from a frozen ImageNet ResNet-18."""
    from torchvision.models import resnet18, ResNet18_Weights
    net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    net.fc = torch.nn.Identity()
    net.eval().to(device)
    mean, std = IMAGENET_MEAN.to(device), IMAGENET_STD.to(device)
    out = []
    for i in range(0, len(imgs_uint8), bs):
        x = torch.as_tensor(imgs_uint8[i:i + bs], device=device)
        x = x.permute(0, 3, 1, 2).float() / 255.0
        x = torch.nn.functional.interpolate(x, size=224, mode="bilinear",
                                            align_corners=False)
        out.append(net((x - mean) / std).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def frozen_embeddings(d, data_path, device):
    """Embed X_t and all six X_{t+H} once; cache next to the data file."""
    cache = os.path.splitext(data_path)[0] + "_emb.npz"
    if os.path.exists(cache):
        z = np.load(cache)
        return dict(e_t=z["e_t"], e_f=z["e_f"])
    n, c = d["x_f"].shape[:2]
    e_t = _embed_images(d["x_t"], device)
    e_f = _embed_images(d["x_f"].reshape(n * c, *d["x_f"].shape[2:]),
                        device).reshape(n, c, -1)
    np.savez(cache, e_t=e_t, e_f=e_f)
    print("cached embeddings ->", cache)
    return dict(e_t=e_t, e_f=e_f)


def flat_rows(d, emb):
    """One training row per (state, candidate): repeat e_t across candidates,
    flatten the H-step action sequence, normalize positions to [-1, 1]."""
    n, c = d["c_f"].shape
    acts = d["actions"].astype(np.float32) / 256.0 - 1.0        # (n,c,H,2)
    return dict(
        e_t=np.repeat(emb["e_t"], c, axis=0),                   # (n*c, 512)
        act=acts.reshape(n * c, -1),                            # (n*c, H*2)
        e_f=emb["e_f"].reshape(n * c, -1),                      # (n*c, 512)
        progress=d["progress"].reshape(n * c).astype(np.float32),
        state_id=np.repeat(np.arange(n), c),
        cand_id=np.tile(np.arange(c), n))


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
