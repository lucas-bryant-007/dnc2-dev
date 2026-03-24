from pathlib import Path
import os, sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn.functional as F
from tqdm import tqdm

def find_checkpoint_files(ckpt_dir, start=0, end=1000):
    """Find all checkpoint files in the given directory within the epoch range."""
    files = []
    for epoch in range(start, end + 1):
        fname = os.path.join(ckpt_dir, f"epoch_{epoch:04d}.ckpt")
        if os.path.exists(fname):
            files.append((epoch, fname))
    # Also include last.ckpt if present
    lastp = os.path.join(ckpt_dir, 'last.ckpt')
    if os.path.exists(lastp):
        files.append(('last', lastp))
    # Sort: numeric epochs first, then 'last'
    files.sort(key=lambda x: (x[0] == 'last', x[0] if isinstance(x[0], int) else 9999))
    return files

@torch.no_grad()
def extract_backbone_features(backbone, images):
    """
    Extract features from backbone, handling both ViT and ResNet architectures.
    Returns features [B, D].
    """
    out = None
    
    # Handle models with nested `vit` attribute (some wrappers)
    if hasattr(backbone, 'vit'):
        vit = backbone.vit
        out = vit.forward_features(images)
    # Try common hooks used by various encoders
    elif hasattr(backbone, 'forward_features'):
        out = backbone.forward_features(images)
    else:
        # Direct call
        out = backbone(images)

    # Unwrap outputs
    feats = None
    if hasattr(out, 'last_hidden_state') or hasattr(out, 'pooler_output'):
        hidden = getattr(out, 'last_hidden_state', None)
        if hidden is not None:
            feats = hidden[:, 0] if hidden.dim() == 3 else hidden
        else:
            pooled = getattr(out, 'pooler_output', None)
            if pooled is not None:
                feats = pooled
    elif isinstance(out, dict):
        if 'last_hidden_state' in out:
            hidden = out['last_hidden_state']
            feats = hidden[:, 0] if hidden.dim() == 3 else hidden
        elif 'pooler_output' in out:
            feats = out['pooler_output']
    elif isinstance(out, torch.Tensor):
        if out.dim() == 3:
            feats = out[:, 0]
        elif out.dim() > 2:
            feats = torch.flatten(out, 1)
        else:
            feats = out
    if feats is None:
        raise RuntimeError("Could not extract features from backbone output")

    return feats

@torch.no_grad()
def extract_features(loader, backbone, device, max_batches=999999):
    """Extract features and labels from a dataloader."""
    feats_list, y_list = [], []

    for batch_idx, (views, y) in enumerate(tqdm(loader)):
        if batch_idx >= max_batches:
            break
        images = (
            views[0].to(device, non_blocking=True)
            if isinstance(views, (list, tuple))
            else views.to(device)
        )

        with torch.no_grad():
            feats = extract_backbone_features(backbone, images)
            feats = F.normalize(feats, dim=1)

        feats_list.append(feats.cpu())
        y_list.append(y.cpu())

    return torch.cat(feats_list, dim=0), torch.cat(y_list, dim=0)

def load_model_from_checkpoint(ckpt_path, device='cpu'):
    """
    Load a model from a PyTorch Lightning checkpoint.
    Uses the hyperparameters stored in the checkpoint itself.
    """
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if 'hyper_parameters' not in ckpt:
        raise RuntimeError(f"Checkpoint {ckpt_path} does not contain hyper_parameters")

    hparams = ckpt['hyper_parameters']
    # Convert dict to Namespace for model compatibility
    from training.config_loader import dict_to_namespace
    cfg = dict_to_namespace(hparams)

    method_name = getattr(getattr(cfg, 'method', {}), 'name', '').lower() if hasattr(cfg, 'method') else ''

    if method_name == 'vicreg':
        from models.vicreg import LightlyVICReg
        model = LightlyVICReg(cfg)
    elif method_name == 'ijepa':
        from models.ijepa import LightlyIJepa
        model = LightlyIJepa(cfg)
    else:
        raise ValueError(f"Unsupported method {method_name} in checkpoint {ckpt_path}")

    epoch = ckpt.get("epoch", None)
    if epoch != 0:
        model.load_state_dict(ckpt["state_dict"], strict=False)
    else:
        print("⚠️ Skipping weight load for epoch 0 checkpoint")
    return model, cfg