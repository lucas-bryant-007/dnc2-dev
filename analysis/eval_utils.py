from pathlib import Path
import os, sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

import random
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
def extract_features(loader, backbone, device, 
                     both_views=False, max_batches=999999):
    """Extract features and labels from a dataloader."""
    feats_view1_list, feats_view2_list, y_list = [], [], []

    for batch_idx, batch in enumerate(tqdm(loader)):
        views, y, _, _ = batch
        if batch_idx >= max_batches:
            break
        images_view1 = (
            views[0].to(device, non_blocking=True)
            if isinstance(views, (list, tuple))
            else views.to(device)
        )
        images_view2 = None
        if both_views:
            if not isinstance(views, (list, tuple)) or len(views) < 2:
                raise ValueError("both_views=True requires two views in every batch")
            images_view2 = views[1].to(device, non_blocking=True)
            
        with torch.no_grad():
            feats_view1 = extract_backbone_features(backbone, images_view1)
            feats_view1 = F.normalize(feats_view1, dim=1)
            if images_view2 is not None:
                feats_view2 = extract_backbone_features(backbone, images_view2)
                feats_view2 = F.normalize(feats_view2, dim=1)
        feats_view1_list.append(feats_view1.cpu())
        if both_views and images_view2 is not None:
            feats_view2_list.append(feats_view2.cpu())

        y_list.append(y.cpu())

    if both_views and len(feats_view2_list) > 0:
        return torch.cat(feats_view1_list, dim=0), torch.cat(feats_view2_list, dim=0), torch.cat(y_list, dim=0)

    return torch.cat(feats_view1_list, dim=0), torch.cat(y_list, dim=0)

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
        from models.ijepa import LightlyIJEPA
        model = LightlyIJEPA(cfg)
    else:
        raise ValueError(f"Unsupported method {method_name} in checkpoint {ckpt_path}")

    if "state_dict" not in ckpt:
        raise RuntimeError(f"Checkpoint {ckpt_path} does not contain state_dict")
    model.load_state_dict(ckpt["state_dict"], strict=True)
    return model, cfg

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Ensures determinism
    random.seed(seed)

def get_subset_dataloader(
    data_source,
    class_indices,
    batch_size=128,
    shuffle=True,
    num_workers=8,
    drop_last=False,
    collate_fn=None,
    pin_memory=True,
):
    """
    Create a DataLoader for a subset of the dataset filtered by class labels.
    
    Example:
        >>> loader = get_subset_dataloader(original_loader, class_indices=[0, 1], 
        ...                                collate_fn=datamodule.eval_collate)
    """
    import numpy as np
    
    # Extract dataset from dataloader if needed
    if isinstance(data_source, torch.utils.data.DataLoader):
        dataset = data_source.dataset
    else:
        dataset = data_source
    
    class_array = np.array(class_indices)
    # Fast index lookup using vectorized operations
    try:
        # Try HuggingFace dataset approach (has 'label' column)
        all_labels = np.array(dataset['label'])
        subset_indices = np.where(np.isin(all_labels, class_array))[0].tolist()
    except (TypeError, KeyError):
        # Fallback to PyTorch dataset approach
        all_labels = []
        for i in range(len(dataset)):
            sample = dataset[i]
            label = sample['label'] if isinstance(sample, dict) else sample[1]
            all_labels.append(label)
        all_labels = np.array(all_labels)
        subset_indices = np.where(np.isin(all_labels, class_array))[0].tolist()
    
    # Create Subset and DataLoader
    # subset = Subset(dataset, subset_indices)
    label_map = {cls: i for i, cls in enumerate(class_indices)}
    subset = LabelMappedDataset(dataset, subset_indices, label_map)
    
    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    if collate_fn is not None:
        kwargs['collate_fn'] = collate_fn
    
    return torch.utils.data.DataLoader(subset, **kwargs)


class LabelMappedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices, label_map):
        self.dataset = dataset
        self.indices = indices
        self.label_map = label_map  # dict: original → new

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample = self.dataset[self.indices[idx]]

        if isinstance(sample, dict):
            label = sample.get('label', sample.get('labels'))
            sample = dict(sample)
            sample['label'] = self.label_map[int(label)]
            return sample

        elif isinstance(sample, (list, tuple)):
            *data, label = sample
            return (*data, self.label_map[int(label)])

        else:
            raise ValueError("Unsupported sample format")
