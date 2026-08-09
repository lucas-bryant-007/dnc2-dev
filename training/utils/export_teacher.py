import os
import torch
from typing import Dict, Any


def export_teacher_encoder_only(
    lightly_mae_module,
    out_path: str,
    extra_meta: Dict[str, Any] | None = None,
):
    """
    Saves only the teacher encoder/backbone weights + minimal metadata.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # In our LightlyMAE module:
    # lightly_mae_module.backbone is MaskedVisionTransformerTIMM
    # We want the underlying ViT weights for use in Stage 2.
    state = lightly_mae_module.backbone.state_dict()

    meta = {"type": "teacher_encoder_only"}
    # I-JEPA/MAE carry these; ResNet methods (VICReg/WMSE) don't, so pull them
    # defensively instead of crashing the post-training export.
    stage = getattr(getattr(lightly_mae_module, "cfg", None), "stage", None)
    if stage is not None:
        if hasattr(stage, "vit_name"):
            meta["vit_name"] = stage.vit_name
        if hasattr(stage, "mask_ratio"):
            meta["mask_ratio_stage1"] = float(stage.mask_ratio)
    if hasattr(lightly_mae_module, "patch_size"):
        meta["patch_size"] = int(lightly_mae_module.patch_size)
    if hasattr(lightly_mae_module, "sequence_length"):
        meta["sequence_length"] = int(lightly_mae_module.sequence_length)
    if extra_meta:
        meta.update(extra_meta)

    payload = {"state_dict": state, "meta": meta}
    torch.save(payload, out_path)
