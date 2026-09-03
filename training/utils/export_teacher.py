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

    teacher = getattr(lightly_mae_module, "teacher", None)
    if teacher is None:
        raise ValueError("teacher export requires a model with an EMA teacher")
    state = {f"teacher.{key}": value for key, value in teacher.state_dict().items()}
    cfg = lightly_mae_module.cfg
    meta = {
        "type": "ijepa_ema_teacher_encoder_only",
        "encoder_type": str(cfg.model.encoder_type),
        "patch_size": int(cfg.model.patch_size),
        "image_size": int(cfg.data.img_size),
        "sequence_length": int(lightly_mae_module.sequence_length),
    }
    if extra_meta:
        meta.update(extra_meta)

    payload = {
        "state_dict": state,
        "hyper_parameters": {
            "method": {"name": "ijepa"},
            "model": {"encoder_type": str(cfg.model.encoder_type)},
        },
        "meta": meta,
    }
    torch.save(payload, out_path)
