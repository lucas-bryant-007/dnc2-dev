import pytest
import torch

from analysis.repair_legacy_ijepa_checkpoint import repair_checkpoint_dict


def _checkpoint(tokens=197, encoder="vit_base_patch16_224"):
    return {
        "hyper_parameters": {
            "model": {
                "encoder_type": encoder,
                "patch_size": 14,
                "image_size": 224,
            },
            "data": {"patch_size": 14, "img_size": 224},
        },
        "state_dict": {
            "backbone.vit.pos_embed": torch.zeros(1, tokens, 4),
            "teacher.vit.pos_embed": torch.zeros(1, tokens, 4),
        },
    }


def test_repairs_metadata_after_validating_position_embeddings():
    repaired = repair_checkpoint_dict(_checkpoint())

    assert repaired["hyper_parameters"]["model"]["patch_size"] == 16
    assert repaired["hyper_parameters"]["data"]["patch_size"] == 16
    assert repaired["legacy_metadata_repair"]["original"]["model.patch_size"] == 14
    assert repaired["legacy_metadata_repair"]["corrected"]["image_size"] == 224


def test_rejects_state_that_disagrees_with_encoder_geometry():
    with pytest.raises(ValueError, match="expected 197 tokens"):
        repair_checkpoint_dict(_checkpoint(tokens=257))


def test_rejects_unknown_encoder():
    with pytest.raises(ValueError, match="Unsupported or missing encoder_type"):
        repair_checkpoint_dict(_checkpoint(encoder="vit_unknown"))
