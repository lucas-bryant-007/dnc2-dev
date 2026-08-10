import argparse
import json

import pytest
import torch

from analysis.repair_legacy_ijepa_checkpoint import main, repair_checkpoint_dict


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


def test_writes_hashed_repair_sidecar_and_can_revalidate_it(tmp_path):
    source = tmp_path / "legacy.ckpt"
    repaired = tmp_path / "repaired.ckpt"
    record = tmp_path / "repair.json"
    torch.save(_checkpoint(), source)

    args = argparse.Namespace(
        input=str(source),
        output=str(repaired),
        force=False,
        record=str(record),
        record_only=False,
    )
    main(args)
    payload = json.loads(record.read_text(encoding="utf-8"))

    assert len(payload["source"]["sha256"]) == 64
    assert len(payload["repaired_copy"]["sha256"]) == 64
    assert payload["repair"]["corrected"] == {"patch_size": 16, "image_size": 224}

    args.record_only = True
    main(args)
