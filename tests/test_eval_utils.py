import sys
import types

import pytest
import torch

from analysis.eval_utils import (
    extract_backbone_features,
    extract_features,
    freeze_model,
    load_model_from_checkpoint,
)


class IdentityBackbone(torch.nn.Module):
    def forward(self, images):
        return images


def test_freeze_model_enters_eval_mode_and_disables_gradients():
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 4),
        torch.nn.BatchNorm1d(4),
    )
    model.train()

    returned = freeze_model(model)

    assert returned is model
    assert not model.training
    assert not model[1].training
    assert all(not parameter.requires_grad for parameter in model.parameters())


class SequenceBackbone(torch.nn.Module):
    """Stands in for a ViT: returns [B, tokens, D] with a distinctive token 0."""

    def forward_features(self, images):
        batch = images.shape[0]
        tokens = torch.arange(1.0, 5.0).view(1, 4, 1).repeat(batch, 1, 3)
        tokens[:, 0] = -99.0  # CLS slot, deliberately unlike the patch tokens
        return tokens


def test_vit_pooling_cls_takes_token_zero():
    feats = extract_backbone_features(SequenceBackbone(), torch.zeros(2, 3, 8, 8),
                                      vit_pooling="cls")
    assert feats.shape == (2, 3)
    assert torch.allclose(feats, torch.full((2, 3), -99.0))


def test_vit_pooling_mean_excludes_the_cls_slot():
    """The I-JEPA fix: average patch tokens, never token 0.

    That token gets no gradient in this implementation, so including it would
    contaminate the representation with an untrained vector.
    """
    feats = extract_backbone_features(SequenceBackbone(), torch.zeros(2, 3, 8, 8),
                                      vit_pooling="mean")
    assert feats.shape == (2, 3)
    assert torch.allclose(feats, torch.full((2, 3), 3.0))  # mean of 2,3,4


def test_vit_pooling_rejects_unknown_choice():
    with pytest.raises(ValueError, match="vit_pooling"):
        extract_backbone_features(SequenceBackbone(), torch.zeros(1, 3, 8, 8),
                                  vit_pooling="average")


def test_pooling_does_not_disturb_convolutional_backbones():
    class ConvBackbone(torch.nn.Module):
        def forward(self, images):
            return torch.ones(images.shape[0], 5, 1, 1)

    reference = extract_backbone_features(ConvBackbone(), torch.zeros(2, 3, 8, 8))
    for pooling in ("cls", "mean"):
        feats = extract_backbone_features(ConvBackbone(), torch.zeros(2, 3, 8, 8),
                                          vit_pooling=pooling)
        assert feats.shape == (2, 5)
        assert torch.allclose(feats, reference)


def _batch(views):
    return views, torch.tensor([0, 1]), None, None


def test_both_view_extraction_rejects_a_missing_later_view():
    loader = [
        _batch([torch.eye(2), torch.eye(2)]),
        _batch([torch.eye(2)]),
    ]
    with pytest.raises(ValueError, match="two views in every batch"):
        extract_features(loader, IdentityBackbone(), "cpu", both_views=True)


def test_epoch_zero_checkpoint_loads_state_dict_strictly(tmp_path, monkeypatch):
    calls = []

    class DummyModel:
        def __init__(self, cfg):
            self.cfg = cfg

        def load_state_dict(self, state_dict, strict):
            calls.append((state_dict, strict))

    module = types.ModuleType("models.vicreg")
    module.LightlyVICReg = DummyModel
    monkeypatch.setitem(sys.modules, "models.vicreg", module)
    checkpoint = tmp_path / "epoch0.ckpt"
    torch.save(
        {
            "epoch": 0,
            "hyper_parameters": {"method": {"name": "vicreg"}},
            "state_dict": {"backbone.weight": torch.tensor([7.0])},
        },
        checkpoint,
    )
    load_model_from_checkpoint(checkpoint)
    assert calls and calls[0][1] is True
    assert calls[0][0]["backbone.weight"].item() == 7.0
