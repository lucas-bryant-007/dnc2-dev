import sys
import types

import pytest
import torch

from analysis.eval_utils import extract_features, freeze_model, load_model_from_checkpoint


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
