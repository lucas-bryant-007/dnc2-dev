from pathlib import Path

from training.config_loader import _substitute_env_vars, load_config


def test_full_environment_expressions_preserve_yaml_types(monkeypatch):
    monkeypatch.delenv("DNC2_MISSING", raising=False)
    assert _substitute_env_vars("${oc.env:DNC2_MISSING,null}") is None
    assert _substitute_env_vars("${oc.env:DNC2_MISSING,true}") is True
    assert _substitute_env_vars("${oc.env:DNC2_MISSING,12}") == 12

    monkeypatch.setenv("DNC2_NUMBER", "3.5")
    assert _substitute_env_vars("${oc.env:DNC2_NUMBER,0}") == 3.5


def test_embedded_environment_expression_remains_text(monkeypatch):
    monkeypatch.setenv("DNC2_NAME", "run-seven")
    assert _substitute_env_vars("outputs/${oc.env:DNC2_NAME,default}") == "outputs/run-seven"


def test_checked_in_ijepa_config_has_consistent_grid_and_lr():
    config = load_config(Path("configs/ijepa/celeba.yaml"))
    assert config["data"]["img_size"] == 224
    assert config["data"]["patch_size"] == 16
    assert config["model"]["patch_size"] == 16
    assert 0 <= config["model"]["min_lr"] <= config["model"]["lr"]
    assert config["paths"]["exp_dir"] is None
