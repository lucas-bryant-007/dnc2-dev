import json
from pathlib import Path

import numpy as np
import torch

from analysis.plot_augmentation_survival import (
    load_survival_rows,
    render_survival_summary,
)
from analysis.plot_dsprites_controls import render_controls
from data_utils.dsprites_core import DSpritesCfg, build_groups
from models.supervised import SupervisedAttributeModel
from training.config_loader import dict_to_namespace, load_config


def _factorial_bits():
    return np.asarray(
        [[(value >> shift) & 1 for shift in range(3)] for value in range(8)],
        dtype=np.int64,
    )


def test_pair_factors_control_which_task_values_are_shared_between_views():
    bits = _factorial_bits()
    latents = np.zeros((bits.shape[0], 6), dtype=np.int64)
    latents[:, 2] = bits[:, 0]
    latents[:, 4] = bits[:, 1]
    latents[:, 5] = bits[:, 2]
    config = DSpritesCfg(
        task_factors=("scale", "posX", "posY"),
        pair_factors=("scale", "posX"),
        pair_mode="exact",
    )

    group_of, groups = build_groups(latents, bits, config)

    assert len(groups) == 4
    for members in groups:
        assert np.unique(bits[members, 0]).size == 1
        assert np.unique(bits[members, 1]).size == 1
        assert np.unique(bits[members, 2]).size == 2
    assert group_of.shape == (8,)


def test_checked_in_dsprites_config_accepts_environment_pair_factor_lists(monkeypatch):
    monkeypatch.setenv("DSPRITES_PAIR_FACTORS", "[posX, posY]")
    config = load_config("configs/vicreg/dsprites.yaml")

    assert config["data"]["pair_factors"] == ["posX", "posY"]


def test_checked_in_dsprites_config_accepts_matched_backbone_override(monkeypatch):
    monkeypatch.setenv("DSPRITES_RESNET", "resnet50")
    monkeypatch.setenv("DSPRITES_SEED", "17")

    config = load_config("configs/vicreg/dsprites.yaml")

    assert config["model"]["resnet_name"] == "resnet50"
    assert config["seed"] == 17


def test_single_task_supervised_control_targets_size_with_resnet18():
    config = dict_to_namespace(load_config("configs/supervised/dsprites_single_task.yaml"))

    model = SupervisedAttributeModel(config)

    assert model.target_name == "scale"
    assert model.target_index == 0
    assert model.classifier.in_features == 512


def test_single_task_supervised_control_uses_both_matched_views(monkeypatch):
    config = dict_to_namespace(load_config("configs/supervised/dsprites_single_task.yaml"))
    model = SupervisedAttributeModel(config)
    seen = []

    def fake_forward(images):
        seen.append(images.clone())
        return torch.stack((1.0 - images[:, 0], images[:, 0]), dim=1)

    monkeypatch.setattr(model, "forward", fake_forward)
    monkeypatch.setattr(model, "log", lambda *args, **kwargs: None)
    labels = torch.tensor([[0, 1, 0], [1, 0, 1]])
    first = torch.tensor([[0.0], [1.0]])
    second = torch.tensor([[0.0], [1.0]])

    loss = model.training_step(([first, second], labels, None, None), 0)

    assert torch.isfinite(loss)
    assert len(seen) == 2


def test_survival_summary_loader_uses_explicit_pair_factors(tmp_path):
    paths = []
    for epoch in (0, 10):
        path = tmp_path / f"result_{epoch}.json"
        path.write_text(
            json.dumps(
                {
                    "dataset": "dsprites",
                    "whitened": True,
                    "attributes": ["scale", "posX", "posY"],
                    "pair_factors": ["posX", "posY"],
                    "epoch": epoch,
                    "mean_abs_offdiag_cosine": 0.02,
                    "metrics": [
                        {"name": "scale", "capture_B": 0.1 + epoch / 100},
                        {"name": "posX", "capture_B": 0.8},
                        {"name": "posY", "capture_B": 0.9},
                    ],
                }
            ),
            encoding="utf-8",
        )
        paths.append(path)

    tasks, rows = load_survival_rows(paths)

    assert tasks == ["scale", "posX", "posY"]
    assert {row["condition"] for row in rows} == {"scale_varies"}
    assert len(rows) == 6


def test_survival_summary_aggregates_replicated_selective_effects(tmp_path):
    tasks = ["scale", "posX", "posY"]
    conditions = {
        "all": tasks,
        "scale": ["posX", "posY"],
        "posX": ["scale", "posY"],
        "posY": ["scale", "posX"],
    }
    paths = []
    for condition, pair_factors in conditions.items():
        for seed in (6, 17):
            for epoch in (0, 10):
                path = tmp_path / f"{condition}_{seed}_{epoch}.json"
                varied = next(
                    (task for task in tasks if task not in pair_factors), None
                )
                path.write_text(
                    json.dumps(
                        {
                            "dataset": "dsprites",
                            "whitened": True,
                            "attributes": tasks,
                            "pair_factors": pair_factors,
                            "epoch": epoch,
                            "training_seed": seed,
                            "mean_abs_offdiag_cosine": 0.02,
                            "metrics": [
                                {
                                    "name": task,
                                    "capture_B": (
                                        0.8
                                        + epoch / 100
                                        - (0.5 if task == varied else 0.05)
                                        + (seed - 6) / 1000
                                    ),
                                }
                                for task in tasks
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
                paths.append(path)
    output = tmp_path / "summary"

    summary = render_survival_summary(paths, output, final_epoch=10)

    assert summary["training_seeds"] == [6, 17]
    assert (output / "augmentation_survival_heatmap.pdf").is_file()
    assert (output / "augmentation_selectivity.pdf").is_file()
    assert (output / "augmentation_effects.csv").is_file()


def test_matched_control_plotter_separates_backbone_and_ssl_scale_spaces(
    tmp_path,
):
    tasks = ["scale", "posX", "posY"]

    def write_series(name, method, space, architecture, target, offset):
        paths = []
        for epoch in (0, 10):
            path = tmp_path / f"{name}_{epoch}.json"
            path.write_text(
                json.dumps(
                    {
                        "dataset": "dsprites",
                        "method": method,
                        "representation_space": space,
                        "architecture": architecture,
                        "supervised_target": target,
                        "attributes": tasks,
                        "epoch": epoch,
                        "training_seed": 6,
                        "feature_dim": 8,
                        "mean_abs_offdiag_cosine": 0.05 + offset,
                        "metrics": [
                            {
                                "name": task,
                                "capture_B": 0.2 + offset + index * 0.1 + epoch / 100,
                            }
                            for index, task in enumerate(tasks)
                        ],
                    }
                ),
                encoding="utf-8",
            )
            paths.append(path)
        return paths

    ssl_backbone = write_series(
        "ssl_backbone",
        "vicreg",
        "l2_normalized_backbone_rewhitened",
        "resnet18",
        None,
        0.0,
    )
    supervised = write_series(
        "supervised",
        "supervised",
        "l2_normalized_backbone_rewhitened",
        "resnet18",
        "scale",
        0.1,
    )
    ssl_r18 = write_series(
        "ssl_r18",
        "vicreg",
        "ssl_selected_subspace_rewhitened",
        "resnet18",
        None,
        0.2,
    )
    ssl_r50 = write_series(
        "ssl_r50",
        "vicreg",
        "ssl_selected_subspace_rewhitened",
        "resnet50",
        None,
        0.3,
    )
    output = tmp_path / "controls"

    summary = render_controls(
        ssl_backbone_json=ssl_backbone,
        supervised_json=supervised,
        ssl_r18_json=ssl_r18,
        ssl_r50_json=ssl_r50,
        output_dir=output,
    )

    assert summary["objective_control_epoch"] == 10
    assert summary["scale_control_epoch"] == 10
    assert (output / "single_task_supervised_control.pdf").is_file()
    assert (output / "single_task_supervised_dynamics.pdf").is_file()
    assert (output / "model_scale_control.pdf").is_file()


def test_s2_launchers_freeze_the_followup_and_factor_survival_designs():
    root = Path(__file__).resolve().parents[1]
    followup = (root / "analysis" / "run_compositional_followups_s2.sh").read_text(
        encoding="utf-8"
    )
    survival = (root / "analysis" / "run_augmentation_survival_s2.sh").read_text(
        encoding="utf-8"
    )

    assert "SENSITIVITY_SHOTS:-8 32 128" in followup
    assert "primary_transfer_metrics_refit=false" in followup
    assert "PREDICTIVE_NULL_PERMUTATIONS" in followup
    assert "verify_primary_reproduction" in followup
    assert "[scale, posX, posY]" in survival
    assert "[posX, posY]" in survival
    assert "[scale, posY]" in survival
    assert "[scale, posX]" in survival
    assert "only_intended_training_difference=pair_factors" in survival
    assert "SEEDS:-6 17 29" in survival
    assert "SUPERVISED_CONFIG" in survival
    assert "DSPRITES_RESNET=resnet50" in survival
    assert "plot_dsprites_controls.py" in survival
