"""
Run this script as:
example:
    python training/train.py --config configs/vicreg/mini_imagenet.yaml
"""

import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_loader import load_config, dict_to_namespace, namespace_to_dict
from models.vicreg import LightlyVICReg
from models.ijepa import LightlyIJEPA
from models.wmse import LightlyWMSE
from data_utils.mini_imagenet_datamodule import MiniImageNetDataModule, MiniImageNetCfg
from data_utils.celebA_datamodule import CelebADataModule, CelebACfg
from utils.export_teacher import export_teacher_encoder_only
from utils.ckpt_schedule import ScheduledCheckpoint
from utils.linear_probe_callback import LinearProbeCallback
from utils.cdnv_callback import CDNVCallback

def main(cfg):
    print("\n========== CONFIG ==========")
    cfg_dict = namespace_to_dict(cfg)
    import json
    print(json.dumps(cfg_dict, indent=2, default=str))
    print("============================\n")

    # build data module
    if cfg.data.name.lower() == "mini_imagenet":
        data_cfg = MiniImageNetCfg(**namespace_to_dict(cfg.data))
        data_cfg.method = cfg.method.name
        data_module = MiniImageNetDataModule(data_cfg)
    elif cfg.data.name.lower() == "celeba":
        data_cfg = CelebACfg(**namespace_to_dict(cfg.data))
        data_cfg.method = cfg.method.name
        data_module = CelebADataModule(data_cfg)
    else:
        raise ValueError(f"Unknown dataset: {cfg.data.name}. Supported: 'mini_imagenet', 'celeba'")

    # build model based on method
    if cfg.method.name.lower() == "vicreg":
        model = LightlyVICReg(cfg)
    elif cfg.method.name.lower() == "ijepa":
        model = LightlyIJEPA(cfg)
    elif cfg.method.name.lower() == "wmse":
        model = LightlyWMSE(cfg)
    else:
        raise ValueError(f"Unknown method: {cfg.method.name}. Supported: 'vicreg', 'ijepa', 'wmse'")

    # custom model checkpointing & logging
    sched_cb = ScheduledCheckpoint(
        dirpath=cfg.ckpt_schedule.dirpath,
        early_every=cfg.ckpt_schedule.early_every,
        early_until=cfg.ckpt_schedule.early_until,
        late_every=cfg.ckpt_schedule.late_every,
        save_last=cfg.ckpt_schedule.save_last,
    )
    # create logger only on rank 0
    @rank_zero_only
    def create_logger(cfg):
        if cfg.logging.backend == "wandb":
            return WandbLogger(
                project=cfg.logging.project,
                # entity=cfg.logging.entity,
                name=cfg.logging.run_name,
                log_model=cfg.logging.log_model,
                tags=list(cfg.logging.tags)
            )
        return CSVLogger(save_dir=cfg.paths.exp_dir, name="logs")
    logger = create_logger(cfg)

    # linear probe callback
    probe_cb = LinearProbeCallback(**namespace_to_dict(cfg.probe))
    # CDNV callback
    cdnv_cb = CDNVCallback(**namespace_to_dict(cfg.cdnv))
    print("=== Callbacks configured ===")
    callbacks = [sched_cb, probe_cb, cdnv_cb]    
    for cb in callbacks:
        print(type(cb))
    print("============================")
    # trainer
    # Normalize strategy: if using plain 'ddp', enable find_unused_parameters to
    # avoid errors when some model parameters (e.g., frozen teacher) are unused.
    strategy = cfg.trainer.strategy if hasattr(cfg.trainer, 'strategy') else None
    if isinstance(strategy, str) and strategy == "ddp":
        strategy = "ddp_find_unused_parameters_true"

    trainer = pl.Trainer(
        default_root_dir=cfg.paths.exp_dir,
        devices=cfg.trainer.devices,
        accelerator=cfg.trainer.accelerator,
        strategy=strategy,
        max_epochs=cfg.trainer.max_epochs,
        use_distributed_sampler=cfg.trainer.get('use_distributed_sampler', False),
        log_every_n_steps=cfg.logging.log_every_n_steps,
        precision=cfg.precision,
        callbacks=callbacks,
        enable_checkpointing=False, # since using custom checkpoint callback
        logger=logger,
    )

    resume_ckpt = cfg.paths.get("resume_from_checkpoint", None)
    if resume_ckpt is not None and os.path.isfile(resume_ckpt):
        print(f"Resuming from checkpoint: {resume_ckpt}")
        trainer.fit(model, datamodule=data_module, ckpt_path=resume_ckpt)
    else:
        trainer.fit(model, datamodule=data_module)

    # export after training (only on global rank 0)
    if trainer.is_global_zero:
        export_path = f"{cfg.paths.exp_dir}/teacher_encoder_only.pt"
        export_teacher_encoder_only(model, export_path, extra_meta={"img_size": cfg.data.img_size})
        print(f"Exported teacher encoder to: {export_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train self-supervised models")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file (e.g., configs/vicreg_resnet50.yaml)"
    )
    args = parser.parse_args()

    # Load config from YAML file
    cfg = load_config(args.config)

    # Set up default paths if not already set
    if cfg.get("paths") is None:
        cfg["paths"] = {}

    output_root = cfg["paths"].get("output_root", "./checkpoints")
    exp_name = cfg.get("exp_name", "experiment")

    if cfg["paths"].get("exp_dir") is None:
        cfg["paths"]["exp_dir"] = os.path.join(output_root, exp_name)

    # Set up checkpoint dir if not set
    if cfg.get("ckpt_schedule") and cfg["ckpt_schedule"].get("dirpath") is None:
        cfg["ckpt_schedule"]["dirpath"] = os.path.join(cfg["paths"]["exp_dir"], "checkpoints")

    # Convert dict to namespace for easier access (cfg.data.x instead of cfg['data']['x'])
    cfg = dict_to_namespace(cfg)

    main(cfg)