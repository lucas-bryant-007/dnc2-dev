"""
Run this script as:
python train.py --config training/configs/vicreg_resnet50.yaml
"""

import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
import wandb
import faulthandler, signal

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_loader import load_config, dict_to_namespace, namespace_to_dict
from models.vicreg import LightlyVICReg
from models.ijepa import LightlyIJepa
from data.mini_imagenet_datamodule import MiniImageNetDataModule, MiniImageNetCfg
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
    data_cfg = MiniImageNetCfg(**namespace_to_dict(cfg.data))
    data_cfg.method = cfg.method.name
    data_module = MiniImageNetDataModule(data_cfg)

    # build model based on method
    if cfg.method.name.lower() == "vicreg":
        model = LightlyVICReg(cfg)
    elif cfg.method.name.lower() == "ijepa":
        model = LightlyIJepa(cfg)
    else:
        raise ValueError(f"Unknown method: {cfg.method.name}. Supported: 'vicreg', 'ijepa'")

    # custom model checkpointing & logging
    sched_cb = ScheduledCheckpoint(
        dirpath=cfg.ckpt_schedule.dirpath,
        early_every=cfg.ckpt_schedule.early_every,
        early_until=cfg.ckpt_schedule.early_until,
        late_every=cfg.ckpt_schedule.late_every,
        save_last=cfg.ckpt_schedule.save_last,
    )
    # Instantiate logger only on rank 0 to avoid hangs when running under DDP.
    # Check several common environment variables that indicate rank.
    def _is_rank0():
        for k in ("PL_GLOBAL_RANK", "GLOBAL_RANK", "RANK", "LOCAL_RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK"):
            v = os.environ.get(k)
            if v is not None:
                try:
                    return int(v) == 0
                except Exception:
                    continue
        # If none are set, assume single-process (rank 0)
        return True

    if _is_rank0():
        if cfg.logging.backend == "wandb":
            logger = WandbLogger(
                project=cfg.logging.project,
                # entity=cfg.logging.entity,
                name=cfg.logging.run_name,
                log_model=cfg.logging.log_model,
                tags=list(cfg.logging.tags)
            )
        else:
            logger = CSVLogger(save_dir=cfg.paths.exp_dir, name="logs")
    else:
        logger = None

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

    # Lightning automatically calls setup() in each distributed process                                                                                                                                                         
    # Do NOT call data_module.setup() manually before trainer.fit() in DDP
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