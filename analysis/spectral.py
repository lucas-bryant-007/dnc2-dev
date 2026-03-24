import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import random
from training.config_loader import load_config, dict_to_namespace, namespace_to_dict
from models.vicreg import LightlyVICReg
from models.ijepa import LightlyIJepa
from data_utils import MiniImageNetDataModule, MiniImageNetCfg

from eval_utils import find_checkpoint_files, load_model_from_checkpoint, extract_features

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Ensures determinism
    random.seed(seed)

def main(args):
    # Load config from YAML file
    cfg = load_config(args.config)
    # Convert dict to namespace for easier access (cfg.data.x instead of cfg['data']['x'])
    cfg = dict_to_namespace(cfg)

    # build data module
    data_cfg = MiniImageNetCfg(**namespace_to_dict(cfg.data))
    data_cfg.method = cfg.method.name
    data_module = MiniImageNetDataModule(data_cfg)
    data_module.setup()

    train_loader = data_module.probe_train_dataloader()
    test_loader = data_module.probe_test_dataloader()

    # build model 
    # get all checkpoint files in the directory
    ckpt_files = find_checkpoint_files(args.ckpt_dir)
    epoch, ckpt_path = ckpt_files[-1]  # get the latest checkpoint
    print(f"Loading model from checkpoint: {ckpt_path} (epoch {epoch})")
    model, cfg = load_model_from_checkpoint(ckpt_path)
    model = model.to('cuda')
    freeze_model(model)

    # get features and labels for both train and test loaders
    train_features, train_labels = extract_features(train_loader, model.backbone, device='cuda')
    test_features, test_labels = extract_features(test_loader, model.backbone, device='cuda')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Directory containing model checkpoints")

    args = parser.parse_args()
    main(args)