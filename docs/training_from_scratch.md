# Training SSL models from scratch

Our repository supports training the following SSL methods (from scratch) using [Lightly-SSL](https://github.com/lightly-ai/lightly-ssl):
- VICReg
- Barlow Twins

## Configuration Files

To train a model from scratch, you will need to create a configuration YAML file specifying the training parameters. You can find example configuration files in the `configs/` directory. An example configurationlooks like this:

```yaml
TODO
```

## Running the training script

To start training, you can run the following command:

```bash
python training/train.py \
--config-path <config-file-name>
```

## Tracking experiments

You can track your experiments using [Weights & Biases](https://wandb.ai/). Navigate to `configs/<config-file-name>.yaml` file and set `logging.backend` to `wandb`. We provide callbacks for logging Linear Probing accuracy, and CDNV metrics to Weights & Biases. 

For example, navigate to `configs/<config-file-name>.yaml` and set `probe.enabled` to `true` to log LP accuracy. You can find such toggles for each method in their respective configuration files. 
