import os
import yaml
import click
from pathlib import Path

import torch
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import warnings
warnings.filterwarnings('ignore')

from LightningModules.Models.gravnet import GravNet

from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.overrides import LightningDistributedModule


class CustomDDPPlugin(DDPPlugin):
    def configure_ddp(self):
        self.pre_configure_ddp()
        self._model = self._setup_model(LightningDistributedModule(self.model))
        self._register_ddp_hooks()
        self._model._set_static_graph()


@click.command()
@click.argument('config', type=str, required=True)

def main(config):
    with open(config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    evaluate(config)
    
def evaluate(config):

    try:
        checkpoint_path = max((str(path) for path in Path(config["artifacts"]).rglob("best*.ckpt")), key=os.path.getctime)
    except:
        raise ValueError(f"No checkpoint saved for {config['artifacts']}")

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model_name = checkpoint["hyper_parameters"]["model"]
    if model_name in globals():
        model = globals()[model_name].load_from_checkpoint(checkpoint_path)
    else:
        raise ValueError(f"Model name {model_name} not found in globals")

    accelerator = "gpu" if torch.cuda.is_available() else None
    trainer = Trainer(
        gpus=config["gpus"],
        accelerator=accelerator,
    )
    trainer.test(model)


if __name__ == "__main__":
    main()
