import os
import sys
import torch
import wandb
import argparse
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

main_path = "/scratch/gpfs/tt1131/projects/dp_tutorial/"
sys.path.append(main_path)

from base.utils import *
from base.data_module_re import *
from base.deep_phase_re import *

torch.set_float32_matmul_precision("medium")

parser = argparse.ArgumentParser()
parser.add_argument("--proj_name", type=str)
args = parser.parse_args()
proj_name = args.proj_name

## Set random seed
seed = 42
seed_everything(seed, workers=True)
rs = np.random.RandomState(seed)

## Dataset Parameters
intvl_size = 1
fs = 100
cutoff = 1
order = 3
num_workers = 10
test_size = 0.2

epochs = 100  ## 102823: Longer epochs because the loss did not seem to converge
num_joints = 62
in_channels = num_joints
out_channels = num_joints // 2

## Model Parameters
lr_monitor = LearningRateMonitor(logging_interval="epoch")
window_size = 39
lr = 1e-4
batch_size = 32
latent_channels = 10

joints_seg = np.load(
    os.path.join(main_path, "data", "joints_102723.npy"), allow_pickle=True
)
states = np.load(
    os.path.join(main_path, "data", "states_102723.npy"), allow_pickle=True
)
lc_hp_joints = butter_pass("hp", joints_seg, cutoff=cutoff, fs=fs, order=order, axis=0)
wandb_path = os.path.join(main_path, "expts_wandb/")
dm = dp_dm(
    lc_hp_joints,
    states,
    batch_size,
    window_size,
    intvl_size,
    num_workers,
    test_size,
    rs,
)
dm.setup()
expt_name = f"expt-{latent_channels}"

wandb_logger = WandbLogger(project=proj_name, name=expt_name, save_dir=wandb_path)
lr_monitor = LearningRateMonitor(logging_interval="epoch")
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath=os.path.join(wandb_path, proj_name, expt_name),
    filename="model-{epoch:02d}-{val_loss:.5f}",
    save_weights_only=True,
)
model = deep_phase(
    in_channels, out_channels, latent_channels, window_size, lr, batch_size
)
trainer = pl.Trainer(
    logger=wandb_logger,
    accelerator="gpu",
    devices=-1,
    max_epochs=epochs,
    check_val_every_n_epoch=1,
    deterministic=True,
    callbacks=[lr_monitor, checkpoint_callback],
)
trainer.fit(model, datamodule=dm)

wandb.finish()
