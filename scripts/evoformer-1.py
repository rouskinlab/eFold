import sys, os

sys.path.append(os.path.abspath("."))
from dmsensei import DataModule, create_model, metrics
from dmsensei.config import device
from dmsensei.core.callbacks import ModelChecker, MyWandbLogger, KaggleLogger
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import pandas as pd
from lightning.pytorch import Trainer
from dmsensei.config import device
import sys
import os
from lightning.pytorch.loggers import WandbLogger
import wandb
import numpy as np
from lightning.pytorch.strategies import DDPStrategy
import torch
from torch import nn

sys.path.append(os.path.abspath("."))

# Train loop
if __name__ == "__main__":
    USE_WANDB = 0
    print("Running on device: {}".format(device))
    if USE_WANDB:
        wandb_logger = WandbLogger(project="Evoformer-dms")

    batch_size = 32
    dm = DataModule(
        name=["ribonanza"],
        data_type=["dms",'shape'],
        force_download=True,
        batch_size=batch_size,
        num_workers=1,
        train_split=47000,
        valid_split=2001,
        validLQ_split=3000,
        overfit_mode=False,
        shuffle_valid=False,
        shuffle_validLQ=False,
    )

    model = create_model(
        model="evoformer",
        data="multi",
        ntoken=5,
        d_model=64,
        c_z=8,
        num_blocks=12,
        no_recycles=0,
        dropout=0,
        lr=3e-3,
        weight_decay=0,
        gamma=0.997,
        wandb=USE_WANDB,
    )

    # model.load_state_dict(torch.load('/root/DMSensei/dmsensei/models/trained_models/desert-puddle-114.pt',
    #                                  map_location=torch.device(device)))

    if USE_WANDB:
        wandb_logger.watch(model, log="all")

    trainer = Trainer(
        accelerator=device,
        devices=1,
        # strategy=DDPStrategy(find_unused_parameters=True),
        max_epochs=1000,
        log_every_n_steps=1,
        accumulate_grad_batches=1,
        logger=wandb_logger if USE_WANDB else None,
        callbacks=[
            LearningRateMonitor(logging_interval='epoch'),
            # PredictionLogger(data="dms"),
            # ModelChecker(log_every_nstep=10000, model=model),
            MyWandbLogger(dm=dm, model=model, batch_size=batch_size),
            KaggleLogger(dm=dm, push_to_kaggle=False),
        ]
        if USE_WANDB
        else [],
        enable_checkpointing=False,
    )
    trainer.fit(model, datamodule=dm)
    # trainer.test(model, datamodule=dm)

    if USE_WANDB:
        wandb.finish()
