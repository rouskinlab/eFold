import sys, os

sys.path.append(os.path.abspath("."))
from dmsensei import DataModule, create_model, metrics
from dmsensei.config import device
from dmsensei.core.callbacks import PredictionLogger, ModelChecker
from lightning.pytorch import Trainer
from dmsensei.core.callbacks import PredictionLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from dmsensei.core.callbacks import PredictionLogger
import pandas as pd
from lightning.pytorch import Trainer
from dmsensei.config import device
import sys
import os
from lightning.pytorch.loggers import WandbLogger
import wandb 
import numpy as np

sys.path.append(os.path.abspath("."))

if __name__ == "__main__":
    
    USE_WANDB = True
    print("Running on device: {}".format(device))
    if USE_WANDB:
        wandb_logger = WandbLogger(project="dms")

    model = 'transformer'
    data = 'dms'

    # Create dataset
    dm = DataModule(
        name="utr",
        data=data,
        force_download=False,
        batch_size=4,
        num_workers=1,
        train_split=0.05,
        valid_split=0,
        overfit_mode=True
    )

    model = create_model(
        data=data,
        model=model,
        ntoken=5,
        n_struct=2,
        d_model= 128,
        nhead=16,
        d_hid=1024,
        nlayers=48,
        dropout=0.0,
        lr=1e-5,
        weight_decay=0,
        wandb=USE_WANDB,
    )

    if USE_WANDB:
        wandb_logger.watch(model, log="all")

    # train with both splits
    trainer = Trainer(
        max_epochs=1000,
        log_every_n_steps=5,
        logger=wandb_logger if USE_WANDB else None,
        accelerator=device,
        callbacks=[  EarlyStopping(monitor="valid/loss", mode='min', patience=5),
           PredictionLogger(data="dms"),
           ModelChecker(log_every_nstep=1000, model=model),
        ] if USE_WANDB else [],
        enable_checkpointing=False,
    )

    trainer.fit(model, datamodule=dm)
    # trainer.test(model, datamodule=dm)
    
    if USE_WANDB:
        wandb.finish()
