import sys, os

sys.path.append(os.path.abspath("."))
from dmsensei import DataModule, create_model, metrics
from dmsensei.config import device
from dmsensei.core.callbacks import PredictionLogger, ModelChecker
from lightning.pytorch.callbacks import LearningRateMonitor
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
import pickle

sys.path.append(os.path.abspath("."))

if __name__ == "__main__":

    USE_WANDB = True
    print("Running on device: {}".format(device))
    if USE_WANDB:
        wandb.login()       
        wandb_logger = WandbLogger(project="dms-alby_test")

    model = "transformer"
    data = "dms"

    d_model = 64
    lr = 1e-3
    gamma = 0.999
    batch_size = 64

    # Create dataset
    dm = DataModule(
        name=["utr"],
        data=data,
        force_download=True,
        batch_size=batch_size,
        num_workers=1,
        train_split=1000,
        valid_split=234,
        overfit_mode=False,
    )

    model = create_model(
        data=data,
        model=model,
        ntoken=5,
        n_struct=2,
        d_model=d_model,
        nhead=16,
        d_hid=d_model,
        nlayers=8,
        dropout=0.0,
        lr=lr,
        weight_decay=0,
        wandb=USE_WANDB,
        gamma=gamma
    )


    if USE_WANDB:
        wandb_logger.watch(model, log="all")

    # train with both splits
    trainer = Trainer(
        accelerator=device,
        devices=2,
        strategy="ddp",
        precision="16-mixed",
        accumulate_grad_batches=2,
        max_epochs=100,
        log_every_n_steps=100,
        logger=wandb_logger if USE_WANDB else None,
        callbacks=[  # EarlyStopping(monitor="valid/loss", mode='min', patience=5),
            LearningRateMonitor(logging_interval="epoch"),
            PredictionLogger(data="dms"),
            ModelChecker(log_every_nstep=1000, model=model),
        ]
        if USE_WANDB
        else [],
        enable_checkpointing=False,
    )

    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)

    if USE_WANDB:
        pickle.dump(model, open(os.path.join('models', WandbLogger.name + ".pkl"), "wb"))

    if USE_WANDB:
        wandb.finish()
