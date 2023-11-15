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
import torch
import pickle

sys.path.append(os.path.abspath("."))

if __name__ == "__main__":
    USE_WANDB = 1
    print("Running on device: {}".format(device))
    if USE_WANDB:
        wandb.login()
        wandb_logger = WandbLogger(project="trash_project_for_testing")

    d_model = 64
    lr = 1e-3
    gamma = 0.999
    batch_size = 16

    # Create dataset
    dm = DataModule(
        name=["ribonanza"],
        data_type=["dms", "shape"],
        force_download=False,
        batch_size=batch_size,
        num_workers=0,
        train_split=500,
        valid_split=250,
        predict_split=0,
        overfit_mode=False,
        shuffle_valid=False,
    )

    model = create_model(
        model="transformer",
        data="multi",
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
        gamma=gamma,
    )

    # model.load_state_dict(
    #     torch.load(
    #         "/Users/yvesmartin/src/DMSensei/smooth-blaze-41.pt",
    #         map_location=torch.device("mps"),
    #     )
    # )
    model.to(device)

    if USE_WANDB:
        wandb_logger.watch(model, log="all")

    # train with both splits
    trainer = Trainer(
        accelerator=device,
        # devices=2,
        # strategy="ddp",
        # precision="16-mixed",
        # accumulate_grad_batches=2,
        max_epochs=1,
        # log_every_n_steps=10,
        logger=wandb_logger if USE_WANDB else None,
        callbacks=[  # EarlyStopping(monitor="valid/loss", mode='min', patience=5),
    #         # LearningRateMonitor(logging_interval="epoch"),
            MyWandbLogger(dm=dm, model=model, batch_size=batch_size),
            KaggleLogger(push_to_kaggle=False),
    #         # ModelChecker(log_every_nstep=1000, model=model),
        ]
    #     if USE_WANDB
    #     else [],
    #     enable_checkpointing=False,
    )

    # trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)
    trainer.predict(model, datamodule=dm)

    if USE_WANDB:
        wandb.finish()
