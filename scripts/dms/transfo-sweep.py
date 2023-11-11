import sys, os

sys.path.append(os.path.abspath("."))
# os.system('source /Users/alberic/Desktop/Pro/RouskinLab/projects/deep_learning/RNA_data/env')
from dmsensei import DataModule, create_model, metrics
from dmsensei.config import device
from dmsensei.util import str2fun
from dmsensei.core.callbacks import PredictionLogger, ModelChecker
from lightning.pytorch import Trainer
import pandas as pd
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
from lightning.pytorch.callbacks import LearningRateMonitor
import torch

sys.path.append(os.path.abspath("."))
# os.system('source /Users/alberic/Desktop/Pro/RouskinLab/projects/deep_learning/RNA_data/env') why do you need this?


sweep_configuration = {
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "final/best_r2"},
    "parameters": {
        "batch_size": {"distribution": "categorical", "values": [4, 8, 16]},
        "loss_fn": {"distribution": "categorical", "values": ["l1_loss", "mse_loss"]},
        "lr": {"distribution": "log_uniform", "max": -7, "min": -11},
        "d_model": {"distribution": "categorical", "values": [64, 128]},
        "nhead": {"distribution": "categorical", "values": [8, 16]},
        "d_hid": {"distribution": "categorical", "values": [128, 256]},
        "nlayers": {"distribution": "categorical", "values": [2, 4, 8, 16]},
        # "optimizer_fn": {"distribution": "categorical", "values": ["Adam"]},
        # "shuffle_train": {"distribution": "categorical", "values": [True]},
        # "shuffle_valid": {"distribution": "categorical", "values": [False]},
        #  "weight_decay": {"distribution": "uniform", "max": 1e-5, "min": 0},
        #   "dropout": {"distribution": "uniform", "max": 0.5, "min": 0},
        #   "max_epochs": {"distribution": "categorical", "values": [100]},
        #    "early_terminate": {"type": "hyperband", "min_iter": 3},
    },
}


def train(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

        loader = DataModule(
            name="ribonanza",
            data="dms",
            force_download=False,
            batch_size=config["batch_size"],
            num_workers=0,
            valid_split=2000,
            shuffle_train=True,  # config["shuffle_train"],
            shuffle_valid=False,  # config["shuffle_valid"],
        )

        model = create_model(
            data="dms",
            model="transformer",
            ntoken=5,
            d_model=config["d_model"],
            nhead=config["nhead"],
            d_hid=config["d_hid"],
            nlayers=config["nlayers"],
            dropout=0,  # config["dropout"],
            lr=config["lr"],
            loss_fn=str2fun[config["loss_fn"]],
            optimizer_fn=torch.optim.Adam,
            weight_decay=0,  # config["weight_decay"],
        ).to(device)

        wandb.watch(model, log="all")
        trainer = Trainer(
            logger=WandbLogger(),
            max_epochs=200,  # config['max_epochs'],
            callbacks=[
                predlogger := PredictionLogger(data="dms"),
                EarlyStopping(monitor="valid/loss", mode="min", patience=5),
                ModelChecker(model=model, log_every_nstep=5),
                LearningRateMonitor(logging_interval="epoch"),
            ],
        )

        print("Running on device: {}".format(device))

        trainer.fit(model, datamodule=loader)

        # load best model for testing
        model = torch.load(predlogger.best_model_path)
        trainer.test(model, datamodule=loader)
        wandb.finish()


if __name__ == "__main__":
    sweep_id = wandb.sweep(
        sweep=sweep_configuration, project="transfo4_ribonanza_sweep"
    )
    wandb.agent(sweep_id=sweep_id, function=train, count=100)
