import numpy as np
import wandb
from lightning.pytorch.loggers import WandbLogger
import os
import sys
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import pandas as pd
from lightning.pytorch import Trainer
from dmsensei.util import str2fun
from dmsensei.config import device
from dmsensei import DataModule, create_model, metrics

sys.path.append(os.path.abspath("."))
# os.system('source /Users/alberic/Desktop/Pro/RouskinLab/projects/deep_learning/RNA_data/env')

sys.path.append(os.path.abspath("."))
# os.system('source
# /Users/alberic/Desktop/Pro/RouskinLab/projects/deep_learning/RNA_data/env')
# why do you need this?


sweep_configuration = {
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "final/best_r2"},
    "parameters": {
        "batch_size": {
            "distribution": "categorical",
            "values": [4, 8, 16, 32, 64, 128],
        },
        "embedding_dim": {"distribution": "int_uniform", "max": 256, "min": 64},
        "loss_fn": {"distribution": "categorical", "values": ["mse_loss", "l1_loss"]},
        "lr": {"distribution": "log_uniform", "max": 0.0005, "min": 0.000001},
        "model_dim": {"distribution": "int_uniform", "max": 32, "min": 8},
        "optimizer_fn": {"distribution": "categorical", "values": ["Adam"]},
        "shuffle_train": {"distribution": "categorical", "values": [True]},
        "shuffle_valid": {"distribution": "categorical", "values": [False]},
        "weight_decay": {"distribution": "uniform", "max": 0.0002, "min": 0},
        "dropout": {"distribution": "uniform", "max": 0.5, "min": 0},
        "hidden_layers": {
            "distribution": "categorical",
            "values": [
                [4096, 4096, 2048, 1024, 512],
                [8096, 4096, 4096, 2048, 1024, 512],
                [8096, 4096, 2048, 1024, 512],
                [8096, 4096, 2048, 1024],
            ],
        },
        "max_epochs": {"distribution": "categorical", "values": [1000]},
    },
}


def train(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

        loader = DataModule(
            name="utr",
            data="dms",
            force_download=False,
            batch_size=config["batch_size"],
            num_workers=0,
            train_split=0.8,
            valid_split=0.2,
            zero_padding_to=1024,
            shuffle_train=config["shuffle_train"],
            shuffle_valid=config["shuffle_valid"],
        )
        model = create_model(
            data="dms",
            model="mlp",
            hidden_layers=config["hidden_layers"],
            lr=config["lr"],
            loss_fn=str2fun[config["loss_fn"]],
            optimizer_fn=str2fun[config["optimizer_fn"]],
            input_dim=1024,
            embedding_dim=config["embedding_dim"],
            model_dim=config["model_dim"],
            weight_decay=config["weight_decay"],
            dropout=config["dropout"],
        ).to(device)

        trainer = Trainer(
            logger=WandbLogger(),
            max_epochs=config["max_epochs"],
            callbacks=[
                PredictionLogger(data="dms"),
                EarlyStopping(monitor="valid/loss"),
                ModelChecker(model=model, log_every_nstep=5),
            ],
        )

        print("Running on device: {}".format(device))

        trainer.fit(model, datamodule=loader)
        wandb.finish()


if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="mlp_dms_sweep")
    wandb.agent(sweep_id=sweep_id, function=train, count=50)
