import sys, os

sys.path.append(os.path.abspath("."))
# os.system('source /Users/alberic/Desktop/Pro/RouskinLab/projects/deep_learning/RNA_data/env')
from dmsensei import DataModule, create_model, metrics
from dmsensei.config import device
from dmsensei.core.callbacks import PredictionLogger, ModelChecker
from pytorch_lightning import Trainer
import pandas as pd
from dmsensei.core.callbacks import PredictionLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from dmsensei.core.callbacks import PredictionLogger
import pandas as pd
from pytorch_lightning import Trainer
from dmsensei.config import device
import sys
import os
from pytorch_lightning.loggers import WandbLogger
import wandb
import numpy as np

sys.path.append(os.path.abspath("."))
# os.system('source /Users/alberic/Desktop/Pro/RouskinLab/projects/deep_learning/RNA_data/env') why do you need this?

if __name__ == "__main__":
    print("Running on device: {}".format(device))
    wandb_logger = WandbLogger(project="dms")

    MAX_LEN = 1024
    model = 'transformer'
    data = 'dms'

    # Create dataset
    dm = DataModule(
        name="utr",
        data=data,
        force_download=False,
        batch_size=4,
        num_workers=0,
        train_split=784,
        valid_split=80,
    )

    model = create_model(
        data=data,
        model=model,
        ntoken=5,
        n_struct=2,
        d_model=512,
        nhead=8,
        d_hid=512,
        nlayers=8,
        dropout=0.3,
        lr=1e-6,
    )

    wandb_logger.watch(model, log="all")

    # train with both splits
    trainer = Trainer(
        max_epochs=100,
        log_every_n_steps=5,
        logger=wandb_logger,
        accelerator=device,
        callbacks=[  # EarlyStopping(monitor="valid/loss", mode='min', patience=5),
            PredictionLogger(data="dms"),
            ModelChecker(log_every_nstep=1000, model=model),
        ],
        enable_checkpointing=False,
    )

    trainer.fit(model, datamodule=dm)
    # trainer.test(model, datamodule=dm)
    wandb.finish()
