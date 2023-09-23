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

    # Create dataset
    dm = DataModule(
        name="utr",
        data="dms",
        force_download=False,
        batch_size=64,
        num_workers=1,
        train_split=0.8,
        valid_split=0.2,
        zero_padding_to=MAX_LEN,
        overfit_mode=True,
    )

    model = create_model(
        data="dms",
        model="mlp",
        hidden_layers=[8192, 8192, 8192, 4096, 2048, 1024, 512],
        lr=1e-4,
        input_dim=MAX_LEN,
        embedding_dim=128,
        model_dim=16,
        #        loss_fn=metrics.r2_score,
    )

    wandb_logger.watch(model, log="all")

    # train with both splits
    trainer = Trainer(
        max_epochs=500,
        log_every_n_steps=1,
        logger=wandb_logger,
        accelerator=device,
        callbacks=[  # EarlyStopping(monitor="valid/loss", mode='min', patience=5),
            PredictionLogger(data="dms"),
            ModelChecker(log_every_nstep=1000),
        ],
        enable_checkpointing=False,
    )

    trainer.fit(model, datamodule=dm)
    # trainer.test(model, datamodule=dm)
    wandb.finish()
