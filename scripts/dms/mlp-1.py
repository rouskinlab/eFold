import sys, os

sys.path.append(os.path.abspath("."))
# os.system('source /Users/alberic/Desktop/Pro/RouskinLab/projects/deep_learning/RNA_data/env')
from dmsensei import DataModule, create_model, metrics
from dmsensei.config import device
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

sys.path.append(os.path.abspath("."))
# os.system('source /Users/alberic/Desktop/Pro/RouskinLab/projects/deep_learning/RNA_data/env') why do you need this?

if __name__ == "__main__":
    print("Running on device: {}".format(device))
    
    BATCH_SIZE = 4
    LR = 5e-5
    WD = 1e-4
    EMBED = 128
    MODEL = 16
    DROPOUT = [0.5, 0.5, 0.3, 0.3, 0.2]
    
    wandb_logger = WandbLogger(project="dms mlp", name="mlp - 5L - lr {} - batch {} - wd {} - embed {} - model {} - dropout".format(LR, BATCH_SIZE, WD, EMBED, MODEL))

    MAX_LEN = 1024

    # Create dataset
    dm = DataModule(
        name=['utr','utr'],
        data="dms",
        force_download=False,
        batch_size=BATCH_SIZE,
        num_workers=1,
        train_split=0.8,
        valid_split=0.2,
        zero_padding_to=MAX_LEN,
        shuffle_train=True,
        shuffle_valid=False,
        # overfit_mode=True
    )

    model = create_model(
        data="dms",
        model="mlp",
        hidden_layers=[4096, 4096, 2048, 1024, 512],
        lr=LR,
        input_dim=MAX_LEN,
        embedding_dim=EMBED,
        model_dim=MODEL,
        weight_decay=WD,
        dropout=DROPOUT,
        #        loss_fn=metrics.r2_score,
    )

    wandb_logger.watch(model, log="all")

    # train with both splits
    trainer = Trainer(
        max_epochs=1,
        log_every_n_steps=1,
        logger=wandb_logger,
        accelerator=device,
        callbacks=[  # EarlyStopping(monitor="valid/loss", mode='min', patience=5),
            PredictionLogger(data="dms"),
            ModelChecker(model=model, log_every_nstep=1000),
        ],
        enable_checkpointing=False,
    )

    trainer.fit(model, datamodule=dm)
    # trainer.test(model, datamodule=dm)
    wandb.finish()
