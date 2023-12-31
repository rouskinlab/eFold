import os
import sys

sys.path.append(os.path.abspath("."))

from dmsensei.core.callbacks import WandbFitLogger, KaggleLogger  # , WandbTestLogger
from lightning.pytorch.strategies import DDPStrategy
import wandb
from lightning.pytorch.loggers import WandbLogger
import pandas as pd
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor
from dmsensei.config import device
from dmsensei import DataModule, create_model
import sys
import os

# import envbash
# envbash.load.load_envbash('.env')


sys.path.append(os.path.abspath("."))

# Train loop
if __name__ == "__main__":
    USE_WANDB = 1
    print("Running on device: {}".format(device))
    if USE_WANDB:
        project = "Structure_prediction"
        wandb_logger = WandbLogger(project=project)

    # fit loop
    batch_size = 1
    dm = DataModule(
        name=["yack_valid"],
        shuffle_train='ddp',
        shuffle_valid='ddp',
        data_type=["dms", "shape", "structure"],  #
        force_download=False,
        batch_size=1,
        max_len=1024,
        structure_padding_value=0,
        train_split=1024,
        external_valid=["yack_valid", "test_valid"],
    )

    model = create_model(
        model="cnn",
        ntoken=5,
        d_model=64,
        d_cnn=128,
        n_heads=16,
        dropout=0,
        lr=1e-4,
        weight_decay=0,
        # gamma=0.995,
        wandb=USE_WANDB,
    )

    # import torch
    # model.load_state_dict(torch.load('/root/DMSensei/dmsensei/models/trained_models/vocal-voice-12.pt',
    #                                  map_location=torch.device(device)))

    if USE_WANDB:
        wandb_logger.watch(model, log="all")

    trainer = Trainer(
        accelerator=device,
        devices=8,
        strategy=DDPStrategy(),
        precision="16-mixed",
        max_epochs=1000,
        log_every_n_steps=1,
        accumulate_grad_batches=32,
        use_distributed_sampler=False,
        logger=wandb_logger if USE_WANDB else None,
        callbacks=[
            LearningRateMonitor(logging_interval="epoch"),
   ]
        if USE_WANDB
        else [],
        enable_checkpointing=False,
    )

    trainer.fit(model, datamodule=dm)

    if USE_WANDB:
        wandb.finish()
