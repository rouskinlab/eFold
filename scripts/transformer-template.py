import os
import sys

import envbash
import wandb
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

from efold import settings
from efold.core import callbacks, datamodule
from efold.models import factory

envbash.load.load_envbash(".env")
sys.path.append(os.path.abspath("."))

sys.path.append(os.path.abspath("."))

if __name__ == "__main__":
    USE_WANDB = True
    print("Running on device: {}".format(settings.device))
    if USE_WANDB:
        wandb_logger = WandbLogger(project="CHANGE_ME", name="debug")

    lr = 5e-4
    gamma = 0.998
    batch_size = 128

    # Create dataset
    dm = datamodule.DataModule(
        name=["ribo-kaggle"],
        data_type=["dms", "shape"],
        force_download=False,
        batch_size=batch_size,
        num_workers=0,
        train_split=256,
        valid_split=128,
        predict_split=0,
        overfit_mode=False,
        shuffle_valid=False,
    )

    model = factory.create_model(
        model="transformer",
        data="multi",
        weight_data=True,
        ntoken=5,
        d_model=128,
        nhead=16,
        d_hid=256,
        nlayers=8,
        dropout=0,
        lr=lr,
        weight_decay=0,
        wandb=USE_WANDB,
        gamma=gamma,
    )

    # import torch
    # model.load_state_dict(torch.load('/root/efold/models/baseline_SRRF_loss19-0.pt',
    #                                  map_location=torch.device(device)))

    if USE_WANDB:
        wandb_logger.watch(model, log="all")

    # train with both splits
    trainer = Trainer(
        accelerator=settings.device,
        # devices=4,
        # strategy="ddp",
        # precision="16-mixed",
        # accumulate_grad_batches=2,
        max_epochs=300,
        logger=wandb_logger if USE_WANDB else None,
        callbacks=[
            LearningRateMonitor(logging_interval="epoch"),
            callbacks.WandbFitLogger(dm=dm, batch_size=batch_size, load_model=None),
        ]
        if USE_WANDB
        else [],
        enable_checkpointing=False,
    )

    trainer.fit(model, datamodule=dm)

    dm = datamodule.DataModule(
        name=["ribo-test"],
        data_type=["dms", "shape"],
        force_download=False,
        batch_size=batch_size,
        num_workers=0,
        train_split=0,
        valid_split=0,
        predict_split=1.0,
        overfit_mode=False,
        shuffle_valid=False,
    )

    trainer = Trainer(
        accelerator=settings.device,
        devices=1,
        callbacks=[
            # don't change this
            callbacks.KaggleLogger(push_to_kaggle=True, load_model=None)
        ],
    )

    trainer.predict(model, datamodule=dm)

    if USE_WANDB:
        wandb.finish()
