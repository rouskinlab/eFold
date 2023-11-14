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
from lightning.pytorch.strategies import DDPStrategy
import torch

sys.path.append(os.path.abspath("."))

# Train loop
if __name__ == "__main__":
    USE_WANDB = True
    print("Running on device: {}".format(device))
    if USE_WANDB:
        wandb_logger = WandbLogger(project="dms-alby_test")

    model = "evoformer"
    data = "dms"

    # if len(wandb.config._as_dict())>1:
    #     print(wandb.config)
    #     batch_train_size = 128# wandb.config.batch_train_size
    #     d_model = wandb.config.d_model
    #     n_heads = wandb.config.n_heads
    #     d_hid = wandb.config.d_hid
    #     num_encoder_layers = wandb.config.num_encoder_layers
    #     num_decoder_layers = wandb.config.num_decoder_layers
    #     dropout = 0# wandb.config.dropout
    #     lr = wandb.config.lr
    #     max_lr = wandb.config.max_lr
    #     gamma = wandb.config.gamma
    #     past_window = wandb.config.past_window
    #     max_gradient = wandb.config.max_gradient
    #     activation = wandb.config.activation
    # else:
    #     batch_train_size = 128
    #     d_model = 64
    #     n_heads = 16
    #     d_hid = 64
    #     num_encoder_layers = 6
    #     num_decoder_layers = 6
    #     dropout = 0
    #     lr = 1e-3
    #     max_lr = 3e-3
    #     gamma = 0.9
    #     # max_gradient = 0.5
    #     activation = 'gelu'

    dm = DataModule(
        name=["ribonanza", "ribonanza_shape"],
        data=data,
        force_download=False,
        batch_size=16,
        num_workers=1,
        train_split=42000,
        valid_split=2531,
        overfit_mode=False,
    )

    model = create_model(
        data=data,
        model=model,
        ntoken=5,
        d_model=64,
        c_z=8,
        num_blocks=8,
        no_recycles=0,
        dropout=0,
        lr=3e-3,
        weight_decay=0,
        wandb=USE_WANDB,
    )

    # model.load_state_dict(torch.load('/root/DMSensei/dmsensei/models/trained_models/desert-puddle-114.pt',
    #                                  map_location=torch.device(device)))

    if USE_WANDB:
        wandb_logger.watch(model, log="all")

    trainer = Trainer(
        accelerator=device,
        devices=8,
        strategy=DDPStrategy(find_unused_parameters=True),
        max_epochs=1000,
        log_every_n_steps=1,
        accumulate_grad_batches=1,
        logger=wandb_logger if USE_WANDB else None,
        # precision="16-mixed",
        # gradient_clip_val=max_gradient,
        # gradient_clip_algorithm="value",
        callbacks=[
            # LearningRateMonitor(logging_interval='epoch'),
            PredictionLogger(data="dms"),
            ModelChecker(log_every_nstep=1000, model=model),
        ]
        if USE_WANDB
        else [],
        enable_checkpointing=False,
    )
    trainer.fit(model, datamodule=dm)
    # trainer.test(model, datamodule=dm)

    if USE_WANDB:
        wandb.finish()
