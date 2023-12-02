import sys, os
# import envbash
# envbash.load.load_envbash('.env')

sys.path.append(os.path.abspath("."))
from dmsensei import DataModule, create_model
from dmsensei.config import device
from dmsensei.core.callbacks import WandbFitLogger, KaggleLogger#, WandbTestLogger
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
from lightning.pytorch.strategies import DDPStrategy

sys.path.append(os.path.abspath("."))

# Train loop
if __name__ == "__main__":
    USE_WANDB = 1
    print("Running on device: {}".format(device))
    if USE_WANDB:
        project = 'Structure_prediction'
        wandb_logger = WandbLogger(project=project)

    # fit loop
    batch_size = 16
    dm = DataModule(
        name=["ribo-kaggle"],
        data_type=["dms",'shape'],
        force_download=False,
        batch_size=batch_size,
        num_workers=1,
        train_split=5000, # all but valid_split
        valid_split=2048,
        predict_split=0,
        overfit_mode=False,
        shuffle_valid=False,
    )

    model = create_model(
        model="cnn",
        data="multi",
        quality=False,
        ntoken=5,
        d_model=128,
        d_cnn=256,
        n_heads=16,
        dropout=0,
        lr=5e-5,
        weight_decay=0,
        gamma=0.997,
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
        strategy=DDPStrategy(find_unused_parameters=True),
        precision="16-mixed",
        max_epochs=1000,
        log_every_n_steps=1,
        accumulate_grad_batches=2,
        logger=wandb_logger if USE_WANDB else None,
        callbacks=[
            LearningRateMonitor(logging_interval='epoch'),
            # PredictionLogger(data="dms"),
            # ModelChecker(log_every_nstep=10000, model=model),
            WandbFitLogger(dm=dm, load_model=None),
            # WandbTestLogger(dm=dm, n_best_worst=10, load_model='best'), # 'best', None or path to model
        ]
        if USE_WANDB
        else [],
        enable_checkpointing=False,
    )
    
    
    trainer.fit(model, datamodule=dm)
    # trainer.test(model, datamodule=dm)
    
    
    # # Predict loop
    # dm = DataModule(
    #     name=["ribo-test"],
    #     data_type=["dms", "shape"],
    #     force_download=False,
    #     batch_size=128,
    #     num_workers=0,
    #     train_split=0,
    #     valid_split=0,
    #     predict_split=1.,
    #     overfit_mode=False,
    #     shuffle_valid=False,
    # )
    
    # trainer = Trainer(
    #     accelerator=device,
    #     callbacks=[KaggleLogger(
    #         push_to_kaggle=True, 
    #         load_model='best' # 'best', None or path to model
    #         )]
    # )
    
    # trainer.predict(model, datamodule=dm)

    if USE_WANDB:
        wandb.finish()
