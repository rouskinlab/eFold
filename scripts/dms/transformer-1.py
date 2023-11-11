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
import torch
import pickle

sys.path.append(os.path.abspath("."))

if __name__ == "__main__":

    USE_WANDB = False
    print("Running on device: {}".format(device))
    if USE_WANDB:
        wandb.login()       
        wandb_logger = WandbLogger(project="dms-alby_test")

    d_model = 64
    lr = 1e-3
    gamma = 0.999
    batch_size = 32

    # Create dataset
    dm = DataModule(
        name=["ribo-test"],
        data_type=["dms", "shape"],
        force_download=False,
        batch_size=batch_size,
        num_workers=9,
        train_split=0,
        valid_split=0,
        predict_split=100,
        overfit_mode=False,
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
        gamma=gamma
    )
    
    model.load_state_dict(torch.load('/Users/yvesmartin/src/DMSensei/smooth-blaze-41.pt', map_location=torch.device('mps')))

    if USE_WANDB:
        wandb_logger.watch(model, log="all")

    # train with both splits
    trainer = Trainer(
        accelerator=device,
        # devices=2,
        # strategy="ddp",
        # precision="16-mixed",
        # accumulate_grad_batches=2,
        max_epochs=2,
        log_every_n_steps=10,
        # logger=wandb_logger if USE_WANDB else None,
        # callbacks=[  # EarlyStopping(monitor="valid/loss", mode='min', patience=5),
        #     LearningRateMonitor(logging_interval="epoch"),
        #     PredictionLogger(data="dms"),
        #     ModelChecker(log_every_nstep=1000, model=model),
        # ]
        # if USE_WANDB
        # else [],
        enable_checkpointing=False,
    )

    # trainer.fit(model, datamodule=dm)
    # trainer.test(model, datamodule=dm)
    out = trainer.predict(model, datamodule=dm)
    
    # post process the prediction
    out = pd.DataFrame([unit for batch in out for unit in batch])
    sequence_ids = pd.read_csv('test_sequences_ids.csv')
    out = pd.merge(sequence_ids, out, on='reference')
    dms, shape = np.concatenate(out['dms'].values), np.concatenate(out['shape'].values)
    pd.DataFrame({'reactivity_DMS_MaP': dms, 'reactivity_2A3_MaP': shape}).reset_index().rename(columns={'index': 'id'}).to_csv('predictions.csv', index=False)
    
    # save predictions
    pickle.dump(out, open("predictions.pkl", "wb"))

    if USE_WANDB:
        pickle.dump(model, open(os.path.join('models', WandbLogger.name + ".pkl"), "wb"))

    if USE_WANDB:
        wandb.finish()
