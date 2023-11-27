import sys, os

sys.path.append(os.path.abspath("."))
from dmsensei import DataModule, create_model
from dmsensei.config import device
from dmsensei.core.callbacks import WandbFitLogger, KaggleLogger, WandbTestLogger
from lightning.pytorch import Trainer
from lightning.pytorch import Trainer
from dmsensei.config import device
import sys
import os
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor
import wandb

sys.path.append(os.path.abspath("."))

if __name__ == "__main__":
    USE_WANDB = 1
    print("Running on device: {}".format(device))
    if USE_WANDB:
        # wandb.login()
        project = 'Transformer-tuning'
        # wandb.init(project=project)
        wandb_logger = WandbLogger(project=project)

        

    d_model = 192
    lr = 5e-4
    # gamma = 0.998
    batch_size = 128

    # Create dataset
    dm = DataModule(
        name=["ribo-kaggle"],
        data_type=["dms", "shape"],
        force_download=False,
        batch_size=batch_size,
        num_workers=0,
        train_split=298281,
        valid_split=4096,
        predict_split=0,
        overfit_mode=False,
        shuffle_valid=False,
    )

    model = create_model(
        model="transformer",
        data="multi",
        quality=False,
        ntoken=5,
        d_model=d_model,
        nhead=6,
        d_hid=d_model*4,
        nlayers=12,
        dropout=0.1,
        lr=lr,
        weight_decay=0.004,
        wandb=USE_WANDB,
        # gamma=gamma,
    )

    # import torch
    # model.load_state_dict(torch.load('/root/DMSensei/models/baseline_SRRF_loss19-0.pt',
    #                                  map_location=torch.device(device)))

    if USE_WANDB:
        wandb_logger.watch(model, log="all")

    # train with both splits
    trainer = Trainer(
        accelerator=device,
        devices=4,
        strategy="ddp",
        # precision="16-mixed",
        # accumulate_grad_batches=2,
        max_epochs=500,
        # log_every_n_steps=10,
        logger=wandb_logger if USE_WANDB else None,
        callbacks=[  # EarlyStopping(monitor="valid/loss", mode='min', patience=5),
            LearningRateMonitor(logging_interval="epoch"),
            WandbFitLogger(dm=dm, batch_size=batch_size, load_model=None),
            WandbTestLogger(dm=dm, n_best_worst=10, load_model='best'), # 'best', None or path to model
        ]
        if USE_WANDB
        else [],
        enable_checkpointing=False,
    )

    trainer.fit(model, datamodule=dm)
    # trainer.test(model, datamodule=dm)
    
    # dm = DataModule(
    #     name=["ribo-test"],
    #     data_type=["dms", "shape"],
    #     force_download=False,
    #     batch_size=batch_size,
    #     num_workers=0,
    #     train_split=0,
    #     valid_split=0,
    #     predict_split=1.,
    #     overfit_mode=False,
    #     shuffle_valid=False,
    # )
    
    # trainer = Trainer(
    #     accelerator=device,
    #     devices=1,
    #     callbacks=[KaggleLogger(
    #         push_to_kaggle=True, 
    #         load_model=None # 'best', None or path to model
    #         )]
    # )

    # trainer.predict(model, datamodule=dm)

    if USE_WANDB:
        wandb.finish()
