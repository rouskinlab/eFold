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
import wandb

sys.path.append(os.path.abspath("."))

if __name__ == "__main__":
    USE_WANDB = 1
    print("Running on device: {}".format(device))
    if USE_WANDB:
        wandb.login()
        project = 'CHANGE_ME'
        wandb.init(project=project)
        wandb_logger = WandbLogger(project=project)

    d_model = 64
    lr = 1e-3
    gamma = 0.999
    batch_size = 16

    # Create dataset
    dm = DataModule(
        name=["ribo-HQ", "ribo-LQ"],
        data_type=["dms", "shape"],
        force_download=False,
        batch_size=batch_size,
        num_workers=0,
        train_split=256,
        valid_split=256,
        predict_split=0,
        overfit_mode=False,
        shuffle_valid=False,
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
        gamma=gamma,
    )

    if USE_WANDB:
        wandb_logger.watch(model, log="all")

    # train with both splits
    trainer = Trainer(
        accelerator=device,
        # devices=2,
        # strategy="ddp",
        # precision="16-mixed",
        # accumulate_grad_batches=2,
        max_epochs=3,
        # log_every_n_steps=10,
        logger=wandb_logger if USE_WANDB else None,
        callbacks=[  # EarlyStopping(monitor="valid/loss", mode='min', patience=5),
    #         # LearningRateMonitor(logging_interval="epoch"),
            WandbFitLogger(dm=dm, batch_size=batch_size, load_model='/Users/yvesmartin/src/DMSensei/models/polar-moon-19.pt'),
            WandbTestLogger(dm=dm, n_best_worst=10, load_model='/Users/yvesmartin/src/DMSensei/models/polar-moon-19.pt'),
    #         # ModelChecker(log_every_nstep=1000, model=model),
        ]
        if USE_WANDB
        else [],
        enable_checkpointing=False,
    )

    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)
    
    dm = DataModule(
        name=["ribo-test"],
        data_type=["dms", "shape"],
        force_download=False,
        batch_size=batch_size,
        num_workers=0,
        train_split=0,
        valid_split=0,
        predict_split=1.,
        overfit_mode=False,
        shuffle_valid=False,
    )
    
    trainer = Trainer(
        accelerator=device,
        callbacks=[KaggleLogger(
            push_to_kaggle=False, 
            load_model='best' # 'best', None or path to model
            )]
    )

    trainer.predict(model, datamodule=dm)

    if USE_WANDB:
        wandb.finish()
