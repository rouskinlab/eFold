import os
import sys

sys.path.append(os.path.abspath("."))

from dmsensei.core.callbacks import ModelCheckpoint
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
from lightning.pytorch.profilers import PyTorchProfiler
# import envbash
# envbash.load.load_envbash('.env')


sys.path.append(os.path.abspath("."))

# Train loop
if __name__ == "__main__":
    USE_WANDB = 1
    STRATEGY = "ddp"
    n_gpu = 8
    batch_size = 32
    lr = 1e-4
    print("Running on device: {}".format(device))
    if USE_WANDB:
        project = "Unet"
        wandb_logger = WandbLogger(project=project)

    # fit loop
    dm = DataModule(
        name=["yack_train"], # finetune: "utr", "pri_miRNA", "archiveII"
        strategy=STRATEGY, #random, sorted or ddp
        shuffle_train=False,
        data_type=["structure"],  #
        force_download=False,
        batch_size=1,
        max_len=1024,
        structure_padding_value=0,
        train_split=None,
        external_valid=["yack_valid", "pri_miRNA", "human_mRNA_sarah", "lncRNA", "viral_fragments"], # finetune: "yack_valid", "human_mRNA"
    )

    model = create_model(
        model="unet",
        img_ch=17,
        output_ch=1,
        lr=lr*n_gpu if STRATEGY == "ddp" else lr,
        gamma=0.99,
        wandb=USE_WANDB,
    )

    # import torch
    # model.load_state_dict(torch.load('/root/DMSensei/dmsensei/models/trained_models/vocal-voice-12.pt',
    #                                  map_location=torch.device(device)))

    if USE_WANDB:
        wandb_logger.watch(model, log="all")

    trainer = Trainer(
        accelerator=device,
        devices=n_gpu if STRATEGY == "ddp" else 1,
        strategy=DDPStrategy(find_unused_parameters=False) if STRATEGY == "ddp" else 'auto',
        precision="16-mixed",
        max_epochs=1000,
        log_every_n_steps=1,
        accumulate_grad_batches=batch_size//n_gpu if STRATEGY == "ddp" else batch_size,
        logger=wandb_logger if USE_WANDB else None,
        callbacks=[
            ModelCheckpoint(every_n_epoch=1),
            LearningRateMonitor(logging_interval="epoch")
        ]
        if USE_WANDB
        else [],
        enable_checkpointing=False,
        use_distributed_sampler=STRATEGY != "ddp",
    )

    trainer.fit(model, datamodule=dm)
    # trainer.test(model, datamodule=dm)

    if USE_WANDB:
        wandb.finish()