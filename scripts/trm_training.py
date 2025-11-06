import os
import sys
# Ensure local repo path takes precedence over any installed efold package
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy

from efold.core.callbacks import ModelCheckpoint
from efold.config import device
from efold import DataModule, create_model

import random
import numpy as np

SEED = 1338

# Seed everything
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
seed_everything(SEED, workers=True)


if __name__ == "__main__":
    USE_WANDB = True
    # Choose strategy dynamically based on accelerator
    STRATEGY = "random" if device == "cpu" else "ddp"
    n_gpu = 1

    print(f"Running on device: {device}")
    if USE_WANDB:
        wandb_logger = WandbLogger(project="efold-trm", entity="rouskin-lab", name="trm-pretraining")

    # Data
    batch_size = 1
    dm = DataModule(
        name=["efold_train"],  # pretraining dataset
        strategy=STRATEGY,
        shuffle_train=False if STRATEGY == "ddp" else True,
        data_type=["structure"],
        force_download=False,
        batch_size=batch_size,
        max_len=1000,
        min_len=1,
        structure_padding_value=0,
        train_split=None,
        external_valid=["yack_valid", "PDB", "archiveII", "lncRNA", "viral_fragments"],
    )

    # Model: Tiny Recursive Model
    model = create_model(
        model="trm",
        ntoken=5,
        c_z=32,
        k_steps=4,
        z_cycles=3,
        dropout=0.0,
        lr=3e-4,
        gamma=0.995,
        wandb=USE_WANDB,
    )

    if USE_WANDB:
        wandb_logger.watch(model, log="all")

    trainer = Trainer(
        accelerator=device,
        devices=n_gpu if STRATEGY == "ddp" else 1,
        strategy=DDPStrategy(find_unused_parameters=False) if STRATEGY == "ddp" else "auto",
        max_epochs=15,
        log_every_n_steps=1,
        accumulate_grad_batches=32,
        use_distributed_sampler=STRATEGY != "ddp",
        logger=wandb_logger if USE_WANDB else None,
        callbacks=[
            LearningRateMonitor(logging_interval="epoch"),
            ModelCheckpoint(every_n_epoch=1),
        ]
        if USE_WANDB
        else [],
        enable_checkpointing=False,
    )

    trainer.fit(model, datamodule=dm)

    if USE_WANDB:
        import wandb
        wandb.finish()


