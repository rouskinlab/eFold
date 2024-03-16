import wandb
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from lightning.pytorch import Trainer
from efold.core.callbacks import WandbFitLogger, KaggleLogger
from efold.config import device
from efold import DataModule, create_model
import torch
from lightning.pytorch.strategies import DDPStrategy

if __name__ == "__main__":
    USE_WANDB = True
    print("Running on device: {}".format(device))
    if USE_WANDB:
        wandb_logger = WandbLogger(project="ribonanza-solution", name="first-run")

    params = {
        "embed_dim": 192,
        "num_heads": 6,
        "hidden_dim": 768,
        "num_encoders": 12,
        "max_len": 210,
        "lr": 1e-3,
        "optimizer_fn": torch.optim.Adam,
        "use_se": True,
        "gamma": 0.998,
        "batch_size": 8,
    }

    # Create dataset
    dm = DataModule(
        name=["ribo500"],
        data_type=["dms", "shape", "structure"],
        force_download=False,
        batch_size=params["batch_size"],
        num_workers=0,
        train_split=None,
        valid_split=2048,
        predict_split=0,
        shuffle_valid=False,
        max_len=params["max_len"],
    )

    params["dim_per_head"] = params["embed_dim"] // params["num_heads"]

    model = create_model(
        model="ribonanza",
        params=params,
    )

    if USE_WANDB:
        wandb_logger.watch(model, log="all")

    # train with both splits
    trainer = Trainer(
        devices=8,
        strategy=DDPStrategy(find_unused_parameters=True),
        max_epochs=1000,
        accelerator=device,
        logger=wandb_logger if USE_WANDB else None,
        callbacks=[
            LearningRateMonitor(logging_interval="epoch"),
            WandbFitLogger(dm=dm),
        ]
        if USE_WANDB
        else [],
        enable_checkpointing=False,
        accumulate_grad_batches=16,
    )

    trainer.fit(model, datamodule=dm)

    if USE_WANDB:
        wandb.finish()
