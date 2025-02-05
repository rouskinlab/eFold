import os
import sys
sys.path.append(os.path.abspath("."))

from lightning.pytorch.strategies import DDPStrategy
import wandb
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor
from efold.core.callbacks import ModelCheckpoint
from efold.config import device
from efold import DataModule, create_model


# Train loop
if __name__ == "__main__":
    USE_WANDB = False
    STRATEGY = "random"
    n_gpu = 1

    print("Running on device: {}".format(device))
    if USE_WANDB:
        wandb_logger = WandbLogger(project='test')

    # fit loop
    batch_size = 1
    dm = DataModule(
        name=["efold_train"], 
        strategy=STRATEGY,
        shuffle_train=False if STRATEGY == "ddp" else True,
        data_type=["structure"],  #
        force_download=False,
        batch_size=batch_size,
        max_len=1000,
        min_len=1,
        structure_padding_value=0,
        train_split=None,
        external_valid=["yack_valid"],
    )

    model = create_model(
        model="efold",
        ntoken=5,
        d_model=64,
        c_z=32,
        d_cnn=64,
        num_blocks=4,
        no_recycles=0,
        dropout=0,
        lr=1e-3,
        weight_decay=0,
        gamma=0.995,
        wandb=USE_WANDB,
    )

    if USE_WANDB:
        wandb_logger.watch(model, log="all")

    trainer = Trainer(
        accelerator=device,
        devices=n_gpu if STRATEGY == "ddp" else 1,
        strategy=DDPStrategy(find_unused_parameters=False) if STRATEGY == "ddp" else 'auto',
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
    trainer.test(model, datamodule=dm)

    if USE_WANDB:
        wandb.finish()
