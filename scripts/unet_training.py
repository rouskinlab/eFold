import os
import sys
sys.path.append(os.path.abspath("."))

from efold.core.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy
import wandb
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor
from efold.config import device
from efold import DataModule, create_model


# Train loop
if __name__ == "__main__":
    USE_WANDB = False
    STRATEGY = "random"
    print("Running on device: {}".format(device))
    if USE_WANDB:
        wandb_logger = WandbLogger(project='test')

    # fit loop
    dm = DataModule(
        name=["efold_train"], # finetune: "utr", "pri_miRNA"
        strategy=STRATEGY,
        shuffle_train=False,
        data_type=["structure"], 
        force_download=False,
        batch_size=1,
        max_len=1000,
        min_len=1,
        structure_padding_value=0,
        train_split=None,
        external_valid=["yack_valid"],
    )

    model = create_model(
        model="unet",
        img_ch=17,
        output_ch=1,
        lr=1e-3,
        gamma=0.99,
        wandb=USE_WANDB,
    )

    if USE_WANDB:
        wandb_logger.watch(model, log="all")

    trainer = Trainer(
        accelerator=device,
        devices=8 if STRATEGY == "ddp" else 1,
        strategy=DDPStrategy(find_unused_parameters=False) if STRATEGY == "ddp" else 'auto',
        max_epochs=15,
        log_every_n_steps=1,
        accumulate_grad_batches=32,
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
    trainer.test(model, datamodule=dm)

    if USE_WANDB:
        wandb.finish()
