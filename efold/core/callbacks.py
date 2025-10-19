import os
from lightning import LightningModule, Trainer
import lightning.pytorch as pl
import torch
import numpy as np
import pandas as pd
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.utilities import rank_zero_only
import wandb
from typing import Any

from efold.core.visualisation import plot_factory
from efold.core.metrics import metric_factory
from efold.core.datamodule import DataModule
from efold.core.loader import Loader
from efold.core.batch import Batch
from efold.settings import (
    TEST_SETS_NAMES,
    REF_METRIC_SIGN,
    REFERENCE_METRIC,
    DATA_TYPES_TEST_SETS,
    POSSIBLE_METRICS,
)
from efold.core.logger import Logger, LocalLogger


class ModelCheckpoint(pl.Callback):
    def __init__(self, every_n_epoch=1) -> None:
        super().__init__()
        self.every_n_epoch = every_n_epoch

    @rank_zero_only
    def on_validation_end(self, trainer: Trainer, pl_module, dataloader_idx=0):
        if dataloader_idx:
            return

        # Save best model
        if wandb.run is None:
            return

        if trainer.current_epoch % self.every_n_epoch != 0:
            return

        name = "{}_epoch{}.pt".format(wandb.run.name, trainer.current_epoch)
        loader = Loader(path="models/" + name)
        # logs what MAE it corresponds to
        loader.dump(pl_module)
