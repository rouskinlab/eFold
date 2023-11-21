import wandb
from ..config import *
import numpy as np
import lightning.pytorch as pl
from .batch import Batch
from .metrics import metric_factory
from ..config import POSSIBLE_METRICS


class Logger:
    def __init__(self, pl_module: pl.LightningModule, batch_size) -> None:
        self._model = pl_module
        self._batch_size = batch_size

    def log(self, stage, metric, value, data_type=None):
        self._model.log(
            name="/".join([k for k in [stage, data_type, metric] if k is not None]),
            value=float(value),
            sync_dist=True,
            # on_step=True,
            # on_epoch=True,
            batch_size=self._batch_size,
            add_dataloader_idx=False,
        )

    def best_score(self, average_score, data_type):
        wandb.log(
            {
                "final/{}_best_{}".format(
                    data_type, REFERENCE_METRIC[data_type]
                ): average_score
            }
        )

    def train_loss(self, loss):
        self.log("train", "loss", loss)

    def valid_loss(self, loss, dataloader_idx):
        if dataloader_idx == 0:
            self.log("valid", "loss", loss)
        if dataloader_idx == 1:
            self.log("valid", "lossLQ", loss)
        if dataloader_idx == 2:
            self.log("valid", "lossHQ", loss)

    def valid_plot(self, data_type, name, plot):
        wandb.log(
            {"/".join(["valid", data_type, name]): plot},
        )

    def test_plot(self, dataloader, data_type, name, plot):
        wandb.log(
            {"/".join(["test", dataloader, data_type, name]): plot},
        )

    def error_metrics_pack(self, stage: str, batch: Batch):
        metrics = batch.compute_metrics()
        for data_type, data in metrics.items():
            for metric, value in data.items():
                self.log(
                    stage=stage,
                    metric=metric,
                    value=value,
                    data_type=data_type,
                )
