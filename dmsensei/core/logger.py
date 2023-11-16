import wandb
from ..config import *
import numpy as np
import lightning.pytorch as pl
from .listofdatapoints import ListOfDatapoints


class Logger:
    def __init__(self, pl_module: pl.LightningModule, batch_size) -> None:
        self._model = pl_module
        self._batch_size = batch_size

    def log(self, stage, metric, value, data_type=None):
        self._model.log(
            name="/".join([k for k in [stage, data_type, metric] if k is not None]),
            value=float(value),
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            batch_size=self._batch_size,
        )

    def final_score(self, average_score, data_type):
        wandb.log(
            {
                "final/{}_best_{}".format(
                    data_type, REFERENCE_METRIC[data_type]
                ): average_score
            }
        )

    def train_loss(self, loss):
        self.log("train", "loss", loss)

    def valid_loss(self, loss):
        self.log("valid", "loss", loss)

    def valid_plot(self, data_type, name, plot):
        wandb.log(
            {"/".join(["valid", data_type, name]): plot},
        )

    def test_plot(self, dataloader, data_type, name, plot):
        wandb.log(
            {"/".join(["test", dataloader, data_type, name]): plot},
        )

    def error_metrics_pack(self, stage: str, list_of_datapoints: ListOfDatapoints):
        for datapoint in list_of_datapoints:
            for data_type, data in datapoint.metrics.items():
                for name_metric, metric in data.items():
                    self.log(stage, name_metric, metric, data_type)
