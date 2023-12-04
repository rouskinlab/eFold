import wandb
from ..config import *
import lightning.pytorch as pl


class Logger:
    def __init__(self, pl_module: pl.LightningModule, batch_size) -> None:
        self._model = pl_module
        self._batch_size = batch_size

    def log(self, stage, metric, value, data_type=None):
        if value is None:
            return
        self._model.log(
            name="/".join([k for k in [stage, data_type, metric]
                          if k is not None]),
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
        self.log(
            "train",
            "loss",
            loss.item() if hasattr(
                loss,
                "item") else loss)

    def valid_loss(self, loss, is_test=False):
        stage = 'valid' if not is_test else 'valid/ribo-test'
        self.log(stage, "loss", loss.item() if hasattr(loss, "item") else loss)

    def valid_loss_pack(self, losses: dict, is_test=False):
        for data_type, loss in losses.items():
            stage = 'valid' if not is_test else 'valid/ribo-test'
            self.log(stage, "loss_{}".format(data_type), loss.item())

    def valid_plot(self, data_type, name, plot):
        wandb.log(
            {"/".join(["valid", data_type, name]): plot},
        )

    def test_plot(self, dataloader, data_type, name, plot):
        wandb.log(
            {"/".join(["test", dataloader, data_type, name]): plot},
        )

    def error_metrics_pack(self, stage: str, metrics: dict):
        for data_type, data in metrics.items():
            for metric, value in data.items():
                self.log(
                    stage=stage,
                    metric=metric,
                    value=value,
                    data_type=data_type,
                )
