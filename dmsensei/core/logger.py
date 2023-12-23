import wandb
from ..config import *
import lightning.pytorch as pl
import os
import matplotlib.pyplot as plt


class LocalLogger:
    def __init__(self, path: str = "local_testing_output", overwrite: bool = False):
        self.path = path
        self.metrics_file = os.path.join(path, "metrics.txt")
        if overwrite:
            if os.path.exists(self.metrics_file):
                os.remove(self.metrics_file)
        os.makedirs(path, exist_ok=True)

    def log(self, stage, metric, value, data_type=None):
        with open(self.metrics_file, "a") as f:
            f.write(f"{stage}/{data_type}/{metric}: {value}\n")

    def error_metrics_pack(self, stage: str, metrics: dict):
        for data_type, data in metrics.items():
            for metric, value in data.items():
                self.log(
                    stage=stage,
                    metric=metric,
                    value=value,
                    data_type=data_type,
                )

    def test_plot(self, dataloader, data_type, name, plot: plt.Figure, idx=None):
        # save the wandb Image to a png
        plot.savefig(
            os.path.join(self.path, f"{dataloader}_{data_type}_{name}_{idx}.png")
        )
        plt.close(plot)


class Logger:
    def __init__(self, pl_module: pl.LightningModule, batch_size) -> None:
        self._model = pl_module
        self._batch_size = batch_size

    def log(self, stage, metric, value, data_type=None):
        if value is None:
            return
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
        self.log("train", "loss", loss.item() if hasattr(loss, "item") else loss)

    def valid_loss(self, loss, is_test=False):
        stage = "valid" if not is_test else "ribo-valid"
        self.log(stage, "loss", loss.item() if hasattr(loss, "item") else loss)

    def valid_loss_pack(self, losses: dict, is_test=False):
        for data_type, loss in losses.items():
            stage = "valid" if not is_test else "ribo-valid"
            self.log(stage, "loss_{}".format(data_type), loss.item())

    def valid_plot(self, data_type, name, plot):
        plot = wandb.Image(plot)
        wandb.log(
            {"/".join(["valid", data_type, name]): plot},
        )

    def test_plot(self, dataloader, data_type, name, plot, idx=None):
        img = wandb.Image(plot)
        plt.close(plot)

        wandb.log(
            {"/".join(["test", dataloader, data_type, name]): img},
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
