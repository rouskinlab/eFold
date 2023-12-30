import wandb
from ..config import *
import lightning.pytorch as pl
import os
import matplotlib.pyplot as plt
import torchmetrics


class LocalLogger:
    def __init__(self, path: str = "local_testing_output", overwrite: bool = False):
        self.path = path
        self.metrics_file = os.path.join(path, "metrics.txt")
        if overwrite:
            if os.path.exists(self.metrics_file):
                os.remove(self.metrics_file)
        os.makedirs(path, exist_ok=True)

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

    def test_plot(self, dataloader, data_type, name, plot, idx=None):
        img = wandb.Image(plot)
        plt.close(plot)

        wandb.log(
            {"/".join(["test", dataloader, data_type, name]): img},
        )
