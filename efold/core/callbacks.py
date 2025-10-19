import lightning.pytorch as pl
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.utilities import rank_zero_only
import wandb

from efold import settings
from efold.core import batch, datamodule, loader, logger, metrics, visualisation


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
        loader_obj = loader.Loader(path="models/" + name)
        # logs what MAE it corresponds to
        loader_obj.dump(pl_module)
