from os.path import dirname, join
from os import makedirs
import pickle
import torch
import wandb
import os


class Loader:
    def __init__(
        self,
        path=None,
    ) -> None:
        self.path = (
            path + ".pkl" if path is not None and not path.endswith(".pkl") else path
        )
        makedirs(dirname(self.get_path()), exist_ok=True)

    def get_path(self, extension=".pkl"):
        return (
            join(
                "models",
                wandb.run.name + extension
                if wandb.run is not None
                else "default_name" + extension,
            )
            if self.path is None
            else self.path
        )

    def load_from_pickle(self):
        return pickle.load(open(self.get_path(), "rb"))

    def load_from_weights(self, safe_load=True):
        if (
            safe_load
            and os.path.exists(self.get_path(extension=".pt"))
            or not safe_load
        ):
            return torch.load(self.get_path(extension=".pt"))

    def dump(self, model):
        torch.save(model.state_dict(), self.get_path(extension=".pt"))
        # pickle.dump(model, open(self.get_path(), "wb"))
