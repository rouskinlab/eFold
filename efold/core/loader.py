import os
from os import listdir, makedirs
from os.path import dirname

import torch


class Loader:
    def __init__(
        self,
        path: str,
    ) -> None:
        self.path = path
        makedirs(dirname(self.get_path()), exist_ok=True)

    @classmethod
    def find_best_model(cls, prefix: str) -> "Loader":
        models = [model for model in listdir("models") if model.startswith(prefix)]
        if len(models) == 0:
            return None
        models.sort(key=lambda x: float(x.split("_mae:")[-1].split(".")[0].replace("-", ".")))
        return cls(path="models/" + models[0])

    def get_path(self) -> str:
        return self.path

    def get_name(self) -> str:
        return self.path.split("/")[-1].split(".")[0]

    def write_in_log(self, epoch: int, mae: float) -> "Loader":
        with open("models/_log.txt", "a") as f:
            f.write(f"{epoch} {self.get_name()}\t{mae}\n")
        return self

    def load_from_weights(self, safe_load: bool = True) -> dict:
        if safe_load and os.path.exists(self.get_path()) or not safe_load:
            return torch.load(self.get_path())
        raise FileNotFoundError(f"File {self.get_path()} not found")

    def dump(self, model) -> "Loader":
        torch.save(model.state_dict(), self.get_path())
        return self
