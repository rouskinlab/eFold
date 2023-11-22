from rouskinhf import import_dataset, seq2int, dot2int, int2dot, int2seq
from torch.utils.data import Dataset as TorchDataset
from torch import nn, tensor, float32, int64, stack, uint8
from numpy import array, ndarray
import numpy as np
from ..config import DEFAULT_FORMAT, device, TEST_SETS_NAMES
from .embeddings import (
    base_pairs_to_int_dot_bracket,
    sequence_to_int,
    base_pairs_to_pairing_matrix,
)
import torch
from torch.utils.data import DataLoader, random_split, Subset
from typing import Tuple
import lightning.pytorch as pl
import torch.nn.functional as F
from functools import partial
import wandb
from lightning.pytorch.loggers import WandbLogger
from ..config import UKN
import copy
import numpy as np
from typing import Union
import pdb
from typing import List
from .batch import Batch
from .listofdatapoints import ListOfDatapoints


class Dataset(TorchDataset):
    def __init__(
        self,
        name: str,
        data_type: List[str],
        force_download: bool,
        tqdm=True,
    ) -> None:
        super().__init__()
        self.name = name
        data = import_dataset(name, force_download=force_download)
        self.data_type = data_type + ["sequence"] if "sequence" not in data_type else data_type
        self.list_of_datapoints = ListOfDatapoints.from_rouskinhf(
            data, data_type, name=name, tqdm=tqdm
        )

    def __len__(self) -> int:
        return len(self.list_of_datapoints)

    def __getitem__(self, index) -> tuple:
        return self.list_of_datapoints[index]

    def collate_fn(self, batch_data):
        batch = Batch.from_list_of_datapoints(batch_data, self.data_type)
        return batch
