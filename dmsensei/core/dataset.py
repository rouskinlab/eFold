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


class Dataset(TorchDataset):
    def __init__(
        self,
        name: str,
        data_type: List[str],
        force_download: bool,
        quality: float = 1.0,
    ) -> None:
        super().__init__()
        self.name = name
        data = import_dataset(name, force_download=force_download)
        self.data_type = data_type + ["sequence"]
        # save data
        self.reference = data["references"]
        self.sequence = data["sequences"]
        self.dms = data["dms"] if "dms" in data else None
        self.base_pairs = data["base_pairs"] if "base_pairs" in data else None
        self.shape = data["shape"] if "shape" in data else None
        self.quality = quality
        self.data, self.metadata = self._wrangle_data()

    def __len__(self) -> int:
        return len(self.sequence)

    def _wrangle_data(self):
        # TODO #5 group the sequences by length
        def get_array(attr, index, astype=None):
            if not hasattr(self, attr) or getattr(self, attr) is None:
                return None
            d = tensor(getattr(self, attr)[index].astype(astype))
            if not torch.isnan(d).all():
                return d

        data, metadata = [], []
        for index in range(len(self)):
            line = {}
            astype = {
                "dms": np.float32,
                "structure": np.int32,
                "shape": np.float32,
                "sequence": np.int32,
            }
            for name in self.data_type:
                arr = get_array(name, index, astype=astype[name])
                if arr != None:
                    line[name] = arr

            data.append(line)
            metadata.append(
                {
                    "reference": self.reference[index],
                    "index": index,
                    "quality": self.quality,
                }
            )
        return data, metadata

    def __getitem__(self, index) -> tuple:
        return self.data[index], self.metadata[index]

    def _pad(self, arr, L, data_type):
        padding_values = {
            "sequence": 0,
            "dms": UKN,
            "shape": UKN,
        }
        if data_type == "structure":
            return base_pairs_to_pairing_matrix(arr, L)
        else:
            return F.pad(arr, (0, L - len(arr)), value=padding_values[data_type])

    def collate_fn(self, batch):
        data, metadata = zip(*batch)

        # pad and stack tensors
        data_out = {}
        lengths = [len(d["sequence"]) for d in data]
        for k in self.data_type:
            padded_data = [
                (idx, self._pad(d[k], max(lengths), k))
                for idx, d in enumerate(data)
                if k in d
            ]
            if len(padded_data) == 0:
                continue
            indexes, values = zip(*padded_data)
            if k == "sequence":
                data_out[k] = stack(values)
            else:
                data_out[k] = {"indexes": tensor(indexes), "values": stack(values)}

        metadata_out = {"length": lengths}
        for k in metadata[0].keys():
            metadata_out[k] = [d[k] for d in metadata]

        return data_out, metadata_out
