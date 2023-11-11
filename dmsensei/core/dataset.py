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
    def __init__(self, name: str, data_type:List[str], force_download: bool, quality: float = 1.0) -> None:
        super().__init__()
        self.name = name
        data = import_dataset(name, force_download=force_download)
        self.data_type = data_type + ["sequence"]
        # save data
        self.references = data["references"]
        self.sequences = data["sequences"]
        self.dms = data["dms"] if "dms" in data else None
        self.base_pairs = data["base_pairs"] if "base_pairs" in data else None
        self.shape = data["shape"] if "shape" in data else None
        self.quality = quality
        self.data, self.metadata = self._wrangle_data()

    def __len__(self) -> int:
        return len(self.sequences)

    def _wrangle_data(self):
        def get_array(attr, index, astype=None):
            if not hasattr(self, attr) or getattr(self, attr) is None:
                return None
            d = tensor(getattr(self, attr)[index].astype(astype))
            if torch.isnan(d).all():
                return None
            return d

        data, metadata = [], []
        for index in range(len(self)):
            sequence = get_array("sequences", index, astype=np.int32)
            dms = get_array("dms", index, astype=np.float32)
            structure = get_array(
                "base_pairs", index, astype=np.int32
            )  # would be great to have a
            shape = get_array("shape", index, astype=np.float32)

            line = {
                "sequence": sequence,
                "dms": dms,
                "structure": structure,
                "shape": shape,
            }
            drop_keys = [k for k, v in line.items() if v is None]
            for k in drop_keys:
                del line[k]
            data.append(line)
            metadata.append(
                {
                    "reference": self.references[index],
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
            "structure": seq2int["X"],
            "shape": UKN,
        }
        if data_type == "sequence":
            return F.pad(arr, (0, L - len(arr)), value=padding_values[data_type])
        elif data_type == "structure":
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

        metadata_out = {'length': lengths}
        for k in metadata[0].keys():
            metadata_out[k] = [d[k] for d in metadata]

        return data_out, metadata_out
