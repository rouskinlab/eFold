import os
import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset as TorchDataset, Dataset
from typing import List

from .batch import Batch
from rouskinhf import get_dataset
from .datatype import DMSDataset, SHAPEDataset, StructureDataset
from .embeddings import sequence_to_int
from .util import _pad
from .path import Path


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
        self.data_type = data_type

        path = Path(name=name)
        if force_download:
            path.clear()

        if os.path.exists(path.get_reference()):
            print("Loading dataset from disk...")

            print("Load references              \r", end="")
            self.refs = path.load_reference()

            print("Load lengths         \r", end="")
            self.length = path.load_length()
            self.L = max(self.length)

            print("Load sequences         \r", end="")
            self.sequence = torch.tensor(path.load_sequence())
            self.dms, self.shape, self.structure = None, None, None
            if "dms" in data_type:
                print("Load dms         \r", end="")
                self.dms = path.load_dms()
            if "shape" in data_type:
                print("Load shape         \r", end="")
                self.shape = path.load_shape()
            if "structure" in data_type:
                print("Load structure      \r", end="")
                self.structure = path.load_structure()

        else:
            data = get_dataset(
                name=name,
                force_download=force_download,
                tqdm=tqdm,
            )
            print("Loading dataset into memory...")

            print("Dump lengths              \r", end="")
            self.length = np.array([len(d["sequence"]) for d in data.values()])
            path.dump_length(self.length)

            print("Dump references              \r", end="")
            self.refs = np.array(list(data.keys()))
            path.dump_reference(self.refs)
            self.L = max(self.length)

            print("Dump sequences              \r", end="")
            self.sequence = torch.stack(
                [
                    _pad(sequence_to_int(data[ref]["sequence"]), self.L, "sequence")
                    for ref in self.refs
                ]
            )
            path.dump_sequence(np.array(self.sequence))

            print("Dump dms              \r", end="")
            self.dms = DMSDataset.from_data_json(data, self.L, self.refs)
            path.dump_dms(self.dms)

            print("Dump shape              \r", end="")
            self.shape = SHAPEDataset.from_data_json(data, self.L, self.refs)
            path.dump_shape(self.shape)

            print("Dump structure              \r", end="")
            self.structure = StructureDataset.from_data_json(data, self.L, self.refs)
            path.dump_structure(self.structure)

            for dt in ["dms", "shape", "structure"]:
                if dt not in self.data_type:
                    setattr(self, dt, None)

            print("Done!")

    def __len__(self) -> int:
        return len(self.sequence)

    def __getitem__(self, index) -> tuple:
        return {
            "reference": self.refs[index],
            "sequence": self.sequence[index],
            "length": self.length[index],
            "dms": self.dms[index] if self.dms is not None else None,
            "shape": self.shape[index] if self.shape is not None else None,
            "structure": self.structure[index] if self.structure is not None else None,
        }

    def __add__(self, other: Dataset) -> ConcatDataset:
        raise NotImplementedError

    def collate_fn(self, batch_data):
        batch = Batch.from_dataset_items(batch_data, self.data_type)
        return batch
