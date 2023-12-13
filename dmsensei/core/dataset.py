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
        refs: np.ndarray,
        length: np.ndarray,
        sequence: torch.Tensor,
        dms: DMSDataset = None,
        shape: SHAPEDataset = None,
        structure: StructureDataset = None,
    ) -> None:
        super().__init__()
        self.name = name
        self.data_type = data_type
        self.refs = refs
        self.length = length
        self.sequence = sequence
        self.dms = dms
        self.shape = shape
        self.structure = structure
        self.L = max(self.length)

    def __add__(self, other: Dataset) -> Dataset:
        if self.name == other.name:
            raise ValueError("Dataset are the same")
        L = max(self.L, other.L)
        return Dataset(
            name=self.name,
            data_type=self.data_type,
            refs=np.concatenate([self.refs, other.refs]),
            length=np.concatenate([self.length, other.length]),
            sequence=torch.cat(
                [
                    _pad(self.sequence, L, "sequence"),
                    _pad(other.sequence, L, "sequence"),
                ]
            ),
            dms=self.dms + other.dms
            if self.dms is not None and other.dms is not None
            else None,
            shape=self.shape + other.shape
            if self.shape is not None and other.shape is not None
            else None,
            structure=self.structure + other.structure
            if self.structure is not None and other.structure is not None
            else None,
        )

    @classmethod
    def from_local_or_download(
        cls,
        name: str,
        data_type: List[str] = ["dms", "shape", "structure"],
        force_download: bool = False,
        tqdm=True,
    ):
        path = Path(name=name)
        if force_download:
            path.clear()

        if os.path.exists(path.get_reference()):
            print("Loading dataset from disk...")

            print("Load references              \r", end="")
            refs = path.load_reference()

            print("Load lengths         \r", end="")
            length = path.load_length()
            L = max(length)

            print("Load sequences         \r", end="")
            sequence = torch.tensor(path.load_sequence())
            dms, shape, structure = None, None, None
            if "dms" in data_type:
                print("Load dms         \r", end="")
                dms = path.load_dms()
            if "shape" in data_type:
                print("Load shape         \r", end="")
                shape = path.load_shape()
            if "structure" in data_type:
                print("Load structure      \r", end="")
                structure = path.load_structure()

        else:
            data = get_dataset(
                name=name,
                force_download=force_download,
                tqdm=tqdm,
            )
            print("Loading dataset into memory...")

            print("Dump lengths              \r", end="")
            length = np.array([len(d["sequence"]) for d in data.values()])
            path.dump_length(length)

            print("Dump references              \r", end="")
            refs = np.array(list(data.keys()))
            path.dump_reference(refs)
            L = max(length)

            print("Dump sequences              \r", end="")
            sequence = torch.stack(
                [
                    _pad(sequence_to_int(data[ref]["sequence"]), L, "sequence")
                    for ref in refs
                ]
            )
            path.dump_sequence(np.array(sequence))

            print("Dump dms              \r", end="")
            dms = DMSDataset.from_data_json(data, L, refs)
            path.dump_dms(dms)

            print("Dump shape              \r", end="")
            shape = SHAPEDataset.from_data_json(data, L, refs)
            path.dump_shape(shape)

            print("Dump structure              \r", end="")
            structure = StructureDataset.from_data_json(data, L, refs)
            path.dump_structure(structure)

            print("Done!")

        return cls(
            name=name,
            data_type=data_type,
            refs=refs,
            length=length,
            sequence=sequence,
            dms=dms,
            shape=shape,
            structure=structure,
        )

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

    def collate_fn(self, batch_data):
        batch = Batch.from_dataset_items(batch_data, self.data_type)
        return batch
