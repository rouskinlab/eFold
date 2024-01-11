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
from ..config import UKN


class Dataset(TorchDataset):
    def __init__(
        self,
        name: str,
        data_type: List[str],
        refs: np.ndarray,
        length: np.ndarray,
        sequence: np.ndarray,
        max_len: int,
        min_len: int,
        structure_padding_value: float,
        use_error: bool,
        dms: DMSDataset = None,
        shape: SHAPEDataset = None,
        structure: StructureDataset = None,
        sort_by_length: bool = False,
    ) -> None:
        super().__init__()
        self.name = name
        self.data_type = data_type
        self.use_error = use_error
        self.refs = refs
        self.length = length
        self.sequence = sequence
        self.dms = dms
        self.shape = shape
        self.structure = structure
        self.structure_padding_value = structure_padding_value
        self.L = max(self.length)
        self._remove_sequences_out_of_length_interval(min_len, max_len)
        
        if sort_by_length:
            self.sort()

    def _remove_sequences_out_of_length_interval(self, min_len, max_len):
        if max_len is None:
            max_len = np.inf    
        if min_len is None:
            min_len = 0
        if min_len > max_len:
            raise ValueError("min_len must be smaller than max_len")
        idx_out = [i for i, l in enumerate(self.length) if l >= max_len or l <= min_len]
        for idx in idx_out[::-1]:
            del self.refs[idx]
            del self.length[idx]
            del self.sequence[idx]
            if self.dms is not None:
                del self.dms[idx]
            if self.shape is not None:
                del self.shape[idx]
            if self.structure is not None:
                del self.structure[idx]

    def __add__(self, other: "Dataset") -> "Dataset":
        # if self.name == other.name:
        #     raise ValueError("Dataset are the same")
        if self.structure_padding_value != other.structure_padding_value:
            raise ValueError("Structure padding value are not the same")
        return Dataset(
            name=self.name,
            data_type=list(set(self.data_type + other.data_type)),
            use_error=self.use_error or other.use_error,
            max_len=None,
            min_len=None,
            structure_padding_value=self.structure_padding_value,
            refs=np.concatenate([self.refs, other.refs]).tolist(),
            length=np.concatenate([self.length, other.length]).tolist(),
            sequence=np.concatenate([self.sequence, other.sequence]).tolist(),
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
        use_error: bool = False,
        max_len=None,
        min_len=None,
        structure_padding_value: float = UKN,
        sort_by_length: bool = False,
        tqdm=True,
    ):
        path = Path(name=name)
        if force_download:
            path.clear()

        if os.path.exists(path.get_reference()):
            print("Loading dataset from disk")

            print("Load references              \r", end="")
            refs = path.load_reference().tolist()

            print("Load lengths         \r", end="")
            length = path.load_length().tolist()
            L = max(length)

            print("Load sequences         \r", end="")
            sequence = path.load_sequence().tolist()
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
            print("Loading dataset from HF")

            print("Dump lengths              \r", end="")
            length = [len(d["sequence"]) for d in data.values()]
            path.dump_length(np.array(length))

            print("Dump references              \r", end="")
            refs = list(data.keys())
            path.dump_reference(np.array(list(data.keys())))
            L = max(length)

            print("Dump sequences              \r", end="")
            sequence = [d["sequence"] for d in data.values()]
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

        print("Done!                            ")

        return cls(
            name=name,
            data_type=data_type,
            use_error=use_error,
            refs=refs,
            length=length,
            sequence=sequence,
            dms=dms,
            shape=shape,
            structure=structure,
            max_len=max_len,
            min_len=min_len,
            structure_padding_value=structure_padding_value,
            sort_by_length=sort_by_length,
        )

    def sort(self):
        idx_sorted = np.argsort(self.length)
        self.refs = [self.refs[i] for i in idx_sorted]
        self.length = [self.length[i] for i in idx_sorted]
        self.sequence = [self.sequence[i] for i in idx_sorted]
        if self.dms is not None:
            self.dms.sort(idx_sorted)
        if self.shape is not None:
            self.shape.sort(idx_sorted)
        if self.structure is not None:
            self.structure.sort(idx_sorted)

    def __len__(self) -> int:
        return len(self.sequence)

    def __getitem__(self, index) -> tuple:
        out = {
            "reference": self.refs[index],
            "sequence": self.sequence[index],
            "length": self.length[index],
        }
        for attr in ["dms", "shape", "structure"]:
            out[attr] = (
                getattr(self, attr)[index] if getattr(self, attr) != None else None
            )
        return out

    def collate_fn(self, batch_data):
        batch = Batch.from_dataset_items(
            batch_data,
            self.data_type,
            use_error=self.use_error,
            structure_padding_value=self.structure_padding_value,
        )
        return batch
