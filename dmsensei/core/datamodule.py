from rouskinhf import import_dataset, seq2int, dot2int, int2dot, int2seq
from torch.utils.data import Dataset as TorchDataset
from torch import nn, tensor, float32, int64, stack
from numpy import array, ndarray
from ..config import DEFAULT_FORMAT, device, TEST_SETS_NAMES
from .embeddings import base_pairs_to_int_dot_bracket, sequence_to_int
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

class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        name: str,
        data: str,
        force_download=False,
        batch_size: int = 32,
        num_workers: int = 1,
        train_split: float = None,
        valid_split: float = 4096,
        predict_split: float = 0,
        zero_padding_to=None,
        overfit_mode=False,
        shuffle_train=True,
        shuffle_valid=False,
        shuffle_test=False,
        **kwargs,
    ):
        """DataModule for the Rouskin lab datasets.

        Args:
            name: name of the dataset on the Rouskin lab HuggingFace repository. Can be a list of names.
            data: type of the data (e.g. 'dms', 'structure')
            force_download: re-download the dataset from the Rouskin lab HuggingFace repository
            batch_size: batch size for the dataloaders
            num_workers: number of workers for the dataloaders
            train_split: percentage of the dataset to use for training or number of samples to use for training. If None, the entire dataset minus the validation set is used for training
            valid_split: percentage of the dataset to use for validation or number of samples to use for validation
            predict_split: percentage of the dataset to use for prediction or number of samples to use for prediction
            zero_padding_to: pad sequences to this length. If None, sequences are not padded.
            overfit_mode: if True, the train set is used for validation and testing. Useful for debugging. Default is False.
        """
        # Save arguments
        super().__init__(**kwargs)

        self.name = name
        self.data = data
        self.force_download = force_download
        self.dataloader_args = {"batch_size": batch_size, "num_workers": num_workers}
        self.splits = (train_split, valid_split, predict_split)
        self.zero_padding_to = zero_padding_to
        self.shuffle = {
            "train": shuffle_train,
            "valid": shuffle_valid,
            "test": shuffle_test,
        }

        # we need to know the max sequence length for padding
        self.overfit_mode = overfit_mode
        self.setup()

        # Log hyperparameters
        train_split, valid_split, _ = self.size_sets
        self.save_hyperparameters(ignore=["force_download"])

    def _use_multiple_datasets(self, name):
        if type(name) == str:
            return False
        elif type(name) == list or type(name) == tuple:
            return True
        raise ValueError("name must be a string or a list of strings")

    def _dataset_merge(self, datasets):
        merge = datasets[0]
        collate_fn = merge.collate_fn
        for dataset in datasets[1:]:
            merge = merge + dataset
        merge.collate_fn = collate_fn
        return merge

    def setup(self, stage: str = None):
        if stage in ["fit", None, "predict"]:
            dataFull = self._dataset_merge(
                [
                    Dataset(
                        name=name,
                        data=self.data,
                        force_download=self.force_download,
                        zero_padding_to=self.zero_padding_to,
                    )
                    for name in self.name
                ]
            )
        if stage == "fit" or stage is None:
            if not self._use_multiple_datasets(self.name):
                self.name = [self.name]

            self.size_sets = _compute_size_sets(len(dataFull), *self.splits[:2])
            self.train_set, self.val_set, _ = random_split(dataFull, self.size_sets)

        if stage == "test" or stage is None:
            self.test_sets = self._select_test_dataset(
                data=self.data, force_download=self.force_download
            )

        if stage == "predict" or stage is None:
            self.predict_set = Subset(dataFull, range(0, round(len(dataFull) * self.splits[2]) if type(self.splits[2]) == float else self.splits[2]))

        if stage is None:
            self.collate_fn = dataFull.collate_fn

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            shuffle=self.shuffle["train"],
            collate_fn=self.collate_fn,
            **self.dataloader_args,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set if not self.overfit_mode else self.train_set,
            shuffle=self.shuffle["valid"],
            collate_fn=self.collate_fn,
            **self.dataloader_args,
        )

    def test_dataloader(self):
        return [
            DataLoader(
                test_set,
                shuffle=self.shuffle["test"],
                collate_fn=self.collate_fn,
                **self.dataloader_args,
            )
            for test_set in self.test_sets
        ]

    def predict_dataloader(self):
        return DataLoader(
            self.predict_set,
            shuffle = False,
            collate_fn=self.collate_fn,
            **self.dataloader_args,
        )

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        pass

    def _select_test_dataset(self, data: str, force_download=False):
        return [
            Dataset(
                name=name,
                data=data,
                force_download=force_download,
                zero_padding_to=self.zero_padding_to,
            )
            for name in TEST_SETS_NAMES[data]
        ]


class Dataset:
    def __new__(cls, name: str, data: str, force_download, zero_padding_to):
        data = data.lower()
        if data == "dms":
            return DMSDataset(
                name, force_download=force_download, zero_padding_to=zero_padding_to
            )
        elif data == "structure":
            return StructureDataset(
                name, force_download=force_download, zero_padding_to=zero_padding_to
            )
        elif data == "sequence":
            return SequenceDataset(
                name, force_download=force_download, zero_padding_to=zero_padding_to
            )
        else:
            raise ValueError("Data must be either 'DMS' or 'structure'")


class TemplateDataset(TorchDataset):
    def __init__(self, name: str, data: str, force_download, zero_padding_to) -> None:
        super().__init__()
        self.name = name
        self.zero_padding_to = zero_padding_to
        data = import_dataset(name, data=data, force_download=force_download)

        # save data
        self.references = data["references"]
        self.sequences = data["sequences"]

        # save the maximum length of the sequences
        self.max_sequence_length = max([len(sequence) for sequence in self.sequences])

        return data

    def __len__(self) -> int:
        return len(self.sequences)

    def collate_fn(self, batch):
        raise NotImplementedError("This method must be implemented in the child class")


class DMSDataset(TemplateDataset):
    def __init__(self, name: str, force_download, zero_padding_to) -> None:
        data = super().__init__(
            name,
            data="DMS",
            force_download=force_download,
            zero_padding_to=zero_padding_to,
        )

        # quality check
        assert (
            len(data["references"]) == len(data["sequences"]) == len(data["DMS"])
        ), "Data is not consistent"

        # Load DMS data
        assert "DMS" in data.keys(), "DMS data not found"
        self.dms = data["DMS"]

    def __getitem__(self, index) -> tuple:
        sequence = tensor(self.sequences[index], dtype=int64)
        dms = tensor(self.dms[index], dtype=DEFAULT_FORMAT)

        assert len(sequence) == len(dms), "Data is not consistent"

        return sequence, dms

    def __repr__(self) -> str:
        return f"DMSDataset(name={self.name}, len={len(self.sequences)})"

    def collate_fn(self, batch):
        """Creates mini-batch tensors from the list of tuples (sequence, dms). Zero-pads sequences and dms. The sequences have variable length.

        Args:
            batch: list of tuple (sequence, dms).

        Returns:
            sequences: torch tensor of shape (batch_size, max_sequence_length)
            dms: torch tensor of shape (batch_size, max_sequence_length)
        """

        sequences, dms = zip(*batch)

        padding_length = 0
        # Find longest sequence in batch
        if self.zero_padding_to != None:
            max_all_sequences_length = max([len(sequence) for sequence in sequences])
            padding_length = self.zero_padding_to - max_all_sequences_length

            if padding_length < 0:
                raise ValueError(
                    "The maximum sequence length of the dataset is greater than the zero padding length. Please increase the zero padding length."
                )

        # Merge sequences (from tuple of 1D tensor to 2D tensor).
        sequences = F.pad(
            nn.utils.rnn.pad_sequence(sequences, batch_first=True),
            (0, padding_length),
            value=0,
        )

        dms = F.pad(
            nn.utils.rnn.pad_sequence(dms, batch_first=True, padding_value=UKN),
            (0, padding_length),
            value=UKN,
        )

        return sequences, dms



class SequenceDataset(TemplateDataset):
    def __init__(self, name: str, force_download, zero_padding_to) -> None:
        data = super().__init__(
            name,
            data="sequence",
            force_download=force_download,
            zero_padding_to=zero_padding_to,
        )

        # quality check
        assert (
            len(data["references"]) == len(data["sequences"])
        ), "Data is not consistent"


    def __getitem__(self, index) -> tuple:

        sequence = tensor(self.sequences[index], dtype=int64)

        return sequence

    def __repr__(self) -> str:
        return f"SequenceDataset(name={self.name}, len={len(self.sequences)})"

    def collate_fn(self, batch):
        """Creates mini-batch tensors from the list of tuples (sequence, dms). Zero-pads sequences and dms. The sequences have variable length.

        Args:
            batch: list of tuple (sequence, dms).

        Returns:
            sequences: torch tensor of shape (batch_size, max_sequence_length)
        """

        sequences = batch
        padding_length = 0
        # Find longest sequence in batch
        if self.zero_padding_to != None:
            max_all_sequences_length = max([len(sequence) for sequence in sequences])
            padding_length = self.zero_padding_to - max_all_sequences_length

            if padding_length < 0:
                raise ValueError(
                    "The maximum sequence length of the dataset is greater than the zero padding length. Please increase the zero padding length."
                )

        # Merge sequences (from tuple of 1D tensor to 2D tensor).
        sequences = F.pad(
            nn.utils.rnn.pad_sequence(sequences, batch_first=True),
            (0, padding_length),
            value=0,
        )

        return sequences


class StructureDataset(TemplateDataset):
    def __init__(self, name: str, force_download, zero_padding_to) -> None:
        data = super().__init__(
            name,
            data="structure",
            force_download=force_download,
            zero_padding_to=zero_padding_to,
        )

        # quality check
        assert (
            len(data["references"]) == len(data["sequences"]) == len(data["base_pairs"])
        ), "Data is not consistent"

        # Load structure data
        assert "base_pairs" in data.keys(), "Structure data not found"
        self.base_pairs = data["base_pairs"]

    def __getitem__(self, index) -> tuple:
        sequence = self.sequences[index]
        structure = base_pairs_to_int_dot_bracket(
            self.base_pairs[index], len(sequence), dtype=int64
        )

        return sequence, structure

    def __repr__(self) -> str:
        return f"StructureDataset(name={self.name}, len={len(self.sequences)})"


def _compute_size_sets(len_data, train_split=None, valid_split=4000):
    """Returns the size of the train and validation sets given the split percentages and the length of the dataset.

    Args:
        len_data: int
        train_split: float between 0 and 1, or integer, or None. If None, the train split is computed as 1 - valid_split. Default is None.
        valid_split: float between 0 and 1, or integer, or None. Default is 4000.

    Returns:
        train_set_size: int
        valid_set_size: int
        buffer_size: int

    Raises:
        AssertionError: if the split percentages do not sum to 1 or less, or if the train split is less than 0.

    Examples:
    >>> _compute_size_sets(100, 40, 0.2)
    (40, 20, 40)
    >>> _compute_size_sets(100, 40, 20)
    (40, 20, 40)
    >>> _compute_size_sets(100, None, 10)
    (90, 10, 0)
    >>> _compute_size_sets(100, 0.4, 0.2)
    (40, 20, 40)
    >>> _compute_size_sets(100, 0.4, 20)
    (40, 20, 40)
    """

    if valid_split <= 1 and type(valid_split) == float:
        valid_split = int(valid_split * len_data)

    if train_split is None:
        train_split = len_data - valid_split

    elif train_split <= 1 and type(train_split) == float:
        train_split = len_data - int((1 - train_split) * len_data)

    assert (
        train_split + valid_split <= len_data
    ), "The sum of the splits must be less than the length of the dataset"

    return train_split, valid_split, len_data - train_split - valid_split
