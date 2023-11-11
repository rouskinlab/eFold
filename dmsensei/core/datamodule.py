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
import numpy as np
from typing import Union, List
from .dataset import Dataset


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        name: Union[str, list],
        data_type: List[str] = ["dms", "shape", "structure"],
        force_download=False,
        batch_size: int = 32,
        num_workers: int = 1,
        train_split: float = 0,
        valid_split: float = 0,
        predict_split: float = 0,
        overfit_mode=False,
        shuffle_train=True,
        shuffle_valid=False,
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

        if not self._use_multiple_datasets(name):
            self.name = [name]
        else:
            self.name = name

        self.force_download = force_download
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_type = data_type
        self.splits = {
            "train": train_split,
            "valid": valid_split,
            "predict": predict_split,
        }
        self.shuffle = {
            "train": shuffle_train,
            "valid": shuffle_valid,
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
        all_datasets = self._dataset_merge(
            [
                Dataset(
                    name=name,
                    data_type=self.data_type,
                    force_download=self.force_download,
                )
                for name in self.name
            ]
        )
        self.collate_fn = all_datasets.collate_fn

        if stage == "fit" or stage is None:
            self.size_sets = _compute_size_sets(
                len(all_datasets),
                train_split=self.splits["train"],
                valid_split=self.splits["valid"],
            )
            self.train_set, self.val_set, _ = random_split(all_datasets, self.size_sets)

        if stage == "test" or stage is None:
            self.test_sets = self._select_test_dataset(
                force_download=self.force_download
            )

        if stage == "predict" or stage is None:
            self.predict_set = Subset(
                all_datasets,
                range(
                    0,
                    round(len(all_datasets) * self.splits["predict"])
                    if type(self.splits["predict"]) == float
                    else self.splits["predict"],
                ),
            )

    def _select_test_dataset(self, force_download=False):
        return [
            Dataset(
                name=name,
                data_type=self.data_type,
                force_download=force_download,
            )
            for name in TEST_SETS_NAMES
        ]

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            shuffle=self.shuffle["train"],
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set if not self.overfit_mode else self.train_set,
            shuffle=self.shuffle["valid"],
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
        )

    def test_dataloader(self):
        return [
            DataLoader(
                test_set,
                collate_fn=self.collate_fn,
                batch_size=self.batch_size,
            )
            for test_set in self.test_sets
        ]

    def predict_dataloader(self):
        return DataLoader(
            self.predict_set,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
        )

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        pass


def _compute_size_sets(len_data, train_split, valid_split):
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
