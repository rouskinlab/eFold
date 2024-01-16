from torch.utils.data import random_split, Subset
import lightning.pytorch as pl
from typing import Union, List
from .dataset import Dataset
from ..config import TEST_SETS, UKN
from .sampler import sampler_factory
from .dataloader import DataLoader
import numpy as np
import datetime


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        name: Union[str, list],
        batch_size: int,
        data_type: List[str] = ["dms", "shape", "structure"],
        force_download=False,
        num_workers: int = 0,
        train_split: float = 1.0,
        predict_split: float = 0,
        strategy="random",
        shuffle_train=True,
        shuffle_valid=False,
        external_valid=None,
        use_error=False,
        max_len=None,
        min_len=None,
        structure_padding_value=UKN,
        tqdm=True,
        buckets=None,
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
            sampler: 'bucket' or 'random'. If 'bucket', the data is sampled by bucketing sequences of similar lengths. If 'random', the data is sampled randomly. Default is 'bucket'.
            strategy: 'random', 'ddp' or 'sorted'
        """
        # Save arguments
        super().__init__(**kwargs)

        if not self._use_multiple_datasets(name):
            self.name = [name]
        else:
            self.name = name

        self.batch_size = batch_size
        self.strategy = strategy
        self.num_workers = num_workers
        self.data_type = data_type
        self.external_valid = external_valid
        self.splits = {
            "train": train_split,
            "predict": predict_split,
        }
        if strategy in ["ddp", "sorted"]:
            assert (
                shuffle_valid == shuffle_train == False
            ), "You can't shuffle in ddp or sorted mode. Set shuffle_train and shuffle_valid to 0 or use strategy='random'."
        self.shuffle = {
            "train": shuffle_train,
            "valid": shuffle_valid,
        }
        self.tqdm = tqdm
        self.dataset_args = {
            "structure_padding_value": structure_padding_value,
            "max_len": max_len,
            "use_error": use_error,
            "force_download": force_download,
            "tqdm": tqdm,
            "max_len": max_len,
            "min_len": min_len,
        }
        self.buckets = buckets

        # Log hyperparameters
        self.save_hyperparameters(ignore=["force_download"])

    def _use_multiple_datasets(self, name):
        if isinstance(name, str):
            return False
        elif isinstance(name, list) or isinstance(name, tuple):
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
        if stage is None or (
            stage in ["fit", "predict"] and not hasattr(self, "all_datasets")
        ):
            self.all_datasets = self._dataset_merge(
                [
                    Dataset.from_local_or_download(
                        name=name,
                        data_type=self.data_type,
                        sort_by_length=self.strategy == "sorted",
                        **self.dataset_args,
                    )
                    for name in self.name
                ]
            )
            self.collate_fn = self.all_datasets.collate_fn

        if stage == "fit":
            if self.splits["train"] == None or self.splits["train"] == 1.0:
                self.train_set = self.all_datasets
            else:
                num_datapoints = (
                    round(len(self.all_datasets) * self.splits["train"])
                    if isinstance(self.splits["train"], float)
                    else self.splits["train"]
                )
                assert num_datapoints > 0, "train_split must be greater than 0"
                assert num_datapoints <= len(
                    self.all_datasets
                ), "train_split must be less than the number of datapoints"
                self.train_set = Subset(
                    self.all_datasets,
                    range(0, num_datapoints),
                )
            if self.external_valid is not None:
                self.external_val_set = []
                for name in self.external_valid:
                    self.external_val_set.append(
                        Dataset.from_local_or_download(
                            name=name,
                            data_type=self.data_type,
                            sort_by_length=True,
                            **self.dataset_args,
                        )
                    )

        if stage == "test":
            self.test_sets = self._select_test_dataset()

        if stage == "predict":
            self.predict_set = Subset(
                self.all_datasets,
                range(
                    0,
                    round(len(self.all_datasets) * self.splits["predict"])
                    if isinstance(self.splits["predict"], float)
                    else self.splits["predict"],
                ),
            )

    def _select_test_dataset(self):
        return [
            Dataset.from_local_or_download(
                name=name,
                data_type=[data_type],
                **self.dataset_args,
            )
            for data_type, datasets in TEST_SETS.items() if data_type in self.data_type
            for name in datasets
        ]

    def train_dataloader(self):
        if self.strategy == "ddp":
            if self.trainer is None:
                raise ValueError(
                    "When using strategy='ddp', the trainer must be passed to the datamodule"
                )
            else: # ddp
                num_replicas = self.trainer.num_devices
                rank = self.trainer.local_rank
        else:
            num_replicas = 1
            rank = 0

        return DataLoader(
            self.train_set,
            shuffle=self.shuffle["train"],
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            to_device=self.strategy != "ddp",
            sampler=sampler_factory(
                dataset=self.train_set,
                strategy=self.strategy,
                num_replicas=num_replicas,
                seed=datetime.datetime.now().hour,
                rank=rank,
            ),
        )

    def val_dataloader(self):
        val_dls = []
        ###################################
        # Add validation set here if needed
        ###################################
        if self.external_valid is not None:
            for val_set in self.external_val_set:
                val_dls.append(
                    DataLoader(
                        val_set,
                        shuffle=self.shuffle["valid"],
                        collate_fn=self.collate_fn,
                        batch_size=self.batch_size,
                        to_device=self.strategy != "ddp",
                        sampler=sampler_factory(
                            dataset=val_set,
                            strategy=self.strategy,
                            num_replicas=self.trainer.num_devices,
                            seed=datetime.datetime.now().hour,
                            rank=self.trainer.local_rank,
                        ),
                    )
                )
        return val_dls

    def test_dataloader(self):
        return [
            DataLoader(
                test_set,
                num_workers=self.num_workers,
                collate_fn=test_set.collate_fn,
                batch_size=self.batch_size,
            )
            for test_set in self.test_sets
        ]

    def predict_dataloader(self):
        return DataLoader(
            self.predict_set,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            shuffle=False,
        )

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        pass
