from torch.utils.data import DataLoader, random_split, Subset
import lightning.pytorch as pl
from typing import Union, List
from .dataset import Dataset
from ..config import TEST_SETS, UKN
from .sampler import BySequenceLengthSampler

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
        external_valid=None,
        use_error=False,
        max_len=500,
        structure_padding_value=UKN,
        tqdm=True,
        buckets = None,
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
        """
        # Save arguments
        super().__init__(**kwargs)

        if not self._use_multiple_datasets(name):
            self.name = [name]
        else:
            self.name = name

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_type = data_type
        self.external_valid = external_valid
        self.splits = {
            "train": train_split,
            "valid": valid_split,
            "predict": predict_split,
        }
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
        }
        self.buckets = buckets  

        # we need to know the max sequence length for padding
        self.overfit_mode = overfit_mode

        # Log hyperparameters
        if hasattr(self, "size_sets"):
            train_split, valid_split, _ = self.size_sets
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
                        **self.dataset_args,
                    )
                    for name in self.name
                ]
            )
            self.collate_fn = self.all_datasets.collate_fn

        if stage == "fit":
            self.size_sets = _compute_size_sets(
                len(self.all_datasets),
                train_split=self.splits["train"],
                valid_split=self.splits["valid"],
            )
            self.train_set, self.val_set, _ = random_split(
                self.all_datasets, self.size_sets
            )
            if self.external_valid is not None:
                self.external_val_set = []
                for name in self.external_valid:
                    self.external_val_set.append(Dataset.from_local_or_download(
                        name=name,
                        data_type=self.data_type,
                        sort_by_length=True,
                        **self.dataset_args,
                    ))

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
            for data_type, datasets in TEST_SETS.items()
            for name in datasets
        ]
        
    def _get_sampler(self, dataset):
        if self.buckets is not None:
            return BySequenceLengthSampler(
                dataset,
                bucket_boundaries=self.buckets,
                batch_size=self.batch_size,
            )
        else:
            return None

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            shuffle=self.shuffle["train"],
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            sampler=self._get_sampler(self.train_set),
        )

    def val_dataloader(self):
        valid = DataLoader(
            self.val_set if not self.overfit_mode else self.train_set,
            shuffle=self.shuffle["valid"],
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
        )

        val_dls = [valid]
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
                    )
                )
        return val_dls

    def test_dataloader(self):
        return [
            DataLoader(
                test_set,
                collate_fn=test_set.collate_fn,
                batch_size=self.batch_size,
            )
            for test_set in self.test_sets
        ]

    def predict_dataloader(self):
        return DataLoader(
            self.predict_set,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            shuffle=False,
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

    if valid_split <= 1 and isinstance(valid_split, float):
        valid_split = int(valid_split * len_data)

    if train_split is None:
        train_split = len_data - valid_split

    elif train_split <= 1 and isinstance(train_split, float):
        train_split = len_data - int((1 - train_split) * len_data)

    assert (
        train_split + valid_split <= len_data
    ), "The sum of the splits must be less than the length of the dataset"

    return train_split, valid_split, len_data - train_split - valid_split
