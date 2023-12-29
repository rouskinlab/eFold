from torch.utils.data import Sampler, Subset
import numpy as np
from random import shuffle
from torch.utils.data import Dataset
from typing import Union

class NoShuffleSampler(Sampler):
    def __init__(self, dataset: Union[Dataset, Subset]):
        self.data_source = dataset

    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)
    

class SeedBasedSampler(Sampler):
    def __init__(self, dataset: Union[Dataset, Subset], seed):
        self.data_source = dataset
        self.seed = seed
        self.sorted_indices = np.argsort(np.array(dataset.dataset.length)[dataset.indices] if isinstance(dataset, Subset) else dataset.length)

    def __iter__(self):
        np.random.seed(self.seed)
        return iter(np.random.permutation(self.sorted_indices))

    def __len__(self):
        return len(self.data_source)
    

class BySequenceLengthSampler(Sampler):
    def __init__(
        self,
        dataset: Union[Dataset, Subset],
        bucket_boundaries,
        batch_size,
    ):
        ind_n_len = []
        length = dataset.dataset.length if isinstance(dataset, Subset) else dataset.length
        for i, p in enumerate(length):
            ind_n_len.append((i, p))
        self.ind_n_len = ind_n_len
        self.bucket_boundaries = bucket_boundaries
        self.batch_size = batch_size
        self.data_source = dataset
        
        # filter out bucket boundaries that are larger than the largest sequence / smaller than the smallest sequence
        self.bucket_boundaries = [
            b for b in self.bucket_boundaries if b < max(length) and b > min(length)
        ]

    def __iter__(self):
        data_buckets = dict()
        # where p is the id number and seq_len is the length of this id number.
        for p, seq_len in self.ind_n_len:
            pid = self.element_to_bucket_id(p, seq_len)
            if pid in data_buckets.keys():
                data_buckets[pid].append(p)
            else:
                data_buckets[pid] = [p]

        for k in data_buckets.keys():
            data_buckets[k] = np.asarray(data_buckets[k])

        iter_list = []
        for k in data_buckets.keys():
            np.random.shuffle(data_buckets[k])
            n_batch = int(len(data_buckets[k]) / self.batch_size)
            if n_batch == 0 and len(data_buckets[k]) > 0:
                n_batch = 1
            iter_list += np.array_split(
                data_buckets[k], n_batch
            )
        shuffle(iter_list)  # shuffle all the batches so they arent ordered by bucket
        
        for l in iter_list:
            for i in l:
                yield i 

    def __len__(self):
        return len(self.data_source)

    def element_to_bucket_id(self, x, seq_length):
        boundaries = list(self.bucket_boundaries)
        buckets_min = [np.iinfo(np.int32).min] + boundaries
        buckets_max = boundaries + [np.iinfo(np.int32).max]
        conditions_c = np.logical_and(
            np.less_equal(buckets_min, seq_length), np.less(seq_length, buckets_max)
        )
        bucket_id = np.min(np.where(conditions_c))
        return bucket_id


def sampler_factory(
    dataset: Union[Dataset, Subset],
    shuffle: str,
    seed: int = None,
    bucket_boundaries = None,
    batch_size: int = None,
):
    if shuffle == 'random':
        return None
    elif shuffle == 'bucket':
        return BySequenceLengthSampler(dataset, bucket_boundaries, batch_size)
    elif shuffle == 'seed':
        return SeedBasedSampler(dataset, seed)
    else:
        raise ValueError(f"Invalid shuffle value: {shuffle}")