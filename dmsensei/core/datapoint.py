from typing import Any
import torch
from torch import tensor
from tqdm import tqdm as tqdm_fun
import numpy as np
from ..config import UKN, DATA_TYPES, POSSIBLE_METRICS, REFERENCE_METRIC
from torch import nn, tensor, float32, int64, stack, uint8
import numpy as np
from numpy import array, ndarray
from .embeddings import (
    base_pairs_to_int_dot_bracket,
    sequence_to_int,
    base_pairs_to_pairing_matrix,
)
import torch.nn.functional as F
from .metrics import metric_factory


class Metadata:
    def __init__(self, reference, length, index=None, quality=1.0):
        self.reference = reference
        self.length = length
        self.index = index
        self.quality = quality


class Data:
    def __init__(self, sequence, dms=None, shape=None, structure=None):
        self.sequence = sequence
        self.dms = dms
        self.shape = shape
        self.structure = structure


class Datapoint:
    def __init__(self, data: Data, metadata: Metadata, prediction=None):
        self.data = data
        self.metadata = metadata
        self.prediction = prediction

    @classmethod
    def from_attributes(
        cls,
        sequence,
        reference,
        dms=None,
        shape=None,
        structure=None,
        index=None,
        quality=1.0,
    ):
        return cls(
            data=Data(sequence, dms, shape, structure),
            metadata=Metadata(reference, index, quality),
        )

    def compute_error_metrics_pack(self):
        self.metrics = {}
        for data_type in DATA_TYPES:
            if not (hasattr(self.data, data_type) and hasattr(self.prediction, data_type)):
                continue
            true = getattr(self.data, data_type)
            pred = getattr(self.prediction, data_type)
            if not (true is not None and pred is not None):
                continue
            true, pred = true.squeeze(), pred.squeeze()
            self.metrics[data_type] = {}
            for metric_name in POSSIBLE_METRICS[data_type]:
                self.metrics[data_type][metric_name] = metric_factory[
                    metric_name
                ](true=true, pred=pred, batch=False)
        return self.metrics

    def read_reference_metric(self, data_type):
        if not data_type in self.metrics:
            return np.nan
        return self.metrics[data_type][REFERENCE_METRIC[data_type]]

    def contains(self, data_type):
        return hasattr(self.data, data_type) and hasattr(self.prediction, data_type)
