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
from ..util import unzip

def set_prefix(data_type):
    if data_type in ["length", "reference", "quality_dms", "quality_shape", "quality_structure"]:
        prefix = "metadata"
    elif data_type in ['sequence', 'dms', 'shape', 'structure']:
        prefix = "data"
    else: 
        raise ValueError("data_type: {} must be in {}".format(data_type, ['sequence', 'dms', 'shape', 'structure']))
    return prefix

class Metadata:
    def __init__(self, reference, length, index=None, quality_dms=1., quality_shape=1., quality_structure=1.):
        self.reference = reference
        self.length = length
        self.index = index
        self.quality_dms = quality_dms
        self.quality_shape = quality_shape
        self.quality_structure = quality_structure


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
        quality_dms=1.,
        quality_shape=1.,
        quality_structure=1.,
    ):
        return cls(
            data=Data(sequence, dms, shape, structure),
            metadata=Metadata(reference, index, quality_dms, quality_shape, quality_structure),
        )

    def compute_error_metrics_pack(self):
        self.metrics = {}
        for data_type in DATA_TYPES:
            if not (hasattr(self.data, data_type) and hasattr(self.prediction, data_type)):
                continue
            pred = self.get(data_type, pred=True, true=False)
            true = self.get(data_type, pred=False, true=True)
            if not (true is not None and pred is not None):
                continue
            # true, pred = true.squeeze(), pred.squeeze()
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
        if self.prediction is not None and not hasattr(self.prediction, data_type):
            return False
        return hasattr(self.data, data_type)

    def get(self, data_type, pred=False, true=True, to_numpy=False):
        prefix = set_prefix(data_type)

        if pred and self.prediction is None and prefix == "data":
            raise ValueError("No prediction available")
        
        # return metadata
        if prefix == "metadata":
            return getattr(self.metadata, data_type)
        
        # now we are in the data part
        out = []
        if pred:
            out.append(getattr(self.prediction, data_type))
        if true:
            out.append(getattr(self.data, data_type))
        if to_numpy:
            for idx, arr in enumerate(out):
                if hasattr(arr, "cpu"):
                    out[idx] = arr.squeeze().cpu().numpy()
        return out[0] if len(out) == 1 else out
        