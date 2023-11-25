from typing import Any
import torch
from torch import tensor
import numpy as np
from ..config import UKN, DATA_TYPES
from torch import nn, tensor, float32, int64, stack, uint8
import numpy as np
import torch.nn.functional as F
from .datapoint import Datapoint
from .embeddings import base_pairs_to_pairing_matrix
import lightning.pytorch as pl
from ..config import device, POSSIBLE_METRICS
from .metrics import metric_factory
from typing import Dict
from .embeddings import sequence_to_int
from .metrics import metric_factory
from ..config import POSSIBLE_METRICS
from .datapoint import Datapoint, data_type_factory, split_data_type


def _pad(arr, L, data_type):
    padding_values = {
        "sequence": 0,
        "dms": UKN,
        "shape": UKN,
    }
    if data_type == "structure":
        return base_pairs_to_pairing_matrix(arr, L)
    else:
        return F.pad(arr, (0, L - len(arr)), value=padding_values[data_type])


from dmsensei.config import UKN, DATA_TYPES
from dmsensei.core.embeddings import base_pairs_to_pairing_matrix
import torch.nn.functional as F
from torch import tensor


def _pad(arr, L, data_type):
    padding_values = {
        "sequence": 0,
        "dms": UKN,
        "shape": UKN,
    }
    if data_type == "structure":
        return base_pairs_to_pairing_matrix(arr, L)
    else:
        return F.pad(arr, (0, L - len(arr)), value=padding_values[data_type])


def get_padded_vector(dp, data_type, data_part, L):
    if getattr(dp, data_type) is None:
        return None
    if getattr(getattr(dp, data_type), data_part) is None:
        return None
    return _pad(getattr(getattr(dp, data_type), data_part), L, data_type)


class Batch:
    def __init__(
        self,
        reference,
        sequence,
        L,
        batch_size,
        data_types,
        dms=None,
        shape=None,
        structure=None,
    ):
        self.reference = reference
        self.sequence = sequence
        self.length = len(sequence)
        self.dms = dms
        self.shape = shape
        self.structure = structure
        self.L = L
        self.batch_size = batch_size
        self.data_types = data_types

    def _stack_data_type(self, list_of_datapoints, data_type, L):
        index, true, error, quality, pred = [], [], [], [], []
        for idx, dp in enumerate(list_of_datapoints):
            if getattr(dp, data_type) is not None:
                index.append(idx)
                true.append(get_padded_vector(dp, data_type, "true", L))
                if (err := get_padded_vector(dp, data_type, "error", L)) is not None:
                    error.append(err)
                if (qual := get_padded_vector(dp, data_type, "quality", L)) is not None:
                    quality.append(qual)
                if (pr := get_padded_vector(dp, data_type, "pred", L)) is not None:
                    pred.append(pr)

        if len(index) == 0:
            return None

        def post_process(x):
            if len(x) == 0:
                return None
            return torch.stack(x)

        return {
            "index": index,
            "true": post_process(true),
            "error": post_process(error),
            "quality": post_process(quality),
            "pred": post_process(pred),
        }

    @classmethod
    def from_list_of_datapoints(cls, datapoints: list, data_types: list):
        reference = [dp.reference for dp in datapoints]
        L = max([dp.length for dp in datapoints])
        sequence = torch.stack([_pad(dp.sequence, L, "sequence") for dp in datapoints])
        batch_size = len(datapoints)

        data = {}
        for data_type in DATA_TYPES:
            vector = cls._stack_data_type(None, datapoints, data_type, L)
            if vector is not None:
                data[data_type] = data_type_factory[data_type](**vector)

        return cls(
            reference,
            sequence,
            L=L,
            batch_size=batch_size,
            data_types=data_types,
            **data,
        )

    def get(self, data_type, to_numpy=False):
        if data_type in ["reference", "sequence", "length"]:
            return getattr(self, data_type)
        data_part, data_type = split_data_type(data_type)

        if not hasattr(self, data_type):
            raise ValueError(f"This batch doesn't contain data type {data_type}.")
        out = getattr(getattr(self, data_type), data_part)

        if to_numpy:
            if hasattr(out, "cpu"):
                out = out.squeeze().cpu().numpy()
        return out

    def to_list_of_datapoints(self) -> list:
        pass

    def integrate_prediction(self, prediction):
        assert len(prediction[list(prediction.keys())[0]]) == len(
            self.get("sequence")
        ), "outputs and batch must have the same length"
        self.prediction = prediction

    def get_pairs(self, data_type):
        if not data_type in DATA_TYPES:
            raise ValueError("data_type must be either dms or shape")
        if not self.contains(data_type):
            return None
        pred = self.prediction[data_type][self.data[data_type].index]
        true = self.data[data_type].values
        return pred, true

    def count(self, data_type):
        pass

    def get_index(self, data_type):
        pass

    def contains(self, data_type):
        data_part, data_type = split_data_type(data_type)
        if (
            not hasattr(self, data_type)
            or getattr(self, data_type) is None
            or not data_type in self.data_types
        ):
            return False
        if (
            not hasattr(getattr(self, data_type), data_part)
            or getattr(getattr(self, data_type), data_part) is None
        ):
            return False
        return True

    def __len__(self):
        return self.count("sequence")

    def compute_metrics(self) -> Dict[str, Dict[str, float]]:
        out = {}
        for data_type in self.data_types:
            if not self.count(data_type) or data_type == "sequence":
                continue
            out[data_type] = {}
            pred, true = self.get_pairs(data_type)
            for metric in POSSIBLE_METRICS[data_type]:
                out[data_type][metric] = metric_factory[metric](
                    pred=pred, true=true, batch=self.count(data_type) > 1
                )
        return out

    def get_weights_as_matrix(self, data_type, L):
        """Returns the weights as a matrix of shape (batch_size*, L)
        where batch_size* is the number of datapoints of this data_type in the batch
        and L is the length of the longest sequence in the batch
        """
        return (
            self.get("quality_{}".format(data_type))[self.get_index(data_type)]
            .unsqueeze(-1)
            .repeat(1, L)
        )
