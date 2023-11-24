from typing import Any
import torch
from torch import tensor
import numpy as np
from ..config import UKN, DATA_TYPES
from torch import nn, tensor, float32, int64, stack, uint8
import numpy as np
import torch.nn.functional as F
from .datapoint import Datapoint, Data, Metadata, set_prefix
from .embeddings import base_pairs_to_pairing_matrix
from .listofdatapoints import ListOfDatapoints
import lightning.pytorch as pl
from ..config import device, POSSIBLE_METRICS
from .metrics import metric_factory
from typing import Dict

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


class BatchData:
    def __init__(self, data_type: str, index: tensor, values: tensor):
        self.data_type = data_type
        self.index = index
        self.values = values


class BatchMetadata:
    def __init__(self, reference: list, length: list, quality_dms: list, quality_shape: list, quality_structure: list):
        self.reference = reference
        self.length = length
        self.quality_dms = quality_dms
        self.quality_shape = quality_shape
        self.quality_structure = quality_structure


class Batch:
    """Batch class to handle padding and stacking of tensors"""

    def __init__(self, data, metadata, data_type, L, batch_size):
        """Batch class to handle padding and stacking of tensors

        Format of data and metadata:
            data is a dict of tensors:
            data = {
                "sequence": {"index": tensor([0, 1, 2]), "values": tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])},
                "dms": {"index": tensor([0, 1, 2]), "values": tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])},
                "shape": {"index": tensor([0, 1, 2]), "values": tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])},
            }
            metadata is a dict of tensors:
            metadata = {
                "length": tensor([3, 3, 3]),
                "reference": tensor([0, 1, 2]),
        }
        """

        self.data = data
        self.metadata = metadata
        self.data_type = data_type
        self.prediction = None
        self.L = L
        self.batch_size = batch_size

    @classmethod
    def from_list_of_datapoints(cls, list_of_datapoints, data_type):
        data, metadata = {}, {}
        lengths = np.array([dp.get("length") for dp in list_of_datapoints])
        L = max(lengths)
        for dt in data_type:
            index, values = [], []
            for idx, dp in enumerate(list_of_datapoints):
                signal = dp.get(dt)
                if signal is not None and len(signal) > 0:
                    index.append(idx)
                    values.append(_pad(signal, L, dt))
            if len(index):
                data[dt] = BatchData(
                    data_type=dt,
                    index=tensor(index).to(device),
                    values=stack(values).to(device),
                )
                
        def parse_data(data_type, dtype=tensor):
            return dtype([dp.get(data_type) for dp in list_of_datapoints])

        metadata = BatchMetadata(
            length=lengths,
            reference=parse_data("reference", dtype=list),
            quality_dms=parse_data("quality_dms").to(device),
            quality_shape=parse_data("quality_shape").to(device),
            quality_structure=parse_data("quality_structure").to(device),
        )

        return cls(
            data=data,
            metadata=metadata,
            data_type=data_type,
            L=L,
            batch_size=len(list_of_datapoints),
        )

    def to_list_of_datapoints(self) -> ListOfDatapoints:
        list_of_datapoints = []
        for idx in range(len(self.metadata.reference)):
            data = Data(sequence=self.get("sequence", index=idx))
            metadata = Metadata(
                reference=self.get("reference", index=idx),
                length=self.get("length", index=idx),
                quality_dms=self.get("quality_dms", index=idx),
                quality_shape=self.get("quality_shape", index=idx),
                quality_structure=self.get("quality_structure", index=idx),
            )
            ######## This part is the tricky one ########
            prediction = (
                Data(
                    sequence=self.get("sequence", index=idx),
                    **{
                        dt: self.get(dt, pred=True, true=False, index=idx)[
                            : metadata.length
                        ]
                        for dt in self.prediction.keys()
                    },
                )
                if self.prediction is not None
                else None
            )
            for dt in self.data_type:
                if dt in self.data and idx in self.get_index(dt):
                    local_idx = torch.where(self.data[dt].index == idx)[0].item()
                    setattr(
                        data,
                        dt,
                        self.data[dt].values[local_idx][
                            : metadata.length
                        ],  # trims the padded part!
                    )
            #############################################
            for metadata_type in Metadata.__annotations__.keys():
                setattr(
                    metadata, metadata_type, getattr(self.metadata, metadata_type)[idx]
                )
            list_of_datapoints.append(
                Datapoint(data=data, metadata=metadata, prediction=prediction)
            )
        return ListOfDatapoints(list_of_datapoints)

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
        if not data_type in self.data:
            return 0
        return len(self.data[data_type].index)

    def get(self, data_type, pred=False, true=True, index=None):
        prefix = set_prefix(data_type)
        if pred and self.prediction is None and prefix == "data":
            raise ValueError("No prediction available")

        # return metadata
        if prefix == "metadata":
            if index != None:
                return getattr(self.metadata, data_type)[index]
            return getattr(self.metadata, data_type)

        # now we know we are dealing with data
        out = []
        if pred:
            out.append(self.prediction[data_type])
        if true:
            out.append(self.data[data_type].values)
        if index is not None:
            out = [v[index][: self.get("length", index=index)] for v in out]
        return out[0] if len(out) == 1 else out

    def get_index(self, data_type):
        return self.data[data_type].index

    def contains(self, data_type):
        if self.prediction is not None and not data_type in self.prediction.keys():
            return False
        return data_type in self.data.keys() and len(self.data[data_type].index) > 0

    def __len__(self):
        return self.count("sequence")

    def compute_metrics(self)->Dict[str, Dict[str, float]]:
        out = {}
        for data_type in self.data_type:
            if not self.count(data_type) or data_type == "sequence":
                continue
            out[data_type] = {}
            pred, true = self.get_pairs(data_type)
            for metric in POSSIBLE_METRICS[data_type]:
                out[data_type][metric] = metric_factory[metric](
                    pred=pred, true=true, batch= self.count(data_type) > 1
                )
        return out
    
    def get_weights_as_matrix(self, data_type, L):
        """Returns the weights as a matrix of shape (batch_size*, L)
        where batch_size* is the number of datapoints of this data_type in the batch
        and L is the length of the longest sequence in the batch
        """
        return self.get('quality_{}'.format(data_type))[self.get_index(data_type)].unsqueeze(-1).repeat(1, L)