from typing import Any
import torch
from torch import tensor
from tqdm import tqdm as tqdm_fun
import numpy as np
from ..config import UKN, DATA_TYPES
from torch import nn, tensor, float32, int64, stack, uint8
import numpy as np
import torch.nn.functional as F
from .datapoint import Datapoint, Data, Metadata, set_prefix
from .embeddings import base_pairs_to_pairing_matrix
from .listofdatapoints import ListOfDatapoints
import lightning.pytorch as pl
from ..config import device
from ..util import unzip


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


class Batch(pl.LightningDataModule):
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
        lengths = [dp.get("length") for dp in list_of_datapoints]
        L = max(lengths)
        for dt in data_type:
            index, values = [], []
            for idx, dp in enumerate(list_of_datapoints):
                signal = dp.get(dt)
                if signal is not None and len(signal) > 0:
                    index.append(idx)
                    values.append(_pad(signal, L, dt))
            if len(index):
                data[dt] = {
                    "index": tensor(index).to(device),
                    "values": stack(values).to(device),
                }

        metadata = {"length": lengths}
        for metadata_type in ["reference", "quality"]:
            metadata[metadata_type] = [
                dp.get(metadata_type) for dp in list_of_datapoints
            ]

        return cls(
            data=data,
            metadata=metadata,
            data_type=data_type,
            L=L,
            batch_size=len(list_of_datapoints),
        )

    def to_list_of_datapoints(self) -> ListOfDatapoints:
        list_of_datapoints = []
        for idx in range(len(self.metadata["reference"])):
            data = Data(sequence=self.get("sequence", index=idx))
            metadata = Metadata(
                reference=self.get("reference", index=idx),
                length=self.get("length", index=idx),
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
                    local_idx = torch.where(self.data[dt]["index"] == idx)[0].item()
                    setattr(
                        data,
                        dt,
                        self.data[dt]["values"][local_idx][
                            : metadata.length
                        ],  # trims the padded part!
                    )
            #############################################
            for metadata_type in Metadata.__annotations__.keys():
                setattr(metadata, metadata_type, self.metadata[metadata_type][idx])
            list_of_datapoints.append(
                Datapoint(data=data, metadata=metadata, prediction=prediction)
            )
        return ListOfDatapoints(list_of_datapoints)

    def integrate_prediction(self, prediction):
        assert len(prediction[list(prediction.keys())[0]]) == len(
            self.data["sequence"]["index"]
        ), "outputs and batch must have the same length"
        self.prediction = prediction
        
    def get_pairs(self, data_type):
        if not data_type in DATA_TYPES:
            raise ValueError("data_type must be either dms or shape")
        if not self.contains(data_type):
            return None
        pred = self.prediction[data_type][self.data[data_type]["index"]]
        true = self.data[data_type]["values"]
        return pred, true
    
    def count(self, data_type):
        return len(self.data[data_type]["index"])

    @unzip
    def get(self, data_type, pred=False, true=True, index=None):
        prefix = set_prefix(data_type)
        out = {}
        if pred and self.prediction is None and prefix == "data":
            raise ValueError("No prediction available")
        if pred and prefix == "data":
            out["pred"] = self.prediction[data_type][index]
        if true:
            if prefix == "data":
                out["true"] = self.data[data_type]["values"]
            else:
                out["true"] = self.metadata[data_type]
        if index != None:
            if prefix == "data":
                return [
                    out[v][: self.get("length", index=index)].squeeze()
                    for v in ["pred", "true"]
                    if v in out.keys()
                ]
            else:
                return out["true"][index]
        if prefix == "data":
            return [out[v] for v in ["pred", "true"] if v in out.keys()]
        else:
            return out["true"]

    def get_index(self, data_type):
        return self.data[data_type]["index"]

    def contains(self, data_type):
        if self.prediction is not None and not data_type in self.prediction.keys():
            return False
        return data_type in self.data.keys() and len(self.data[data_type]["index"]) > 0
