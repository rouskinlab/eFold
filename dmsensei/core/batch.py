from typing import Any
import torch
from torch import tensor
from tqdm import tqdm as tqdm_fun
import numpy as np
from ..config import UKN, DATA_TYPES
from torch import nn, tensor, float32, int64, stack, uint8
import numpy as np
import torch.nn.functional as F
from .datapoint import Datapoint, Data, Metadata
from .embeddings import base_pairs_to_pairing_matrix
from .listofdatapoints import ListOfDatapoints
import lightning.pytorch as pl
from ..config import device


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

    def __init__(self, data, metadata, data_type, L):
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

    @classmethod
    def from_list_of_datapoints(cls, list_of_datapoints, data_type, padding=True):
        data_out, metadata_out = {}, {}
        lengths = [len(dp.data.sequence) for dp in list_of_datapoints]
        L = max(lengths)
        for dt in data_type:
            padded_data = [
                (idx, _pad(getattr(datapoint.data, dt), L, dt))
                if padding
                else (idx, datapoint[data_type])
                for idx, datapoint in enumerate(list_of_datapoints)
                if getattr(datapoint.data, dt) is not None
            ]
            if len(padded_data) == 0:
                continue
            index, values = zip(*padded_data)
            data_out[dt] = {
                "index": tensor(index).to(device),
                "values": stack(values).to(device),
            }

        metadata_out = {"length": lengths}
        for metadata_type in ["reference", "quality"]:
            metadata_out[metadata_type] = [
                getattr(dp.metadata, metadata_type) for dp in list_of_datapoints
            ]

        return cls(
            data=data_out,
            metadata=metadata_out,
            data_type=data_type,
            L=L,
        )

    def to_list_of_datapoints(self):
        list_of_datapoints = []
        for idx in range(len(self.metadata["reference"])):
            data = Data(sequence=self.get_value("sequence", idx))
            metadata = Metadata(
                reference=self.get_value("reference", idx),
                length=self.metadata["length"][idx],
            )
            prediction = (
                Data(sequence=self.get_value("sequence", idx))
                if self.prediction is not None
                else None
            )
            for dt in self.data_type:
                if dt in self.data and idx in self.data[dt]["index"]:
                    ######## This part is the tricky one ########
                    local_idx = torch.where(self.data[dt]["index"] == idx)[0].item()
                    setattr(
                        data, dt, self.data[dt]["values"][local_idx][: metadata.length]
                    )
                    if prediction is not None and dt in self.prediction:
                        setattr(
                            prediction, dt, self.prediction[dt][idx][: metadata.length]
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

    def get_value(self, data_type, index, return_prediction=True):
        if data_type in ["sequence", "length", "reference", "quality"]:
            return_prediction = False
        if return_prediction:
            pred, true = self.get_values(data_type, return_prediction=return_prediction)
            return pred[index], true[index]
        else:
            return self.get_values(data_type, return_prediction=return_prediction)[
                index
            ]

    def get_values(self, data_type, return_prediction=True):
        if return_prediction and self.prediction is None:
            raise ValueError("No prediction available")
        if data_type in ["sequence", "length", "reference", "quality"]:
            return_prediction = False
        if data_type in self.data.keys():
            prefix = "data"
        elif data_type in self.metadata.keys():
            prefix = "metadata"
            return_prediction = False
        else:
            raise ValueError("data_type must be in data or metadata")

        if not return_prediction:
            return (
                getattr(self, prefix)[data_type]["values"]
                if prefix == "data"
                else getattr(self, prefix)[data_type]
            )
        if prefix == "data":
            return (
                self.prediction[data_type][self.data[data_type]["index"]],
                self.data[data_type]["values"],
            )
        return self.prediction[data_type], getattr(self, prefix)[data_type]

    def get_index(self, data_type):
        return self.data[data_type]["index"]

    def contains(self, data_type):
        if self.prediction is not None and not data_type in self.prediction.keys():
            return False
        return data_type in self.data.keys() and len(self.data[data_type]["index"]) > 0
