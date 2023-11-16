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
import copy


def get_array(data, attr, index, astype=None):
    if not attr in data:
        return None
    d = tensor(data[attr][index].astype(astype))
    if not torch.isnan(d).all() and not (d == UKN).all():
        return d


class ListOfDatapoints:
    def __init__(self, list_of_datapoints=None) -> None:
        self.list_of_datapoints = (
            list_of_datapoints if list_of_datapoints is not None else []
        )

    def __add__(self, other):
        if isinstance(other, Datapoint):
            return ListOfDatapoints(self.list_of_datapoints + [other])
        elif isinstance(other, ListOfDatapoints):
            return ListOfDatapoints(self.list_of_datapoints + other.list_of_datapoints)
        elif isinstance(other, list):
            return ListOfDatapoints(self.list_of_datapoints + other)
        else:
            raise ValueError("Cannot add {} to ListOfDatapoints".format(type(other)))

    def __len__(self):
        return len(self.list_of_datapoints)

    def __getitem__(self, index):
        return self.list_of_datapoints[index]

    def __call__(self):
        return self.list_of_datapoints

    def __iter__(self):
        return iter(self.list_of_datapoints)

    def tolist(self):
        return self.list_of_datapoints

    def copy(self):
        return copy.deepcopy(self)

    def sort(self, key=None, reverse=False):
        self.list_of_datapoints.sort(key=key, reverse=reverse)
        return self

    @classmethod
    def from_batch(cls, batch, prediction=None):
        if prediction is not None:
            batch.integrate_prediction(prediction)
        return batch.to_list_of_datapoints()

    @classmethod
    def from_rouskinhf(cls, data, data_type, name=None, tqdm=True):
        # TODO #5 group the sequences by length
        self = cls()
        self.data_type = data_type
        # TODO #14 add  quality score per dataset or datapoint

        for index in tqdm_fun(
            range(len(data["references"])),
            desc="Wrangling data for {}".format(name),
            total=len(self),
            disable=not tqdm,
            colour="green",
        ):
            d = Data(
                sequence=get_array(data, "sequences", index, astype=np.int32),
                dms=get_array(data, "dms", index, astype=np.float32),
                shape=get_array(data, "shape", index, astype=np.float32),
                structure=get_array(data, "structures", index, astype=np.int32),
            )
            md = Metadata(
                reference=data["references"][index],
                length=len(d.sequence),
                index=index,
                quality=1.0,
            )
            self.list_of_datapoints.append(
                Datapoint(
                    data=d,
                    metadata=md,
                )
            )

        return self
