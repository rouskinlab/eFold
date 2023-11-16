from typing import Any
import torch
from torch import tensor
from tqdm import tqdm as tqdm_fun
import numpy as np
from ..config import UKN, DATA_TYPES
from torch import nn, tensor, float32, int64, stack, uint8
import numpy as np
from numpy import array, ndarray
from .embeddings import (
    base_pairs_to_int_dot_bracket,
    sequence_to_int,
    base_pairs_to_pairing_matrix,
)
import torch.nn.functional as F


class Metadata:
    def __init__(self, reference, index=None, quality=1.0):
        self.reference = reference
        self.index = index
        self.quality = quality
        
class Data:
    def __init__(self, sequence, dms=None, shape=None, structure=None):
        self.sequence = sequence
        self.dms = dms
        self.shape = shape
        self.structure = structure

class Datapoint:
    def __init__(self, data:Data, metadata:Metadata):
        self.data = data
        self.metadata = metadata
        
    @classmethod
    def from_attributes(cls, sequence, reference, dms=None, shape=None, structure=None, index=None, quality=1.0):
        return cls(
            data=Data(sequence, dms, shape, structure),
            metadata=Metadata(reference, index, quality)
        )
        
        
def get_array(data, attr, index, astype=None):
    if not attr in data:
        return None
    d = tensor(data[attr][index].astype(astype))
    if not torch.isnan(d).all() and not (d==UKN).all():
        return d


class ListOfDatapoints:
    
    def __init__(self, list_of_datapoints=None) -> None:
        self.list_of_datapoints = list_of_datapoints if list_of_datapoints is not None else []
        
    def __add__(self, other):
        return ListOfDatapoints(self.list_of_datapoints + other.list_of_datapoints)
        
    def __len__(self):  
        return len(self.list_of_datapoints)
    
    def __getitem__(self, index):
        return self.list_of_datapoints[index]
    
    def __call__(self):
        return self.list_of_datapoints
    
    def tolist(self):
        return self.list_of_datapoints
    
    @classmethod
    def from_rouskinhf(cls, data, data_type, name=None, tqdm=True):
        # TODO #5 group the sequences by length
        self = cls()
        self.data_type = data_type
        #TODO #14 add  quality score per dataset or datapoint
        
        for index in tqdm_fun(
            range(len(data['references'])),
            desc="Wrangling data for {}".format(name),
            total=len(self),
            disable=not tqdm,
            colour="green",
        ):

            self.list_of_datapoints.append(
                Datapoint(
                    data=Data(
                        sequence=get_array(data, "sequences", index, astype=np.int32),
                        dms=get_array(data, "dms", index, astype=np.float32),
                        shape=get_array(data, "shape", index, astype=np.float32),
                        structure=get_array(data, "structures", index, astype=np.int32),
                    ),
                    metadata=Metadata(
                        reference=data["references"][index],
                        index=index,
                        quality=1.0,
                    )
                ))

        return self

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

class Batch:
    """Batch class to handle padding and stacking of tensors"""
    
    def __init__(self, data, metadata, data_type):
        self.data = data
        self.metadata = metadata
        self.data_type = data_type

    @classmethod
    def from_list_of_datapoints(cls, list_of_datapoints, data_type, padding=True):
        
        data_out, metadata_out = {}, {}
        lengths = [len(dp.data.sequence) for dp in list_of_datapoints]
        for dt in data_type:
            padded_data = [
                (idx, _pad(getattr(datapoint.data, dt), max(lengths), dt))
                if padding
                else (idx, datapoint[data_type])
                for idx, datapoint in enumerate(list_of_datapoints)
                if getattr(datapoint.data, dt) is not None
            ]
            if len(padded_data) == 0:
                continue
            index, values = zip(*padded_data)
            data_out[dt] = {"index": tensor(index), "values": stack(values)}

        metadata_out = {"length": lengths}
        for metadata_type in Metadata.__annotations__.keys():
            metadata_out[metadata_type] = tensor([getattr(dp.metadata, metadata_type) for dp in list_of_datapoints])

        return cls(
            data=data_out,
            metadata=metadata_out,
            data_type=data_type,
        )
        
    def __len__(self):
        return len(self.list_of_datapoints)
    
    def __getitem__(self, index):
        return self.list_of_datapoints[index]
    
    def to_list_of_datapoints(self):
        pass
    