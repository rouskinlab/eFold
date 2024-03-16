from ..config import UKN
from .embeddings import base_pairs_to_pairing_matrix
import torch.nn.functional as F
from torch import tensor
import torch


def _pad(arr, L, data_type):
    padding_values = {
        "sequence": 0,
        "dms": UKN,
        "shape": UKN,
    }
    if data_type == "structure":
        return base_pairs_to_pairing_matrix(arr, L)
    else:
        if isinstance(arr, list):
            arr = torch.tensor(arr)
        return F.pad(arr, (0, L - arr.shape[1]), value=padding_values[data_type])


def split_data_type(data_type):
    if "_" not in data_type:
        data_part = "true"
    else:
        data_part, data_type = data_type.split("_")
    return data_part, data_type
