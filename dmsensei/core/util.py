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
        if type(arr) == list:
            arr = torch.tensor(arr)
        return F.pad(arr, (0, L - len(arr)), value=padding_values[data_type])
