import torch
import torch.nn.functional as F

from efold.constants import config
from efold.core import embeddings


def _pad(arr: torch.Tensor, L: int, data_type: str) -> torch.Tensor:
    padding_values = {
        "sequence": 0,
        "dms": config.pytorch.unknown_value,
        "shape": config.pytorch.unknown_value,
    }
    if data_type == "structure":
        return embeddings.base_pairs_to_pairing_matrix(arr, L)
    else:
        if isinstance(arr, list):
            arr = torch.tensor(arr)
        return F.pad(arr, (0, L - arr.shape[1]), value=padding_values[data_type])


def split_data_type(data_type: str) -> tuple[str, str]:
    if "_" not in data_type:
        data_part = "true"
    else:
        data_part, data_type = data_type.split("_")
    return data_part, data_type
