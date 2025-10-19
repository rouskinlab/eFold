from typing import Dict
import torch
from torch import float32, cuda, backends


seq2int: Dict[str, int] = {"X": 0, "A": 1, "C": 2, "G": 3, "U": 4}
int2seq: Dict[int, str] = {v: k for k, v in seq2int.items()}

START_TOKEN: None = None
END_TOKEN: None = None
PADDING_TOKEN: int = seq2int["X"]

DEFAULT_FORMAT = float32
torch.set_default_dtype(DEFAULT_FORMAT)

UKN: float = -1000.0
VAL_GU: float = 0.095

device: str = (
    "cuda"
    if cuda.is_available()
    else "mps"
    if backends.mps.is_available()
    else "cpu"
)

torch.set_default_dtype(torch.float32)

