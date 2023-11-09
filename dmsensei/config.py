from torch import nn, tensor, float32, cuda, backends
import torch

DEFAULT_FORMAT = float32
UKN = -1000.0
VAL_G = 0.095
VAL_U = 0.095
device = (
    "cuda"
    if cuda.is_available()
    #   else "mps"
    #  if backends.mps.is_available()
    else "cpu"
)

TEST_SETS_NAMES = {
    "structure": ["CT_files_pdbee"],
    'sequence': [],
    "dms": ["sarah_supermodel", "utr", "SARS2", "pri-miRNA"],
}


torch.set_default_dtype(torch.float32)
