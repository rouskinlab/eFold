from torch import nn, tensor, float32, cuda, backends
import torch

DEFAULT_FORMAT = float32
UKN = -1000.0

device = (
    "cuda"
    if cuda.is_available()
    #   else "mps"
    #  if backends.mps.is_available()
    else "cpu"
)

TEST_SETS_NAMES = {
    "structure": ["CT_files_pdbee"],
    "dms": ["sarah_supermodel", "utr", "SARS2"],
}


torch.set_default_dtype(torch.float32)
