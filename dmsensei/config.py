from torch import nn, tensor, float32, cuda, backends
import torch
import numpy as np


DEFAULT_FORMAT = float32
torch.set_default_dtype(DEFAULT_FORMAT)
UKN = -1000.0
VAL_GU = 0.095
device = (
    "cuda"
    if cuda.is_available()
    else "mps"  # moi j'aime bien le mps
    if backends.mps.is_available()
    else "cpu"
)

TEST_SETS = {
    "structure": [],#["CT_files_pdbee"], #TODO #16 add structure test sets
    "sequence": [],
    "dms": [ "sarah_supermodel", "utr", "SARS2", "pri-miRNA"],
}

TEST_SETS_NAMES = [i for j in TEST_SETS.values() for i in j]

DATA_TYPES = ["structure", "dms", "shape"]
REFERENCE_METRIC = {"structure": "f1", "dms": "mae", "shape": "mae"}
POSSIBLE_METRICS = {
    "structure": ["f1", "mFMI"],
    "dms": ["mae", "r2", "pearson"],
    "shape": ["mae", "r2", "pearson"],
}
REF_METRIC_SIGN = {"structure": 1, "dms": -1, "shape": -1}


torch.set_default_dtype(torch.float32)
