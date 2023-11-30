from torch import float32, cuda, backends
import torch


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
    "dms": ["sarah_supermodel"],#"utr", "SARS2", "pri-miRNA"],
    "shape": []#'ribonanza_LQ']
}


TEST_SETS_NAMES = [i for j in TEST_SETS.values() for i in j]
DATA_TYPES_TEST_SETS = [k for k, v in TEST_SETS.items() for i in v]

DATA_TYPES = ["structure", "dms", "shape"]
DATA_TYPES_FORMAT = {
    "structure": torch.int32,
    "dms": DEFAULT_FORMAT,
    "shape": DEFAULT_FORMAT,
}
REFERENCE_METRIC = {"structure": "f1", "dms": "mae", "shape": "mae"}
REF_METRIC_SIGN = {"structure": 1, "dms": -1, "shape": -1}
POSSIBLE_METRICS = {
    "structure": ["f1", "mFMI"],
    "dms": ["mae", "r2", "pearson"],
    "shape": ["mae", "r2", "pearson"],
}


torch.set_default_dtype(torch.float32)
