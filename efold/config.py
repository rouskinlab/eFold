from torch import float32, cuda, backends
import torch

seq2int = {"X": 0, "A": 1, "C": 2, "G": 3, "U": 4}  # , 'S': 5, 'E': 6}
# seq2int = {"X": 0, "A": 1, "U": 2, "C": 3, "G": 4}  
int2seq = {v: k for k, v in seq2int.items()}

START_TOKEN = None  # seq2int['S']
END_TOKEN = None  # seq2int['E']
PADDING_TOKEN = seq2int["X"]

DEFAULT_FORMAT = float32
torch.set_default_dtype(DEFAULT_FORMAT)
UKN = -1000.0
VAL_GU = 0.095
device = (
    "cuda"
    if cuda.is_available()
    # else "mps"  # moi j'aime bien le mps
    # if backends.mps.is_available()
    else "cpu"
)

TEST_SETS = {
    "structure": ["PDB", "archiveII_blast", "lncRNA", "viral_fragments"], 
    "sequence": [],
    "dms": [],  
    "shape": [],  
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
    "structure": ["f1"],  # , "mFMI"],
    "dms": ["mae", "r2", "pearson"],
    "shape": ["mae", "r2", "pearson"],
}

DTYPE_PER_DATA_TYPE = {
    "structure": torch.int32,
    "dms": DEFAULT_FORMAT,
    "shape": DEFAULT_FORMAT,
}

torch.set_default_dtype(torch.float32)
