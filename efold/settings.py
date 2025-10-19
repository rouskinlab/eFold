from typing import Dict, List
import torch
from pathlib import Path
import yaml

from efold.constants import DEFAULT_FORMAT


_CONF_DIR = Path(__file__).parent.parent / "conf"
_DATASETS_CONFIG = _CONF_DIR / "datasets.yaml"


def _load_datasets_config() -> dict:
    """Load datasets configuration from YAML.
    
    :return: Dictionary with dataset configuration
    """
    with open(_DATASETS_CONFIG, "r") as f:
        return yaml.safe_load(f)


_config = _load_datasets_config()

TEST_SETS: Dict[str, List[str]] = _config["test_sets"]
TEST_SETS_NAMES: List[str] = [i for j in TEST_SETS.values() for i in j]
DATA_TYPES_TEST_SETS: List[str] = [k for k, v in TEST_SETS.items() for i in v]

DATA_TYPES: List[str] = _config["data_types"]

DATA_TYPES_FORMAT: Dict[str, torch.dtype] = {
    "structure": torch.int32,
    "dms": DEFAULT_FORMAT,
    "shape": DEFAULT_FORMAT,
}

REFERENCE_METRIC: Dict[str, str] = _config["metrics"]["reference"]
REF_METRIC_SIGN: Dict[str, int] = _config["metrics"]["sign"]
POSSIBLE_METRICS: Dict[str, List[str]] = _config["metrics"]["possible"]

DTYPE_PER_DATA_TYPE: Dict[str, torch.dtype] = {
    "structure": torch.int32,
    "dms": DEFAULT_FORMAT,
    "shape": DEFAULT_FORMAT,
}

