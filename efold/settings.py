from pathlib import Path
from typing import Any

import torch
import yaml
from torch import backends, cuda, float32

_settings_path = Path(__file__).parent / "settings.yaml"


def _load_settings() -> dict[str, Any]:
    """
    Load settings from the YAML configuration file.

    :return: Dictionary containing all settings
    """
    with open(_settings_path, "r") as f:
        return yaml.safe_load(f)


_config = _load_settings()

seq2int = _config["seq2int"]
int2seq = {v: k for k, v in seq2int.items()}

START_TOKEN = _config["start_token"]
END_TOKEN = _config["end_token"]
PADDING_TOKEN = seq2int[_config["padding_token_key"]]

DEFAULT_FORMAT = float32
torch.set_default_dtype(DEFAULT_FORMAT)
UKN = _config["unknown_value"]
VAL_GU = _config["val_gu"]

device = "cuda" if cuda.is_available() else "cpu"

TEST_SETS = _config["test_sets"]
TEST_SETS_NAMES = [i for j in TEST_SETS.values() for i in j]
DATA_TYPES_TEST_SETS = [k for k, v in TEST_SETS.items() for i in v]

DATA_TYPES = _config["data_types"]

_dtype_mapping = {
    "float32": torch.float32,
    "int32": torch.int32,
}

DATA_TYPES_FORMAT = {k: _dtype_mapping[v] for k, v in _config["data_types_format"].items()}

REFERENCE_METRIC = _config["reference_metric"]
REF_METRIC_SIGN = _config["ref_metric_sign"]
POSSIBLE_METRICS = _config["possible_metrics"]

DTYPE_PER_DATA_TYPE = DATA_TYPES_FORMAT

torch.set_default_dtype(torch.float32)
