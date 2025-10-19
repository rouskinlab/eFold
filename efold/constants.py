from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import yaml
from torch import cuda


@dataclass
class TokenMapping:
    """Token mappings for nucleotides.

    :param seq2int: Mapping from sequence characters to integers
    :param start_token: Optional start token
    :param end_token: Optional end token
    :param padding_token_key: Key used for padding token
    """
    seq2int: dict[str, int]
    start_token: Optional[str]
    end_token: Optional[str]
    padding_token_key: str

    @property
    def int2seq(self) -> dict[int, str]:
        """Reverse mapping from integers to sequences.

        :return: Dictionary mapping integers to sequence characters
        """
        return {v: k for k, v in self.seq2int.items()}

    @property
    def padding_token(self) -> int:
        """Get the padding token integer value.

        :return: Integer value of padding token
        """
        return self.seq2int[self.padding_token_key]


@dataclass
class PyTorchConfig:
    """PyTorch configuration settings.

    :param unknown_value: Value used for unknown/missing data
    :param val_gu: Value for GU base pairing
    """
    unknown_value: float
    val_gu: float


@dataclass
class TestSets:
    """Test set definitions.

    :param structure: List of structure test sets
    :param sequence: List of sequence test sets
    :param dms: List of DMS test sets
    :param shape: List of SHAPE test sets
    """
    structure: list[str]
    sequence: list[str]
    dms: list[str]
    shape: list[str]

    @property
    def as_dict(self) -> dict[str, list[str]]:
        """Return test sets as dictionary.

        :return: Dictionary of test set types to names
        """
        return {
            "structure": self.structure,
            "sequence": self.sequence,
            "dms": self.dms,
            "shape": self.shape
        }

    @property
    def all_names(self) -> list[str]:
        """Get all test set names flattened.

        :return: List of all test set names
        """
        return [name for sets in self.as_dict.values() for name in sets]

    @property
    def data_types_per_test_set(self) -> list[str]:
        """Get data type for each test set.

        :return: List of data types matching test sets
        """
        return [dtype for dtype, names in self.as_dict.items() for _ in names]


@dataclass
class DataConfig:
    """Data types and format configuration.

    :param types: List of available data types
    :param types_format: Mapping of data types to their torch dtype
    """
    types: list[str]
    types_format: dict[str, str]

    @property
    def types_format_torch(self) -> dict[str, torch.dtype]:
        """Get data types as torch dtypes.

        :return: Dictionary mapping data types to torch.dtype objects
        """
        dtype_map = {
            "int32": torch.int32,
            "float32": torch.float32,
            "float64": torch.float64,
        }
        return {k: dtype_map[v] for k, v in self.types_format.items()}


@dataclass
class MetricsConfig:
    """Metrics configuration.

    :param reference_metric: Primary metric for each data type
    :param ref_metric_sign: Sign convention for metrics (1=higher is better, -1=lower is better)
    :param possible_metrics: List of available metrics per data type
    """
    reference_metric: dict[str, str]
    ref_metric_sign: dict[str, int]
    possible_metrics: dict[str, list[str]]


@dataclass
class Config:
    """Main configuration class combining all config sections.

    :param tokens: Token mapping configuration
    :param pytorch: PyTorch-specific configuration
    :param test_sets: Test set definitions
    :param data: Data type configuration
    :param metrics: Metrics configuration
    """
    tokens: TokenMapping
    pytorch: PyTorchConfig
    test_sets: TestSets
    data: DataConfig
    metrics: MetricsConfig
    device: str = field(init=False)

    def __post_init__(self) -> None:
        """Initialize device after other fields are set.

        :return: None
        """
        object.__setattr__(self, 'device', "cuda" if cuda.is_available() else "cpu")


def _load_config() -> Config:
    """Load configuration from YAML file.

    :return: Instantiated Config object
    """
    config_path = Path(__file__).parent / "constants.yaml"

    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)

    return Config(
        tokens=TokenMapping(
            seq2int=data['tokens']['seq2int'],
            start_token=data['tokens']['start_token'],
            end_token=data['tokens']['end_token'],
            padding_token_key=data['tokens']['padding_token_key']
        ),
        pytorch=PyTorchConfig(
            unknown_value=data['pytorch']['unknown_value'],
            val_gu=data['pytorch']['val_gu']
        ),
        test_sets=TestSets(
            structure=data['test_sets']['structure'],
            sequence=data['test_sets']['sequence'],
            dms=data['test_sets']['dms'],
            shape=data['test_sets']['shape']
        ),
        data=DataConfig(
            types=data['data']['types'],
            types_format=data['data']['types_format']
        ),
        metrics=MetricsConfig(
            reference_metric=data['metrics']['reference_metric'],
            ref_metric_sign=data['metrics']['ref_metric_sign'],
            possible_metrics=data['metrics']['possible_metrics']
        )
    )


config = _load_config()

torch.set_default_dtype(torch.float32)
