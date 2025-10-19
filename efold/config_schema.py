from dataclasses import dataclass, field
from typing import List, Optional, Union
from omegaconf import MISSING


@dataclass
class ModelConfig:
    """Base configuration for all models.
    
    :param model: Model type identifier
    :param ntoken: Number of tokens in vocabulary
    :param dropout: Dropout rate
    :param lr: Learning rate
    :param weight_decay: Weight decay for optimizer
    :param gamma: Learning rate scheduler gamma
    :param wandb: Whether to log to wandb
    """
    _target_: str = "efold.models.factory.create_model"
    model: str = MISSING
    ntoken: int = 5
    dropout: float = 0.0
    lr: float = 1e-3
    weight_decay: float = 0.0
    gamma: float = 0.995
    wandb: bool = False


@dataclass
class EFoldModelConfig(ModelConfig):
    """Configuration for eFold model.
    
    :param d_model: Model dimension
    :param c_z: Pairwise representation dimension
    :param d_cnn: CNN hidden dimension
    :param num_blocks: Number of transformer blocks
    :param no_recycles: Number of recycles
    """
    model: str = "efold"
    d_model: int = 64
    c_z: int = 32
    d_cnn: int = 64
    num_blocks: int = 4
    no_recycles: int = 0


@dataclass
class CNNModelConfig(ModelConfig):
    """Configuration for CNN model.
    
    :param d_model: Model dimension
    :param d_cnn: CNN hidden dimension
    :param n_heads: Number of attention heads
    """
    model: str = "cnn"
    d_model: int = 640
    d_cnn: int = 512
    n_heads: int = 16


@dataclass
class TransformerModelConfig(ModelConfig):
    """Configuration for Transformer model.
    
    :param data: Data type
    :param weight_data: Whether to weight data
    :param d_model: Model dimension
    :param nhead: Number of attention heads
    :param d_hid: Hidden dimension
    :param nlayers: Number of layers
    """
    model: str = "transformer"
    data: str = "multi"
    weight_data: bool = True
    d_model: int = 128
    nhead: int = 16
    d_hid: int = 256
    nlayers: int = 8


@dataclass
class DataConfig:
    """Configuration for data module.
    
    :param name: Dataset name or list of dataset names
    :param strategy: Sampling strategy
    :param shuffle_train: Whether to shuffle training data
    :param shuffle_valid: Whether to shuffle validation data
    :param data_type: List of data types
    :param force_download: Whether to force download
    :param batch_size: Batch size
    :param max_len: Maximum sequence length
    :param min_len: Minimum sequence length
    :param structure_padding_value: Padding value for structure
    :param train_split: Training split size
    :param external_valid: External validation datasets
    :param num_workers: Number of dataloader workers
    """
    _target_: str = "efold.core.datamodule.DataModule"
    name: Union[str, List[str]] = MISSING
    strategy: str = "random"
    shuffle_train: bool = True
    shuffle_valid: bool = False
    data_type: List[str] = field(default_factory=lambda: ["structure"])
    force_download: bool = False
    batch_size: int = 1
    max_len: Optional[int] = None
    min_len: Optional[int] = None
    structure_padding_value: int = 0
    train_split: Optional[Union[float, int]] = None
    external_valid: Optional[List[str]] = None
    num_workers: int = 0


@dataclass
class TrainerConfig:
    """Configuration for PyTorch Lightning trainer.
    
    :param accelerator: Accelerator type
    :param devices: Number of devices
    :param strategy: Training strategy
    :param max_epochs: Maximum number of epochs
    :param log_every_n_steps: Logging frequency
    :param accumulate_grad_batches: Gradient accumulation steps
    :param use_distributed_sampler: Whether to use distributed sampler
    :param enable_checkpointing: Whether to enable checkpointing
    :param precision: Training precision
    """
    accelerator: str = "auto"
    devices: int = 1
    strategy: str = "auto"
    max_epochs: int = 15
    log_every_n_steps: int = 1
    accumulate_grad_batches: int = 32
    use_distributed_sampler: bool = True
    enable_checkpointing: bool = False
    precision: Optional[str] = None


@dataclass
class LoggingConfig:
    """Configuration for logging.
    
    :param use_wandb: Whether to use Weights & Biases
    :param project: WandB project name
    :param name: Run name
    :param watch_model: Whether to watch model with wandb
    :param log_interval: Logging interval
    :param checkpoint_every_n_epochs: Checkpoint frequency
    """
    use_wandb: bool = False
    project: str = "efold-training"
    name: Optional[str] = None
    watch_model: bool = False
    log_interval: str = "epoch"
    checkpoint_every_n_epochs: int = 1


@dataclass
class Config:
    """Root configuration.
    
    :param model: Model configuration
    :param data: Data configuration
    :param trainer: Trainer configuration
    :param logging: Logging configuration
    :param seed: Random seed
    """
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    seed: int = 42

