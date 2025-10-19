from typing import Any, Dict
from omegaconf import DictConfig, OmegaConf
from hydra.core.config_store import ConfigStore
import torch
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor
import wandb

from .config_schema import (
    Config,
    EFoldModelConfig,
    CNNModelConfig,
    TransformerModelConfig,
    DataConfig,
    TrainerConfig,
    LoggingConfig,
)


def register_configs() -> None:
    """Register structured configs with Hydra ConfigStore.
    
    This function registers all configuration schemas with Hydra's ConfigStore,
    enabling type checking and validation of configuration files.
    """
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=Config)
    cs.store(group="model", name="efold", node=EFoldModelConfig)
    cs.store(group="model", name="cnn", node=CNNModelConfig)
    cs.store(group="model", name="transformer", node=TransformerModelConfig)
    cs.store(group="data", name="base", node=DataConfig)
    cs.store(group="trainer", name="base", node=TrainerConfig)
    cs.store(group="logging", name="base", node=LoggingConfig)


def instantiate_model(cfg: DictConfig) -> Any:
    """Instantiate model from Hydra config.
    
    :param cfg: Hydra model configuration
    :return: Instantiated model
    """
    from efold.models.factory import create_model
    
    model_cfg = OmegaConf.to_container(cfg, resolve=True)
    model_cfg.pop("_target_", None)
    return create_model(**model_cfg)


def instantiate_datamodule(cfg: DictConfig) -> Any:
    """Instantiate data module from Hydra config.
    
    :param cfg: Hydra data configuration
    :return: Instantiated DataModule
    """
    from efold.core.datamodule import DataModule
    
    data_cfg = OmegaConf.to_container(cfg, resolve=True)
    data_cfg.pop("_target_", None)
    return DataModule(**data_cfg)


def instantiate_trainer(
    cfg: DictConfig, logger: Any = None, callbacks: list = None
) -> Trainer:
    """Instantiate PyTorch Lightning Trainer from Hydra config.
    
    :param cfg: Hydra trainer configuration
    :param logger: Optional logger instance
    :param callbacks: Optional list of callbacks
    :return: Instantiated Trainer
    """
    trainer_cfg = OmegaConf.to_container(cfg, resolve=True)
    
    strategy = trainer_cfg.pop("strategy", "auto")
    if strategy == "ddp":
        find_unused = trainer_cfg.pop("find_unused_parameters", False)
        strategy = DDPStrategy(find_unused_parameters=find_unused)
    
    return Trainer(
        strategy=strategy,
        logger=logger,
        callbacks=callbacks or [],
        **trainer_cfg,
    )


def setup_logging(cfg: DictConfig, model: Any = None) -> tuple[Any, list]:
    """Setup logging and callbacks from config.
    
    :param cfg: Hydra logging configuration
    :param model: Optional model to watch with wandb
    :return: Tuple of (logger, callbacks)
    """
    logger = None
    callbacks = []
    
    if cfg.use_wandb:
        logger = WandbLogger(
            project=cfg.project,
            name=cfg.name,
        )
        
        if model is not None and cfg.watch_model:
            logger.watch(model, log="all")
        
        callbacks.extend([
            LearningRateMonitor(logging_interval=cfg.log_interval),
        ])
        
        if cfg.checkpoint_every_n_epochs > 0:
            from .core.callbacks import ModelCheckpoint
            callbacks.append(
                ModelCheckpoint(every_n_epoch=cfg.checkpoint_every_n_epochs)
            )
    
    return logger, callbacks


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility.
    
    :param seed: Random seed value
    """
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

