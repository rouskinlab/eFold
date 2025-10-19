import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

from efold.constants import device
from efold.hydra_utils import (
    register_configs,
    instantiate_model,
    instantiate_datamodule,
    instantiate_trainer,
    setup_logging,
    set_seed,
)


register_configs()


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function using Hydra configuration.
    
    Run with different configs using command line overrides:
        python train.py model=efold data=efold_train
        python train.py model=cnn data=structure trainer=ddp
        python train.py model=transformer data=ribonanza logging=wandb_enabled
    
    :param cfg: Hydra configuration object
    """
    print(OmegaConf.to_yaml(cfg))
    print(f"Running on device: {device}")
    
    set_seed(cfg.seed)
    
    dm = instantiate_datamodule(cfg.data)
    model = instantiate_model(cfg.model)
    
    logger, callbacks = setup_logging(cfg.logging, model)
    trainer = instantiate_trainer(cfg.trainer, logger=logger, callbacks=callbacks)
    
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)
    
    if cfg.logging.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()

