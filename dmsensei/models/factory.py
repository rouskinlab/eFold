from .dms import MultiLayerPerceptron as dms_mlp
from .multi import Transformer as multi_transformer
from .multi import Evoformer as multi_evoformer

import wandb as wandb_module


class ModelFactory:
    def __init__(self):
        self.models = {
            "dms": ["mlp", "transformer", "evoformer"],
            "structure": [],
            "multi": ["transformer", "evoformer"],
        }

    def __call__(self, data: str, model: str, **kwargs):
        data = data.lower()
        return self.get_model(data, model, **kwargs)

    def get_model(self, data: str, model: str, *args, **kwargs):
        assert data in list(
            self.models.keys()
        ), f"Data must be one of {list(self.models.keys())}"
        assert model in self.models[data], f"Model must be one of {self.models[data]}"

        if data == "dms":
            if model == "mlp":
                return dms_mlp(**kwargs)
        if data == "multi":
            if model == "transformer":
                return multi_transformer(*args, **kwargs)
            if model == "evoformer":
                return multi_evoformer(*args, **kwargs)

        raise NotImplementedError(f"Model {model} for data {data} not implemented.")


def create_model(data: str, model: str, wandb: bool = True, **kwargs):
    # log hyperparameters into wandb
    if wandb:
        wandb_module.init()
        wandb_module.log({"model": model})

    return ModelFactory()(data, model, **kwargs)
