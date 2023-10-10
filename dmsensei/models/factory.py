from .dms import MultiLayerPerceptron as dms_mlp
from .dms import Transformer as dms_transformer
import wandb as wandb_module


class ModelFactory:
    def __init__(self):
        self.data = ["dms", "structure"]
        self.models = {"dms": ["mlp", "transformer"], "structure": []}

    def __call__(self, data: str, model: str, **kwargs):
        data = data.lower()
        return self.get_model(data, model, **kwargs)

    def get_model(self, data: str, model: str, *args, **kwargs):
        assert data in self.data, f"Data must be one of {self.data}"
        assert model in self.models[data], f"Model must be one of {self.models[data]}"

        if data == "dms":
            if model == "mlp":
                return dms_mlp(**kwargs)
            if model == "transformer":
                return dms_transformer(*args, **kwargs)

        raise NotImplementedError(f"Model {model} for data {data} not implemented.")


def create_model(data: str, model: str, wandb: bool = True, **kwargs):
    # log hyperparameters into wandb
    if wandb:
        wandb_module.init()
        wandb_module.log({"model": model})

    return ModelFactory()(data, model, **kwargs)
