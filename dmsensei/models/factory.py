from .transformer import Transformer
from .evoformer import Evoformer


class ModelFactory:
    models = ["transformer", "evoformer"]

    def __init__(self):
        pass

    def __call__(self, model: str, **kwargs):
        return self.get_model(model, **kwargs)

    def get_model(self, model: str, *args, **kwargs):
        assert model in self.models, f"Model must be one of {self.models}"

        if model == "transformer":
            return Transformer(*args, **kwargs)
        if model == "evoformer":
            return Evoformer(*args, **kwargs)


def create_model(model: str, **kwargs):
    return ModelFactory()(model, **kwargs)
