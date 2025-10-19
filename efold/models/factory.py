from efold.models.transformer import Transformer
from efold.models.efold import eFold
from efold.models.cnn import CNN
from efold.models.ribonanza import Ribonanza
from efold.models.unet import U_Net


def create_model(model: str, *args, **kwargs):
    if model == "transformer":
        return Transformer(*args, **kwargs)
    if model == "efold":
        return eFold(*args, **kwargs)
    if model == "cnn":
        return CNN(*args, **kwargs)
    if model == "unet":
        return U_Net(*args, **kwargs)
    if model == "ribonanza":
        return Ribonanza(*args, **kwargs)
    raise ValueError(f"Unknown model: {model}")
