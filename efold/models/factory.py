from efold.models import cnn, efold, ribonanza, transformer, unet


def create_model(model: str, *args, **kwargs):
    if model == "transformer":
        return transformer.Transformer(*args, **kwargs)
    if model == "efold":
        return efold.eFold(*args, **kwargs)
    if model == "cnn":
        return cnn.CNN(*args, **kwargs)
    if model == "unet":
        return unet.U_Net(*args, **kwargs)
    if model == "ribonanza":
        return ribonanza.Ribonanza(*args, **kwargs)
    raise ValueError(f"Unknown model: {model}")
