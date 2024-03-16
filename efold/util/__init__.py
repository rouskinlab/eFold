from torch.optim import Adam
from torch import nn

str2fun = {
    "adam": Adam,
    "Adam": Adam,
    "mse_loss": nn.functional.mse_loss,
    "l1_loss": nn.functional.l1_loss,
}


def unzip(f):
    def wrapper(*args, **kwargs):
        out = f(*args, **kwargs)
        if not hasattr(out, "__iter__"):
            return out
        if len(out) == 1:
            return out[0]
        return out

    return wrapper
