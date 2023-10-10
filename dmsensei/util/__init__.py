from torch.optim import Adam
from torch import nn

str2fun = {
    "adam": Adam,
    "Adam": Adam,
    "mse_loss": nn.functional.mse_loss,
    "l1_loss": nn.functional.l1_loss,
}
