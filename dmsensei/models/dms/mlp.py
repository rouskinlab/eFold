import torch
import torch.nn as nn
import numpy as np
from ...config import DEFAULT_FORMAT, device
from torch import optim, nn, utils, Tensor, tensor
import lightning.pytorch as pl
from ...core import metrics
from ..templates import DMSModel
from ...core.embeddings import NUM_BASES
from torcheval.metrics import R2Score


torch.seed()
np.random.seed(0)


class MultiLayerPerceptron(DMSModel):
    def __init__(
        self,
        input_dim,
        hidden_layers=[2048, 2048, 1024, 512],
        lr=1e-3,
        loss_fn=nn.functional.mse_loss,
        optimizer_fn=torch.optim.Adam,
        embedding_dim=640,
        model_dim=128,
        weight_decay=0,
        dropout=0,
        **kwargs,
    ):
        super().__init__(
            lr=lr,
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
            weight_decay=weight_decay,
            **kwargs,
        )

        hidden_layers = hidden_layers + [input_dim]
        self.embedding = nn.Embedding(NUM_BASES, embedding_dim)

        self.block1 = nn.Sequential(
            nn.Linear(embedding_dim, model_dim, dtype=DEFAULT_FORMAT),
            nn.ReLU(),
        )

        self.flatten = nn.Flatten(start_dim=1)

        # dynamically create the hidden layers with D
        self.block2 = nn.Sequential(
            nn.Linear(
                input_dim * model_dim,
                hidden_layers[0],
                dtype=DEFAULT_FORMAT,
            ),
            nn.ReLU(),
            *[
                nn.Sequential(
                    nn.Linear(
                        hidden_layers[i], hidden_layers[i + 1], dtype=DEFAULT_FORMAT
                    ),
                    # dropout
                    nn.Dropout(
                        dropout if not hasattr(dropout, "__len__") else dropout[i]
                    ),
                    nn.ReLU() if i < len(hidden_layers) - 2 else nn.Identity(),
                )
                for i in range(len(hidden_layers) - 1)
            ],
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor):
        x = self.embedding(x)
        x = self.block1(x)
        x = self.flatten(x)
        x = self.block2(x)
        # logits = self.sigmoid(x)
        return x  # logits
