import sys, os

sys.path.append(os.path.abspath("."))
# os.system('source /Users/alberic/Desktop/Pro/RouskinLab/projects/deep_learning/RNA_data/env')
from dmsensei import DataModule, create_model, metrics
from dmsensei.config import device
from rouskinhf import seq2int
import pandas as pd
from dmsensei.core.callbacks import PredictionLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from dmsensei.core.callbacks import PredictionLogger
import pandas as pd
from pytorch_lightning import Trainer
from dmsensei.config import device
import sys
import os
from pytorch_lightning.loggers import WandbLogger
import numpy as np
from torch import nn, tensor, mean
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from rouskinhf.util import seq2int, dot2int, int2dot, int2seq
from torch import nn, tensor, float32, cuda, backends, Tensor

NUM_BASES = len(set(seq2int.values()))

UKN = -1000.0
DEFAULT_FORMAT = float32


sys.path.append(os.path.abspath("."))
# os.system('source /Users/alberic/Desktop/Pro/RouskinLab/projects/deep_learning/RNA_data/env') why do you need this?

if __name__ == "__main__":
    print("Running on device: {}".format(device))

    MAX_LEN = 1024

    # Create dataset
    dm = DataModule(
        name="utr",
        data="dms",
        force_download=False,
        batch_size=64,
        num_workers=1,
        train_split=0.8,
        valid_split=0.2,
        zero_padding_to=MAX_LEN,
        overfit_mode=True,
    )


    class MultiLayerPerceptron(nn.Module):
        def __init__(
            self,
            input_dim = MAX_LEN,
            hidden_layers=[2048, 2048, 1024, 512],
            lr=1e-3,
            loss_fn=nn.functional.mse_loss,
            optimizer_fn=torch.optim.Adam,
            embedding_dim=640,
            model_dim=128,
            weight_decay=0,
        ):
            super(MultiLayerPerceptron, self).__init__()

            hidden_layers = hidden_layers + [input_dim]
            self.embedding = nn.Embedding(NUM_BASES, embedding_dim)
            self.lr = lr
            self.loss_fn = loss_fn
            self.optimizer_fn = optimizer_fn
            self.weight_decay = weight_decay

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
                        nn.ReLU() if i < len(hidden_layers) - 2 else nn.Identity(),
                    )
                    for i in range(len(hidden_layers) - 1)
                ],
            )

        def forward(self, x: Tensor):
            x = self.embedding(x)
            x = self.block1(x)
            x = self.flatten(x)
            x = self.block2(x)
            return x  # logits

    # Training and testing functions
    def train(model, device, train_loader, optimizer, epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = model.loss_fn(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )

    def test(model, device, test_loader):
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += model.loss_fn(output, target).item()

        test_loss /= len(test_loader.dataset)
        print(
            "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                test_loss,
                len(test_loader.dataset),
            )
        )

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    # Initialize model and optimizer
    model = MultiLayerPerceptron(
        ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train and test
    for epoch in range(1, 6):  # 5 epochs
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, val_loader)
