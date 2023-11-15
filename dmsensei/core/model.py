from typing import Any
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT
from . import metrics
from .embeddings import (
    NUM_BASES,
    sequence_to_one_hot,
    int_dot_bracket_to_one_hot,
    pairing_matrix_to_base_pairs,
    int_to_sequence,
)
import torch.nn as nn
from torch import Tensor, tensor, zeros_like, mean
import torch
import numpy as np
from ..config import TEST_SETS_NAMES, UKN, VAL_GU
from scipy.stats.stats import pearsonr
from rouskinhf import seq2int
import wandb
import torch.nn.functional as F


class Model(pl.LightningModule):
    def __init__(self, lr: float, optimizer_fn, **kwargs):
        super().__init__()

        # Set attributes
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.lr = lr
        self.optimizer_fn = optimizer_fn
        self.automatic_optimization = True
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = self.optimizer_fn(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay if hasattr(self, "weight_decay") else 0,
        )
        if not hasattr(self, "scheduler") or self.scheduler is None:
            return optimizer

        scheduler = {
            "scheduler": self.scheduler(
                optimizer, patience=5, factor=0.5, verbose=True
            ),
            "interval": "epoch",
            "monitor": "valid/loss",
        }
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.gamma)
        return [optimizer], [scheduler]

    def loss_fn(self, outputs, data):
        loss = torch.tensor(0.0, device=self.device)
        L = len(data["sequence"]["values"][0])
        if "dms" in data.keys() and "dms" in outputs.keys():
            target = data["dms"]["values"]
            mask = target != UKN
            loss += F.mse_loss(
                input=outputs["dms"][data["dms"]["index"]][mask].squeeze(-1),
                target=target[mask],
            )
        if "shape" in data.keys() and "shape" in outputs.keys():
            target = data["shape"]["values"]
            mask = target != UKN
            loss += F.mse_loss(
                input=outputs["shape"][data["shape"]["index"]][mask].squeeze(-1),
                target=target[mask],
            )
        if "structure" in data.keys() and "structure" in outputs.keys():
            loss += F.binary_cross_entropy(
                input=outputs["structure"], target=data["structure"]
            )
        return loss

    def training_step(self, batch, batch_idx):
        data, metadata = batch
        predictions = self.forward(data["sequence"]["values"])
        loss = self.loss_fn(predictions, data)

        return loss

    def validation_step(self, batch, batch_idx):
        data, metadata = batch
        predictions = self.forward(data["sequence"]["values"])
        loss = self.loss_fn(predictions, data)

        return predictions, loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        data, metadata = batch
        predictions = self.forward(data["sequence"]["values"])

        return predictions

    def predict_step(self, batch, batch_idx):
        data, metadata = batch

        predictions = self.forward(data["sequence"]["values"])

        # Hardcoded values for DMS G/U bases
        if "dms" in predictions.keys():
            predictions["dms"][
                (data["sequence"]["values"] == seq2int["G"])
                | (data["sequence"]["values"] == seq2int["U"])
            ] = VAL_GU

        # pack predictions into lines
        outputs = []
        for i in range(len(metadata["reference"])):
            L = metadata["length"][i]
            d = {
                "sequence": int_to_sequence(data["sequence"]["values"][i])[:L],
                "reference": metadata["reference"][i],
            }
            for signal in ["dms", "shape"]:
                if signal in predictions.keys():
                    d[signal] = [round(d.item(), 4) for d in predictions[signal][i, :L]]
            if "structure" in predictions.keys():
                d["structure"] = pairing_matrix_to_base_pairs(
                    predictions["structure"][i, :L]
                )
            outputs.append(d)

        return outputs
