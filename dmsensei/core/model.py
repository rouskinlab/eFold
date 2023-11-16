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
from .batch import Batch


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

    def loss_fn(self, batch:Batch):
        loss = torch.tensor(0.0, device=self.device)
        if batch.contains("dms"):
            pred, true = batch.get_pairs("dms")
            mask = true != UKN
            loss += F.mse_loss(input=pred[mask], target=true[mask])
        if batch.contains("shape"):
            pred, true = batch.get_pairs("shape")
            mask = true != UKN
            loss += F.mse_loss(input=pred[mask], target=true[mask])
        if batch.contains("structure"):
            pred, true = batch.get_pairs("structure")
            loss += F.binary_cross_entropy(input=pred, target=true)
        return loss

    def training_step(self, batch:Batch, batch_idx:int):
        predictions = self.forward(
            batch.get("sequence")
        )
        batch.integrate_prediction(predictions)
        loss = self.loss_fn(batch)
        return loss

    def validation_step(self, batch:Batch, batch_idx:int):
        predictions = self.forward(
            batch.get("sequence")
        )
        batch.integrate_prediction(predictions)
        loss = self.loss_fn(batch)
        return loss

    def test_step(self, batch:Batch, batch_idx:int, dataloader_idx=0):
        predictions = self.forward(
            batch.get("sequence")
        )
        batch.integrate_prediction(predictions)

    def predict_step(self, batch:Batch, batch_idx:int):
        predictions = self.forward(
            batch.get("sequence")
        )
        # Hardcoded values for DMS G/U bases
        if "dms" in predictions.keys():
            predictions["dms"][
              (batch.get("sequence") == seq2int["G"])
                | (batch.get("sequence") == seq2int["U"])
            ] = VAL_GU

        batch.integrate_prediction(predictions)
