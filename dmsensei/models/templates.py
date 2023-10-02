import lightning.pytorch as pl
from ..core import metrics
from ..core.embeddings import NUM_BASES, sequence_to_one_hot, int_dot_bracket_to_one_hot
import torch.nn as nn
from torch import Tensor, tensor, zeros_like, mean
import torch
import numpy as np
from ..config import TEST_SETS_NAMES, UKN
from scipy.stats.stats import pearsonr
from rouskinhf import seq2int


class Model(pl.LightningModule):
    def __init__(self, lr: float, loss_fn, optimizer_fn, **kwargs):
        super().__init__()

        # Set attributes
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.lr = lr
        self.loss_fn = loss_fn
        self.optimizer_fn = optimizer_fn
        self.automatic_optimization = True
        self.save_hyperparameters()

    def configure_optimizers(self):
        return self.optimizer_fn(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay if hasattr(self, "weight_decay") else 0,
        )


class StructureModel(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data_type = "structure"

    def validation_step(self, batch, batch_idx):
        inputs, label = batch
        outputs = self.forward(inputs)

        # Compute loss
        loss = self.loss_fn(outputs, label)
        f1 = metrics.compute_f1(outputs, label, threshold=0.5)
        mFMI = metrics.compute_mFMI(outputs, label, threshold=0.5)

        # Logging to Wandb
        self.log("valid/loss", loss)
        self.log("valid/f1", f1)
        self.log("valid/mFMI", mFMI)

        return outputs, loss

    def training_step(self, batch, batch_idx):
        inputss, targets = batch
        outputs = self.forward(inputss)

        # Compute loss
        loss = self.loss_fn(outputs, targets)

        # Logging to TensorBoard
        self.log("train/loss", loss)

        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        inputs, label = batch
        outputs = self.forward(inputs)

        # Compute and log loss
        loss = self.loss_fn(outputs, label)
        test_set_name = TEST_SETS_NAMES[self.data_type][dataloader_idx]
        self.log(f"test/{test_set_name}", loss, add_dataloader_idx=False)

        return outputs


class DMSModel(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data_type = "dms"

    def validation_step(self, batch, batch_idx):
        
        inputs, label = batch

        mask = (inputs == seq2int['G']) | (inputs == seq2int['U'])
        assert (label[mask] == UKN).all(), "Data is not consistent: G and U bases are not UKN."

        outputs = self.forward(inputs)

        # Compute and log loss
        mask = label != UKN
        loss = self.loss_fn(outputs[mask], label[mask])
        r2 = mean(
            tensor(
                [
                    metrics.r2_score(y_true, y_pred)
                    for y_true, y_pred in zip(label, outputs)
                ]
            )
        )

        # Logging to Wandb
        self.log("valid/loss", loss)
        self.log("valid/r2", r2)

        return outputs, loss


    def training_step(self, batch, batch_idx):
        inputs, label = batch
        
        mask = (inputs == seq2int['G']) | (inputs == seq2int['U'])
        assert (label[mask] == UKN).all(), "Data is not consistent: G and U bases are not UKN."
        
        outputs = self.forward(inputs)

        # Compute and log loss
        mask = label != UKN
        loss = self.loss_fn(outputs[mask], label[mask])

        # Logging to TensorBoard
        self.log("train/loss", loss.item())

        return loss


    def test_step(self, batch, batch_idx, dataloader_idx=0):
        inputs, label = batch
        outputs = self.forward(inputs)

        r2 = mean(
            tensor(
                [
                    metrics.r2_score(y_true, y_pred)
                    for y_true, y_pred in zip(label, outputs)
                ]
            )
        )

        test_set_name = TEST_SETS_NAMES[self.data_type][dataloader_idx]
        self.log(f"test/{test_set_name}/r2", r2, add_dataloader_idx=False)

        return outputs
