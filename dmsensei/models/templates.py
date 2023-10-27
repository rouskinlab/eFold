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
import wandb

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
        
        optimizer = self.optimizer_fn(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay if hasattr(self, "weight_decay") else 0,
        )
        if not hasattr(self, "scheduler") or self.scheduler is None:
            return optimizer

        scheduler = {"scheduler": self.scheduler(optimizer, patience=5, factor=0.5, verbose=True), "interval": "epoch", "monitor": "valid/loss"}
        return [optimizer], [scheduler]
        



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
        self.log("valid/loss", loss, sync_dist=True)
        self.log("valid/f1", f1, sync_dist=True)
        self.log("valid/mFMI", mFMI, sync_dist=True)

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

        # mask = (inputs == seq2int["G"]) | (inputs == seq2int["U"])
        # assert (
        #     label[mask] == UKN
        # ).all(), "Data is not consistent: G and U bases are not UKN."

        outputs = self.forward(inputs)

        # Compute and log loss
        mask = label != UKN
        loss = self.loss_fn(outputs[mask], label[mask])
        
        # label[label==UKN] = -1
        # loss = self.loss_fn(outputs, label)
        
        r2_scores =  tensor(
                [
                    metrics.r2_score(y_true, y_pred)
                    for y_true, y_pred in zip(label, outputs)
                ]
            )
        mae = mean(
            tensor(
                [
                    metrics.mae_score(y_true, y_pred)
                    for y_true, y_pred in zip(label, outputs)
                ]
            )
        )
        # mae_ACGU = mean(
        #     tensor(
        #         [
        #             metrics.mae_score_ACGU(seq, y_true, y_pred)
        #             for seq, y_true, y_pred in zip(inputs, label, outputs)
        #         ]
        #     )
        # )
        
        this_mean, this_std = metrics.mean_std_dms(outputs)

        # Logging to Wandb
        self.log("valid/loss", np.sqrt(loss.item()), sync_dist=True)
        self.log("valid/r2",  mean(r2_scores), sync_dist=True)
        self.log("valid/mae", mae, sync_dist=True)
        self.log("valid/mean", this_mean, sync_dist=True)
        self.log("valid/std", this_std, sync_dist=True)
        # wandb.log({"valid/r2_hist": wandb.Histogram(r2_scores)})
        # self.log("valid/mae_ACGU", mae_ACGU)
        
        return outputs, loss

    def training_step(self, batch, batch_idx):
        inputs, label = batch

        # mask = (inputs == seq2int["G"]) | (inputs == seq2int["U"])
        # assert (
        #     label[mask] == UKN
        # ).all(), "Data is not consistent: G and U bases are not UKN."

        outputs = self.forward(inputs)

        # Compute and log loss
        mask = label != UKN
        loss = self.loss_fn(outputs[mask], label[mask])
        # mask = torch.zeros_like(label)
        # mask[label != UKN] = 1
        # loss = self.loss_fn(outputs*mask, label*mask)
        # label[label==UKN] = -1
        # loss = self.loss_fn(outputs, label)

        # Logging to TensorBoard
        self.log("train/loss", np.sqrt(loss.item()), sync_dist=True)

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
        
        mae = mean(
            tensor(
                [
                    metrics.mae_score(y_true, y_pred)
                    for y_true, y_pred in zip(label, outputs)
                ]
            )
        )

        test_set_name = TEST_SETS_NAMES[self.data_type][dataloader_idx]
        self.log(f"test/{test_set_name}/r2", r2, add_dataloader_idx=False, sync_dist=True)
        self.log(f"test/{test_set_name}/mae", mae, add_dataloader_idx=False, sync_dist=True)
        
        return outputs
