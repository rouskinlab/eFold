from typing import Any
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT
from ..core import metrics
from ..core.embeddings import NUM_BASES, sequence_to_one_hot, int_dot_bracket_to_one_hot, pairing_matrix_to_base_pairs, int_to_sequence
import torch.nn as nn
from torch import Tensor, tensor, zeros_like, mean
import torch
import numpy as np
from ..config import TEST_SETS_NAMES, UKN, VAL_GU
from scipy.stats.stats import pearsonr
from rouskinhf import seq2int
import wandb
import torch.nn.functional as F


def extract_from_batch_data(batch, key):
    return batch[key]["values"][batch[key]["indexes"]]


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
        L = len(data["sequence"][0])
        if "dms" in data.keys() and "dms" in outputs.keys():
            target = data["dms"]["values"]
            mask = target != UKN
            loss += F.mse_loss(
                input=outputs["dms"][data["dms"]["indexes"]][mask].squeeze(-1), target=target[mask]
            )
        if "shape" in data.keys() and "shape" in outputs.keys():
            target = data["shape"]["values"]
            mask = target != UKN
            loss += F.mse_loss(input=outputs["shape"][data["shape"]["indexes"]][mask].squeeze(-1), target=target[mask])
        if "structure" in data.keys() and "structure" in outputs.keys():
            loss += F.binary_cross_entropy(
                input=outputs["structure"], target=data["structure"]
            )
        return loss

    def predict_step(self, batch, batch_idx):
        data, metadata = batch

        predictions = self.forward(data["sequence"])

        # Hardcoded values for DMS G/U bases
        if "dms" in predictions.keys():
            predictions["dms"][
                (data['sequence'] == seq2int["G"]) | (data['sequence'] == seq2int["U"])
            ] = VAL_GU
        
        # group by reference
        
        outputs = []
        for i in range(len(metadata["reference"])):
            L = metadata["length"][i]
            d = {
                "sequence": int_to_sequence(data["sequence"][i])[:L],
                "reference": metadata["reference"][i],
            }
            if "dms" in predictions.keys():
                d["dms"] = [round(d.item(),4) for d in predictions["dms"][i,:L]]
            if "shape" in predictions.keys():
                d["shape"] = [round(d.item(),4) for d in predictions["shape"][i,:L]]
            if "structure" in predictions.keys():
                d["structure"] = pairing_matrix_to_base_pairs(predictions["structure"][i,:L])
            outputs.append(d)
            
        return outputs

    def training_step(self, batch, batch_idx):
        data, metadata = batch

        outputs = self.forward(data["sequence"])
        loss = self.loss_fn(outputs, data)

        # Logging to TensorBoard
        self.log("train/loss", torch.sqrt(loss).item(), sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        data, metadata = batch
        outputs = self.forward(data["sequence"])

        # Compute metrics
        loss = self.loss_fn(outputs, data)
        batch_metrics = self._compute_metrics(outputs, data)

        # Logging to Wandb
        self._log_metrics(batch_metrics, "valid")

        return outputs, loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        data, metadata = batch
        outputs = self.forward(data["sequence"])

        # Compute metrics
        batch_metrics = self._compute_metrics(outputs, data)

        # Logging to Wandb
        self._log_metrics(batch_metrics, "test")

        return outputs

    def _compute_metrics(self, outputs, data):
        if "structure" in data and "structure" in outputs:
            struct_preds, struct_targets = (
                outputs["structure"],
                extract_from_batch_data(data, "structure"),
            )  # TODO dotbrackets vs matrix to fix
        if "dms" in data and "dms" in outputs:
            dms_preds, dms_targets = (
                outputs["dms"],
                extract_from_batch_data(data, "dms"),
            )
        if "shape" in data and "shape" in outputs:
            shape_preds, shape_targets = (
                outputs["shape"],
                extract_from_batch_data(data, "shape"),
            )

        return {
            "structure": {
                "f1": metrics.compute_f1_batch(
                    pred=struct_preds,
                    true=struct_targets,
                    threshold=0.5,
                ),
                "mFMI": metrics.compute_mFMI_batch(
                    pred=struct_preds,
                    true=struct_targets,
                    threshold=0.5,
                ),
            }
            if "structure" in data and "structure" in outputs
            else None,
            "dms": {
                "r2": metrics.r2_score_batch(pred=dms_preds, true=dms_targets),
                "mae": metrics.mae_score_batch(pred=dms_preds, true=dms_targets),
                "pearson": metrics.pearson_coefficient_batch(
                    pred=dms_preds, true=dms_targets
                ),
            }
            if "dms" in data and "dms" in outputs
            else None,
            "shape": {
                "r2": metrics.r2_score_batch(pred=shape_preds, true=shape_targets),
                "mae": metrics.mae_score_batch(pred=shape_preds, true=shape_targets),
                "pearson": metrics.pearson_coefficient_batch(
                    pred=shape_preds, true=shape_targets
                ),
            }
            if "shape" in data and "shape" in outputs
            else None,
        }

    def _log_metrics(self, metrics, prefix):
        for name_data, data in metrics.items():
            if data is None:
                continue
            for name_metric, metric in data.items():
                self.log(
                    f"{prefix}/{name_data}/{name_metric}",
                    metric,
                    sync_dist=True,
                )
