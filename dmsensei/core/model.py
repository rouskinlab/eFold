import lightning.pytorch as pl
import torch.nn as nn
import torch
from ..config import device, UKN
import torch.nn.functional as F
from .batch import Batch
from torchmetrics import R2Score, PearsonCorrCoef, MeanAbsoluteError, F1Score

METRIC_ARGS = dict(sync_on_compute=False)

class Model(pl.LightningModule):
    def __init__(self, lr: float, optimizer_fn, weight_data: bool = False, **kwargs):
        super().__init__()

        # Set attributes
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.lr = lr
        self.optimizer_fn = optimizer_fn
        self.automatic_optimization = True

        self.weight_data = weight_data
        self.save_hyperparameters()
        self.lossBCE = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([300])).to(device)
        
        # Metrics
        self.shape_R2 = R2Score(**METRIC_ARGS)
        self.shape_r2 = PearsonCorrCoef(**METRIC_ARGS)
        self.shape_mae = MeanAbsoluteError(**METRIC_ARGS)
        self.dms_R2 = R2Score(**METRIC_ARGS)
        self.dms_r2 = PearsonCorrCoef(**METRIC_ARGS)
        self.dms_mae = MeanAbsoluteError(**METRIC_ARGS)
        self.structure_f1 = F1Score('binary', **METRIC_ARGS)
        self.metrics_pack = dict(
            shape = dict(
                R2 = self.shape_R2, 
                r2 = self.shape_r2, 
                mae = self.shape_mae
            ),
            dms = dict(
                R2 = self.dms_R2, 
                r2 = self.dms_r2, 
                mae = self.dms_mae
            ),
            structure = dict(
                f1 = self.structure_f1
            )
        )
    
    def update_metrics(self, batch:Batch):
        for dt, metrics in self.metrics_pack.items():
            pred, true = batch.get_pairs(dt)
            mask = true != UKN
            if not len(mask):
                continue
            for metric in metrics.values():
                metric.update(pred[mask], true[mask])

    def configure_optimizers(self):
        optimizer = self.optimizer_fn(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay if hasattr(self, "weight_decay") else 0,
        )

        if not hasattr(self, "gamma") or self.gamma is None:
            return optimizer

        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.gamma)
        return [optimizer], [scheduler]

    def _loss_signal(self, batch: Batch, data_type: str):
        assert data_type in [
            "dms",
            "shape",
        ], "This function only works for dms and shape data"
        pred, true = batch.get_pairs(data_type)
        mask = torch.zeros_like(true)
        mask[true!=UKN] = 1
        loss = F.mse_loss(pred*mask, true*mask)
        assert not torch.isnan(loss), "Loss is NaN for {}".format(data_type)
        return loss

    def _loss_structure(self, batch: Batch):
        pred, true = batch.get_pairs("structure")
        loss = self.lossBCE(pred, true)
        assert not torch.isnan(loss), "Loss is NaN for structure"
        return loss

    def loss_fn(self, batch: Batch):
        count = {k:v for k,v in batch.dt_count.items() if k in self.data_type_output}
        losses = {}
        if "dms" in count.keys():
            losses["dms"] = self._loss_signal(batch, "dms")
        if "shape" in count.keys():
            losses["shape"] = self._loss_signal(batch, "shape")
        if "structure" in count.keys():
            losses["structure"] = self._loss_structure(batch)
        assert len(losses) > 0, "No data types to train on"
        assert len(count) == len(losses), "Not all data types have a loss function"
        loss = sum([losses[k] for k in count.keys()])
        assert not torch.isnan(loss), "Loss is NaN"
        return loss, losses

    def _clean_predictions(self, batch, predictions):
        # clip values to [0, 1]
        for data_type in set(["dms", "shape"]).intersection(predictions.keys()):
            predictions[data_type] = torch.clip(predictions[data_type], min=0, max=1)
        return predictions

    def training_step(self, batch: Batch, batch_idx: int):
        predictions = self.forward(batch)
        batch.integrate_prediction(predictions)
        loss = self.loss_fn(batch)[0]
        self.log(f"train/loss", loss, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Batch, batch_idx: int, dataloader_idx=0):
        predictions = self.forward(batch)
        batch.integrate_prediction(predictions)
        loss, losses = self.loss_fn(batch)
        self.update_metrics(batch)
        val_dataloader = self.trainer.datamodule.external_valid[dataloader_idx-1] if dataloader_idx else 'train_subset'
        for dt, metrics in self.metrics_pack.items():
            for name, metric in metrics.items():
                if metric.update_count:
                    self.log(f"valid/{val_dataloader}/{dt}/{name}", metric, logger=True, on_epoch=True, sync_dist=True, add_dataloader_idx=False)
        return loss, losses
    
    def test_step(self, batch: Batch, batch_idx: int, dataloader_idx=0):
        predictions = self.forward(batch)
        predictions = self._clean_predictions(batch, predictions)
        batch.integrate_prediction(predictions)

    def predict_step(self, batch: Batch, batch_idx: int):
        predictions = self.forward(batch)
        predictions = self._clean_predictions(batch, predictions)
        batch.integrate_prediction(predictions)

    def teardown(self, batch:Batch):
        del batch