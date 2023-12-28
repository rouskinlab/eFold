import lightning.pytorch as pl
import torch.nn as nn
import torch
from ..config import device, UKN
import torch.nn.functional as F
from .batch import Batch


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
        # pred = pred.clone(); true = true.clone()
        # mask = true == UKN
        mask = torch.zeros_like(true)
        mask[true!=UKN] = 1
        
        # true[mask] = true[mask] * torch.tensor(0.0, requires_grad=True)
        # pred[mask] = pred[mask] * torch.tensor(0.0, requires_grad=True)
        # loss = torch.sqrt(F.mse_loss(pred, true))
        loss = F.mse_loss(pred*mask, true*mask)
        assert not torch.isnan(loss), "Loss is NaN for {}".format(data_type)
        return loss

    def _loss_structure(self, batch: Batch):
        # Unsure if this is the correct loss function
        pred, true = batch.get_pairs("structure")
        # pred = pred.clone(); true = true.clone()
        # mask = true == UKN
        # true[mask] = true[mask] * torch.tensor(0.0, requires_grad=True)
        # pred[mask] = pred[mask] * torch.tensor(0.0, requires_grad=True)
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
        # loss = sum([losses[k] * count[k] for k in count.keys()]) / sum(count.values())
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
        return loss

    def validation_step(self, batch: Batch, batch_idx: int, dataloader_idx=0):
        predictions = self.forward(batch)
        batch.integrate_prediction(predictions)
        loss, losses = self.loss_fn(batch)
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