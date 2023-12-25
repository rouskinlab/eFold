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
        if true==None: return torch.tensor(0.0)
        mask = true != UKN
        # print(batch.get("error_{}".format(data_type)))
        # error = batch.get("error_{}".format(data_type))
        # if self.weight_data and
        #     return F.gaussian_nll_loss(
        #         pred[mask],
        #         true[mask],
        #         batch.get("error_{}".format(data_type))[mask],
        #         full=True,
        #         eps=5e-2,
        #     )/8
        return torch.sqrt(F.mse_loss(pred[mask], true[mask]))

    def _loss_structure(self, batch: Batch):
        # Unsure if this is the correct loss function
        pred, true = batch.get_pairs("structure")
        if true is None or pred is None:
            return torch.tensor(0.0)
        mask = true != UKN
        return self.lossBCE(pred[mask], true[mask])

    def loss_fn(self, batch: Batch):
        count = {k:v for k,v in batch.dt_count.items() if k in self.data_type_output}
        losses = {}
        if "dms" in count.keys():
            losses["dms"] = self._loss_signal(batch, "dms")
        if "shape" in count.keys():
            losses["shape"] = self._loss_signal(batch, "shape")
        if "structure" in count.keys():
            losses["structure"] = self._loss_structure(batch)
        loss = sum([losses[k] * count[k] for k in count.keys()]) / sum([v for v in count.values()])
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
