from typing import Any
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch.nn as nn
import torch
from ..config import device, UKN, TEST_SETS_NAMES
import torch.nn.functional as F
from .batch import Batch
from torchmetrics import R2Score, PearsonCorrCoef, MeanAbsoluteError, F1Score
from .metrics import MetricsStack
from .datamodule import DataModule
import time

from .postprocess import postprocess_new_nc as postprocess

METRIC_ARGS = dict(dist_sync_on_step=True)


def loss_pearson(pred, target, eps=1e-10):
    pearson = ((pred - pred.mean()) * (target - target.mean())).mean() / (
        pred.std() * target.std() + eps
    )
    return 1 - pearson**2


def corrcoef(target, pred):
    # np.corrcoef in torch from @mdo
    # https://forum.numer.ai/t/custom-loss-functions-for-xgboost-using-pytorch/960
    pred_n = pred - pred.mean()
    target_n = target - target.mean()
    pred_n = pred_n / pred_n.norm()
    target_n = target_n / target_n.norm()
    return 1 - ((pred_n * target_n).sum()) ** 2


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
        self.save_hyperparameters(ignore=['loss_fn'])
        self.lossBCE = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([300])).to(device)

        # Metrics
        self.metrics_stack = None
        self.tic = None

        self.test_results = {'reference':[], 'sequence':[] ,'structure':[]}

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
        ## vv Pearson loss vv ##
        # if (true==UKN).sum() == true.numel():
        #     loss = F.mse_loss(0*true, 0*pred)
        # else:
        #     mask = true != UKN
        #     loss = corrcoef(pred[mask], true[mask])
        ## ^^ Pearson loss ^^ ##

        ## vv MSE loss vv ##
        mask = torch.zeros_like(true)
        mask[true != UKN] = 1
        loss = F.mse_loss(pred * mask, true * mask)

        non_zeros = (mask == 1).sum() / mask.numel()
        if non_zeros != 0:
            loss /= non_zeros
        ## ^^ MSE loss ^^ ##

        assert not torch.isnan(loss), "Loss is NaN for {}".format(data_type)
        return loss

    def _loss_structure(self, batch: Batch):
        pred, true = batch.get_pairs("structure")
        loss = self.lossBCE(pred, true)
        assert not torch.isnan(loss), "Loss is NaN for structure"
        return loss

    def loss_fn(self, batch: Batch):
        count = {k: v for k, v in batch.dt_count.items() if k in self.data_type_output}
        losses = {}
        if "dms" in count.keys():
            losses["dms"] = self._loss_signal(batch, "dms")
        if "shape" in count.keys():
            losses["shape"] = self._loss_signal(batch, "shape")
        if "structure" in count.keys():
            losses["structure"] = self._loss_structure(batch)
        assert len(losses) > 0, "No data types to train on"
        assert len(count) == len(losses), "Not all data types have a loss function"
        loss = 0
        for data_type, loss_value in losses.items():
            loss += loss_value * count[data_type]
        loss /= sum(count.values())

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
        self.log(f"train/loss", loss, sync_dist=True)
        return loss

    def on_validation_start(self):
        val_dl_names = self.trainer.datamodule.external_valid
        self.metrics_stack = [
            MetricsStack(name=name, data_type=self.data_type_output)
            for name in val_dl_names
        ]

    def on_train_batch_end(
        self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        del outputs
        del batch
        if batch_idx % 100 == 0:
            torch.cuda.empty_cache()

    def on_train_end(self) -> None:
        torch.cuda.empty_cache()

    def validation_step(self, batch: Batch, batch_idx: int, dataloader_idx=0):
        predictions = self.forward(batch)
        predictions['structure'] = postprocess(predictions['structure'], 
                                               self.seq2oneHot(batch.get('sequence')),
                                               0.01, 0.1, 100, 1.6, True, 1.5)

        batch.integrate_prediction(predictions)
        # loss, losses = self.loss_fn(batch)
        self.metrics_stack[dataloader_idx].update(batch)
        # return loss, losses
        return 0, {}

    def on_validation_batch_end(
        self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        if batch_idx % 100 == 0:
            torch.cuda.empty_cache()

    def on_validation_epoch_end(self) -> None:
        # aggregate the stack and log it
        for metrics_dl in self.metrics_stack:
            metrics_pack = metrics_dl.compute()
            for dt, metrics in metrics_pack.items():
                for name, metric in metrics.items():
                    # to replace with a gather_all?
                    self.log(
                        f"valid/{metrics_dl.name}/{dt}/{name}",
                        metric,
                        add_dataloader_idx=False,
                        sync_dist=True,
                    )
        self.metrics_stack = None
        torch.cuda.empty_cache()

    def test_step(self, batch: Batch, batch_idx: int, dataloader_idx=0):
        predictions = self.forward(batch)
        predictions['structure'] = postprocess(predictions['structure'], 
                                               self.seq2oneHot(batch.get('sequence')),
                                               0.01, 0.1, 100, 1.6, True, 1.5)
        

        from ..config import int2seq
        self.test_results['reference'] += batch.get('reference')
        self.test_results['sequence'] += [''.join([int2seq[base] for base in seq]) for seq in batch.get('sequence').detach().tolist()]
        self.test_results['structure'] += predictions['structure'].tolist()

        predictions = self._clean_predictions(batch, predictions)
        batch.integrate_prediction(predictions)

    def on_test_batch_end(
        self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        # push the metric directly
        metric_pack = MetricsStack(
            name=TEST_SETS_NAMES[dataloader_idx],
            data_type=self.data_type_output,
            )
        for dt, metrics in metric_pack.update(batch).compute().items():
            for name, metric in metrics.items():
                self.log(
                    f"test/{metric_pack.name}/{dt}/{name}",
                    float(metric),
                    add_dataloader_idx=False,
                    batch_size=len(batch),
                )
        if batch_idx % 100 == 0:
            torch.cuda.empty_cache()

    def on_test_epoch_end(self) -> None:
        torch.cuda.empty_cache()
        
    def on_test_end(self) -> None:
        
        import pandas as pd
        df = pd.DataFrame(self.test_results)
        df.to_feather('test_results_PT+FT.feather')

        torch.cuda.empty_cache()

    def predict_step(self, batch: Batch, batch_idx: int):
        predictions = self.forward(batch)
        predictions = self._clean_predictions(batch, predictions)
        batch.integrate_prediction(predictions)

    def seq2oneHot(self, seq):
        one_hot_embed = torch.zeros((5, 4), device=self.device)
        one_hot_embed[1:] = torch.eye(4)

        return one_hot_embed[seq].type(torch.long)
