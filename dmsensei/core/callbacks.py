from typing import Any
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
import numpy as np
from torch import Tensor, tensor
import wandb
from .metrics import f1, r2_score
from .visualisation import plot_factory

# from lightning.pytorch.utilities import _zero_only
import os
import pandas as pd
from rouskinhf import int2seq
import plotly.graph_objects as go
from lightning.pytorch.utilities import rank_zero_only

from ..config import (
    TEST_SETS_NAMES,
    REF_METRIC_SIGN,
    REFERENCE_METRIC,
    DATA_TYPES_TEST_SETS,
    POSSIBLE_METRICS,
)
from .metrics import metric_factory

from ..core.datamodule import DataModule
from . import metrics
from .loader import Loader
from os.path import join
from .logger import Logger
import pandas as pd
from lightning.pytorch import Trainer
from lightning.pytorch import LightningModule
from kaggle.api.kaggle_api_extended import KaggleApi
import pickle
from .batch import Batch
from .listofdatapoints import ListOfDatapoints
from typing import Union


class LoadBestModel(pl.Callback):
    
    def __init__(self) -> None:
        self.stage = None
        
    def load_model(self, pl_module:LightningModule, model_file:str=None): 
        if model_file is None:
            return
        if model_file == 'best':
            loader = Loader.find_best_model(wandb.run.name)
            if loader is None:
                raise ValueError("Stage:{}, No best model found for this run.".format(self.stage))
        else:
            loader = Loader(path=model_file)
        
        # Load the model
        weights = loader.load_from_weights(safe_load=True)
        if weights is not None:
            pl_module.load_state_dict(weights)
            print(f"Loaded model on {self.stage}: {loader.get_name()}")
            return self.model_file.split("/")[-1]
        

class WandbFitLogger(LoadBestModel):
    def __init__(
        self,
        dm: DataModule,
        batch_size: int,
        load_model:str = None,
    ):
        """
        load_model: path to the model to load. None if no model to load.
        """
        self.stage = "fit"
        self.data_type = dm.data_type
        self.batch_size = batch_size
        self.model_file = load_model
        self.val_losses = []
        self.dm = dm
        self.best_loss = np.inf
        self.validation_examples_references = {d:None for d in self.data_type}
        
    def on_start(self, trainer: Trainer, pl_module: LightningModule):
        self.load_model(pl_module, self.model_file)
        
    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs,
        batch: Any,
        batch_idx: int,
    ) -> None:
        # Log the train loss
        loss = outputs["loss"]
        logger = Logger(pl_module, self.batch_size)
        logger.train_loss(torch.sqrt(loss).item())

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        pass

    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch: Batch,
        batch_idx,
        dataloader_idx=0,
    ):
        ### LOG ###
        # Log loss to Wandb
        logger = Logger(pl_module, self.batch_size)
        loss = outputs
        # Dataloader_idx is 0 for the validation set
        # The other dataloader_idx are for complementary validation sets
        logger.valid_loss(torch.sqrt(loss).item(), dataloader_idx)
        # Save val_loss for evaluating if this model is the best model
        self.val_losses.append(loss)

        # Compute metrics and log them to Wandb.
        metrics = batch.compute_metrics()
        logger.error_metrics_pack("valid", metrics)
        
        # ### END LOG ###

        # ----------------------------------------------------------------------------------------------

        # #### PLOT ####
        
        # For multi-gpu training. Only plot for the validation set
        if rank_zero_only.rank or dataloader_idx:
            return
        
        # plot one example per epoch and per data_type
        for data_type, name in plot_factory.keys():
            # INITIALIZATION: find one reference per data_type
            if data_type in self.data_type and self.validation_examples_references[data_type] is None:
                if batch.contains(data_type):
                    idx = batch.get_index(data_type)[0]
                    self.validation_examples_references[data_type] = batch.get(
                        "reference", index=idx
                    )
            
            # Check if the batch contains the examples 
            if not batch.contains(data_type):
                continue
            if not self.validation_examples_references[data_type] in batch.get(
                "reference"
            ):
                continue
            
            # If so, plot the example
            idx = batch.get("reference").index(
                self.validation_examples_references[data_type]
            )
            pred, true = batch.get(data_type, pred=True, true=True, index=idx)
            plot = plot_factory[(data_type, name)](
                pred=pred,
                true=true,
                sequence=batch.get("sequence", index=idx),
                reference=batch.get("reference", index=idx),
                length=batch.get("length", index=idx),
            )
            
            # Log the plot to Wandb
            logger.valid_plot(data_type, name, plot)
            
        #### END PLOT ####

    @rank_zero_only
    def on_validation_end(self, trainer: Trainer, pl_module, dataloader_idx=0):

        if dataloader_idx:
            return

        # Save best model 
        this_epoch_loss = torch.mean(torch.tensor(self.val_losses)).item()
        if this_epoch_loss < self.best_loss:
            # save the best model per integer MAE
            self.best_loss = this_epoch_loss
            name = "{}_MAE{}.pt".format(wandb.run.name, np.ceil(this_epoch_loss))
            loader = Loader(path='models/'+name)
            # logs what MAE it corresponds to
            loader.dump(pl_module).write_in_log(name, np.round(this_epoch_loss, 3))
            
        self.val_losses = []
        
class WandbTestLogger(LoadBestModel):
    def __init__(
        self,
        dm: DataModule,
        n_best_worst: int = 10,
        load_model:str = None,
    ):
        """
        load_model: path to the model to load. None if no model to load. 'best' to load the best model from wandb run.
        """
        self.dm = dm
        self.batch_size = dm.batch_size
        self.stage = "test"
        self.n_best_worst = n_best_worst
        self.test_stacks = [[] for _ in TEST_SETS_NAMES]
        self.model_file = load_model
        
    @rank_zero_only
    def on_test_start(self, trainer, pl_module):
        self.load_model(pl_module, self.model_file)

    @rank_zero_only
    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch: Batch, batch_idx, dataloader_idx=0
    ):
        logger = Logger(pl_module, batch_size=self.batch_size)

        # Compute metrics and log them to Wandb for each test set
        metrics = batch.compute_metrics()
        logger.error_metrics_pack(
            "test/{}".format(TEST_SETS_NAMES[dataloader_idx]), metrics
        )
        
        # Log the scores        
        data_type = DATA_TYPES_TEST_SETS[dataloader_idx]  
        preds, trues = batch.get_pairs(data_type) 
        for pred, true, idx in zip(preds, trues, batch.get_index(data_type)):
            metric = metric_factory[REFERENCE_METRIC[data_type]](pred=pred, true=true)
            self.test_stacks[dataloader_idx].append((batch.get("reference", index=idx), metric))

    @rank_zero_only
    def on_test_end(self, trainer, pl_module):
        logger = Logger(pl_module, batch_size=self.batch_size)

        for dataloader_idx, data_type in enumerate(DATA_TYPES_TEST_SETS):
                
            # Find the best and worst examples
            df = pd.DataFrame(
                self.test_stacks[dataloader_idx],
                columns=["reference", "score"],
            )
            if not len(df):
                continue
            df.sort_values(
                by="score", inplace=True, ascending=REF_METRIC_SIGN[data_type] < 0
            )
            refs = set(df["reference"].values[: self.n_best_worst]).union(
                set(df["reference"].values[-self.n_best_worst :])
            )
            
            # Useful to find the score for a reference
            mid_score = df["score"].values[len(df) // 2]
            df.set_index("reference", inplace=True)
            best_worse_suffix = lambda ref: (
                "best" if df.loc[ref, "score"] > mid_score else "worst"
            )
            
            # retrieve examples datapoints in the test set
            list_of_datapoints = ListOfDatapoints()
            for dp in self.dm.test_sets[dataloader_idx]:
                if dp.get("reference") in refs:
                    list_of_datapoints = (
                        list_of_datapoints + dp
                    )  # lowkey flex on the __add__ method

            # compute predictions for the examples
            batch = Batch.from_list_of_datapoints(
                list_of_datapoints, ["sequence", data_type]
            )
            prediction = pl_module(batch.get("sequence"))
            batch.integrate_prediction(prediction)

            # plot the examples and log them into wandb
            for dp in batch.to_list_of_datapoints():
                for plot_data_type, plot_name in plot_factory.keys():
                    if plot_data_type != data_type:
                        continue

                    # generate plot
                    plot = plot_factory[(plot_data_type, plot_name)](
                        pred=dp.get(plot_data_type, pred=True, true=False),
                        true=dp.get(plot_data_type, pred=False, true=True),
                        sequence=dp.get("sequence"),
                        reference=dp.get("reference"),
                        length=dp.get("length"),
                    )  # add arguments here if you want

                    # log plot
                    logger.test_plot(
                        dataloader=TEST_SETS_NAMES[dataloader_idx],
                        data_type=data_type,
                        name=plot_name + "_" + best_worse_suffix(dp.get("reference")),
                        plot=plot,
                    )


class KaggleLogger(LoadBestModel):
    def __init__(self, load_model:str = None, push_to_kaggle:bool=True) -> None:
        """
        load_model: path to the model to load. None if no model to load. 'best' to load the best model from wandb run.
        """
        self.stage = "predict"
        self.pred_dms = []
        self.pred_shape = []
        self.pred_ref = []
        self.push_to_kaggle = push_to_kaggle
        self.model_file = load_model

    @rank_zero_only
    def on_predict_start(self, trainer, pl_module):
        self.load_model(self.model_file)
    
    @rank_zero_only
    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        
        self.pred_dms.extend(batch.get("dms", pred=True, true=False).cpu().tolist())
        self.pred_shape.extend(batch.get("shape", pred=True, true=False).cpu().tolist())
        self.pred_ref.extend(batch.get("reference"))

    @rank_zero_only
    def on_predict_end(self, trainer, pl_module):

        # load data
        df = pd.DataFrame({"reference": self.pred_ref, "dms": self.pred_dms, "shape": self.pred_shape})

        # sort by reference
        sequence_ids = pd.read_csv(
            os.path.join(
                os.path.dirname(__file__), "../resources/test_sequences_ids.csv"
            )
        )
        df = pd.merge(sequence_ids, df, on="reference")
        assert len(df) == len(sequence_ids), "Not all sequences were predicted. Are you sure you are using ?"

        # save predictions as csv
        dms = np.concatenate(df["dms"].values)
        shape = np.concatenate(df["shape"].values)
        pd.DataFrame(
            {"reactivity_DMS_MaP": dms, "reactivity_2A3_MaP": shape}
        ).reset_index().rename(columns={"index": "id"}).to_csv(
            "predictions.csv", index=False
        )

        # upload to kaggle
        if self.push_to_kaggle:
            api = KaggleApi()
            api.authenticate()
            api.competition_submit(
                file_name="predictions.csv",
                message=self.message,
                competition="stanford-ribonanza-rna-folding",
            )


class ModelChecker(pl.Callback):
    def __init__(self, model, log_every_nstep=1000):
        self.step_number = 0
        self.model = model
        self.log_every_nstep = log_every_nstep

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.step_number % self.log_every_nstep == 0:
            # Get all parameters
            params = []
            for param in pl_module.parameters():
                params.append(param.view(-1))
            params = torch.cat(params).cpu().detach().numpy()

            # Compute histogram
            if rank_zero_only.rank == 0 and self.model in ["mlp"]:
                wandb.log({"model_params": wandb.Histogram(params)})

        self.step_number += 1
