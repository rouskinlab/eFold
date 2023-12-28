import os
from lightning import LightningModule, Trainer
import lightning.pytorch as pl
import torch
import numpy as np
import pandas as pd
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.utilities import rank_zero_only
from kaggle.api.kaggle_api_extended import KaggleApi
import wandb
from typing import Any

from .visualisation import plot_factory
from .metrics import metric_factory
from .datamodule import DataModule
from .loader import Loader
from .batch import Batch
from ..config import (
    TEST_SETS_NAMES,
    REF_METRIC_SIGN,
    REFERENCE_METRIC,
    DATA_TYPES_TEST_SETS,
    POSSIBLE_METRICS,
)
from .logger import Logger, LocalLogger


class LoadBestModel(pl.Callback):
    def __init__(self) -> None:
        self.stage = None

    def load_model(self, pl_module: LightningModule, model_file: str = None):
        if model_file is None:
            return
        if model_file == "best":
            loader = Loader.find_best_model(wandb.run.name)
            if loader is None:
                raise ValueError(
                    "Stage:{}, No best model found for this run.".format(self.stage)
                )
        else:
            loader = Loader(path=model_file)

        # Load the model
        weights = loader.load_from_weights(safe_load=False)
        if weights is not None:
            pl_module.load_state_dict(weights)
            print(f"Loaded model on {self.stage}: {loader.get_name()}")
            return self.model_file.split("/")[-1]


class WandbFitLogger(LoadBestModel):
    def __init__(
        self,
        dm: DataModule,
        batch_size: int = None,
        load_model: str = None,
        log_plots_every_n_epoch: int = 100000000,  # deactivated for now
    ):
        """
        load_model: path to the model to load. None if no model to load.
        """
        self.stage = "fit"
        self.data_type = dm.data_type
        self.batch_size = dm.batch_size if batch_size is None else batch_size
        self.model_file = load_model
        self.dm = dm
        self.log_plots_every_n_epoch = log_plots_every_n_epoch

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule):
        self.best_loss = np.inf
        self.val_losses = []
        self.validation_examples_references = {d: None for d in self.data_type}
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
        logger.train_loss(loss.item())

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        pass

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch: Batch, batch_idx, dataloader_idx=0
    ):
        return

        # ### END LOG ###

        # ----------------------------------------------------------------------------------------------

        # #### PLOT ####

        # For multi-gpu training. Only plot for the validation set
        if (
            (rank_zero_only.rank or dataloader_idx)
            and not trainer.current_epoch % self.log_plots_every_n_epoch
            or True
        ):  # TODO: remove True
            return

        # plot one example per epoch and per data_type
        for data_type, name in plot_factory.keys():
            # INITIALIZATION: find one reference per data_type
            if (
                data_type in self.data_type
                and self.validation_examples_references[data_type] is None
                and batch.contains(f"pred_{data_type}")
            ):
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
            pred, true = batch.get(f"pred_{data_type}", index=idx), batch.get(
                data_type, index=torch.where(batch.get("index_dms") == idx)[0][0]
            )
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
        if wandb.run is None:
            return

        name = "{}_epoch{}.pt".format(
            wandb.run.name,
            trainer.current_epoch
        )
        loader = Loader(path="models/" + name)
        # logs what MAE it corresponds to
        loader.dump(pl_module)
        

class WandbTestLogger(LoadBestModel):
    def __init__(
        self,
        dm: DataModule,
        n_best_worst: int = 10,
        load_model: str = None,
        local: bool = False,
    ):
        """
        load_model: path to the model to load. None if no model to load. 'best' to load the best model from wandb run.
        """
        self.dm = dm
        self.batch_size = dm.batch_size
        self.stage = "test"
        self.n_best_worst = n_best_worst
        self.model_file = load_model
        self.local = local

    @rank_zero_only
    def on_test_start(self, trainer, pl_module):
        self.test_stacks = [[] for _ in TEST_SETS_NAMES]
        self.refs = [[] for _ in TEST_SETS_NAMES]
        self.load_model(pl_module, self.model_file)

    @rank_zero_only
    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch: Batch, batch_idx, dataloader_idx=0
    ):
        if self.local:
            logger = LocalLogger()
        else:
            logger = Logger(pl_module, batch_size=self.batch_size)

        # Compute metrics and log them to Wandb for each test set
        metrics = batch.compute_metrics()
        logger.error_metrics_pack(
            "test/{}".format(TEST_SETS_NAMES[dataloader_idx]), metrics
        )

        # Log the scores
        data_type = DATA_TYPES_TEST_SETS[dataloader_idx]
        if not batch.contains(f"pred_{data_type}"):
            return
        preds, trues = batch.get_pairs(data_type)
        for idx, (pred, true) in enumerate(zip(preds, trues)):
            if pred is None or true is None:
                continue
            metric = metric_factory[REFERENCE_METRIC[data_type]](pred=pred, true=true)
            self.test_stacks[dataloader_idx].append(
                (batch.get("reference", index=idx), metric)
            )

    @rank_zero_only
    def on_test_end(self, trainer, pl_module):
        if self.local:
            logger = LocalLogger()
        else:
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
            list_of_datapoints = []
            for dp in self.dm.test_sets[dataloader_idx]:
                if dp["reference"] in refs:
                    list_of_datapoints.append(dp)

            # compute predictions for the examples
            batch = Batch.from_dataset_items(
                list_of_datapoints, [data_type], use_error=False
            )
            prediction = pl_module(batch)
            batch.integrate_prediction(prediction)

            # plot the examples and log them into wandb
            for idx in range(len(batch)):
                ref = batch.get("reference", index=idx)
                true = batch.get(data_type, index=idx)
                pred = batch.get(f"pred_{data_type}", index=idx)
                if ref not in refs or true is None or pred is None:
                    continue
                for plot_data_type, plot_name in plot_factory.keys():
                    if plot_data_type != data_type:
                        continue

                    # generate plot
                    plot = plot_factory[(plot_data_type, plot_name)](
                        pred=pred,
                        true=true,
                        sequence=batch.get("sequence", index=idx),
                        reference=ref,
                        length=batch.get("length", index=idx),
                    )  # add arguments here if you want

                    # log plot
                    logger.test_plot(
                        dataloader=TEST_SETS_NAMES[dataloader_idx],
                        data_type=data_type,
                        name=plot_name + "_" + best_worse_suffix(ref),
                        plot=plot,
                    )


class KaggleLogger(LoadBestModel):
    def __init__(self, load_model: str = None, push_to_kaggle: bool = True) -> None:
        """
        load_model: path to the model to load. None if no model to load. 'best' to load the best model from wandb run.
        """
        self.stage = "predict"
        self.push_to_kaggle = push_to_kaggle
        self.model_file = load_model

    @rank_zero_only
    def on_predict_start(self, trainer, pl_module):
        self.message = self.load_model(self.model_file)
        self.pred_dms = []
        self.pred_shape = []
        self.pred_ref = []

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
        dms = batch.get("pred_dms", to_numpy=True).tolist()
        shape = batch.get("pred_shape", to_numpy=True).tolist()
        lengths = batch.get("length")
        for l, s, d in zip(lengths, shape, dms):
            self.pred_dms.append(d[:l])
            self.pred_shape.append(s[:l])
        self.pred_ref.extend(batch.get("reference"))

    @rank_zero_only
    def on_predict_end(self, trainer, pl_module):
        # load data
        df = pd.DataFrame(
            {"reference": self.pred_ref, "dms": self.pred_dms, "shape": self.pred_shape}
        )

        # sort by reference
        sequence_ids = pd.read_csv(
            os.path.join(
                os.path.dirname(__file__), "../resources/test_sequences_ids.csv"
            )
        )
        df = pd.merge(sequence_ids, df, on="reference")
        assert len(df) == len(
            sequence_ids
        ), "Not all sequences were predicted. Are you sure you are using all of ribo-test?"

        # save predictions as csv
        dms = np.concatenate(df["dms"].values).round(4)
        shape = np.concatenate(df["shape"].values).round(4)
        pred = (
            pd.DataFrame({"reactivity_DMS_MaP": dms, "reactivity_2A3_MaP": shape})
            .reset_index()
            .rename(columns={"index": "id"})
        )

        pred.to_csv("predictions.csv", index=False)

        # zip predictions, return 1 if successful
        import subprocess

        use_zip = not subprocess.call(["zip", "predictions.zip", "predictions.csv"])

        assert (
            len(pred) == 269796671
        ), "predictions.csv should have 269796671 rows and has {}".format(len(pred))

        # upload to kaggle
        if self.push_to_kaggle:
            api = KaggleApi()
            api.authenticate()
            api.competition_submit(
                file_name="predictions.zip" if use_zip else "predictions.csv",
                message=self.message
                if self.message is not None
                else "pushed from KaggleLogger",
                competition="stanford-ribonanza-rna-folding",
            )

        # remove zip
        if use_zip:
            os.remove("predictions.zip")
