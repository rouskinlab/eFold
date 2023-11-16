from typing import Any
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
import numpy as np
from torch import Tensor, tensor
import wandb
from .metrics import f1, r2_score
from .visualisation import plot_factory
from lightning.pytorch.utilities import rank_zero_only
import os
import pandas as pd
from rouskinhf import int2seq
import plotly.graph_objects as go
from ..config import (
    TEST_SETS_NAMES,
    REF_METRIC_SIGN,
    REFERENCE_METRIC,
    DATA_TYPES,
    POSSIBLE_METRICS,
)
from .metrics import metric_factory

from ..core.datamodule import DataModule
from . import metrics
from .loader import Loader
from os.path import join
from . import logger
from ..util.stack import Stack
import pandas as pd
from lightning.pytorch import Trainer
from lightning.pytorch import LightningModule
from kaggle.api.kaggle_api_extended import KaggleApi
import pickle
from .batch import Batch
from .listofdatapoints import ListOfDatapoints


class MyWandbLogger(pl.Callback):
    def __init__(
        self,
        dm: DataModule,
        model: LightningModule,
        batch_size: int,
        n_best_worst: int = 10,
        wandb_log: bool = True,
    ):
        # init
        self.wandb_log = wandb_log
        self.dm = dm
        self.data_type = dm.data_type
        self.model = model
        self.logger = logger.Logger(model, batch_size)

        # validation
        # TODO: #12 the validation examples aren't in the validation set
        # self.validation_examples_refs = dm.find_one_reference_per_data_type(
        #     dataset_name="valid"
        # )
        self.val_losses = []
        self.batch_scores = {d: {} for d in self.data_type}
        self.best_score = {d: -torch.inf * REF_METRIC_SIGN[d] for d in self.data_type}
        self.validation_examples_references = {
            data_type: None for data_type in self.data_type
        }
        # testing
        self.n_best_worst = n_best_worst
        self.test_stacks = [
            {
                d: {
                    "best": Stack(L=n_best_worst, mode="best", data_type=d),
                    "worse": Stack(L=n_best_worst, mode="worse", data_type=d),
                }
                for d in self.data_type
            }
            for _ in TEST_SETS_NAMES
        ]

        self.test_start_buffer = [
            {d: [] for d in self.data_type} for _ in TEST_SETS_NAMES
        ]

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs,
        batch: Any,
        batch_idx: int,
    ) -> None:
        loss = outputs['loss']
        self.logger.train_loss(torch.sqrt(loss).item())

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        pass

    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch:Batch,
        batch_idx,
    ):
        ### LOG ###
        # Log loss to Wandb
        loss = outputs
        self.logger.valid_loss(torch.sqrt(loss).item())

        # Store scores for averaging by batch
        list_of_datapoints = batch.to_list_of_datapoints()
        # Compute metrics
        for dp in list_of_datapoints:
            dp.compute_error_metrics_pack()
            # save scores for averaging by epoch
            for data_type in dp.metrics.keys():
                metric = REFERENCE_METRIC[data_type]
                if metric not in self.batch_scores[data_type]:
                    self.batch_scores[data_type][metric] = []
                self.batch_scores[data_type][metric].append(
                    dp.read_reference_metric(data_type)
                )
        # Log metrics to Wandb
        self.logger.error_metrics_pack(stage="valid", list_of_datapoints=list_of_datapoints)
        ### END LOG ###

        #### PLOT ####
        # plot one example per epoch and per data_type
        # initialisation: choose one reference per data_type
        for data_type in self.data_type:
            if self.validation_examples_references[data_type] is None:
                for dp in self.dm.val_set:
                    if dp.contains(data_type):
                        self.validation_examples_references[
                            data_type
                        ] = dp.get('reference')
                        break
        # plot the examples
        for data_type, name in plot_factory.keys():
            if batch.contains(data_type) and self.validation_examples_references[
                data_type
            ] in set(batch.get('reference')):
                idx = batch.get('reference').index(
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
                self.logger.valid_plot(data_type, name, plot)
        #### END PLOT ####

    def on_validation_end(self, trainer, pl_module, dataloader_idx=0):
        # Save best model
        loader = Loader()
        for data_type, scores in self.batch_scores.items():
            # if best metric, save metric as best
            average_score = np.nanmean(scores[REFERENCE_METRIC[data_type]])
            # The definition of best depends on the metric
            if (
                average_score * REF_METRIC_SIGN[data_type]
                > self.best_score[data_type] * REF_METRIC_SIGN[data_type]
            ):
                self.best_score[data_type] = average_score
                self.logger.final_score(average_score, data_type)
                # save model for the best r2
                if (
                    data_type == "dms" and rank_zero_only.rank == 0
                ):  # only keep best model for dms
                    loader.dump(self.model)

        self.batch_scores = {d: {} for d in self.data_type}

    def on_test_start(self, trainer, pl_module):
        if not self.wandb_log or rank_zero_only.rank != 0:
            return

        loader = Loader()

        # Load best model for testing
        weights = loader.load_from_weights(safe_load=True)
        if weights is not None:
            pl_module.load_state_dict(weights)

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch:Batch, batch_idx, dataloader_idx=0
    ):
        # compute scores
        list_of_datapoints = batch.to_list_of_datapoints()
        for dp in list_of_datapoints:
            dp.compute_error_metrics_pack()

        # Logging metrics to Wandb
        self.logger.error_metrics_pack("test/{}".format(TEST_SETS_NAMES[dataloader_idx]), list_of_datapoints)

        # stack best and worse predictions
        for data_type in self.data_type:
            # wait to have a full buffer to start stacking
            if (
                len(self.test_start_buffer[dataloader_idx][data_type])
                < 2 * self.n_best_worst
            ):
                self.test_start_buffer[dataloader_idx][data_type].extend(
                    dp for dp in list_of_datapoints if data_type in dp.metrics.keys()
                )
            else:
                # initialise the stacks here
                if (
                    self.test_stacks[dataloader_idx][data_type]["best"].is_empty()
                    and self.test_stacks[dataloader_idx][data_type]["worse"].is_empty()
                ):
                    merged_buffer_and_current_lines = (
                        list_of_datapoints.copy()
                        + self.test_start_buffer[dataloader_idx][data_type]
                    )
                    merged_buffer_and_current_lines.sort(
                        key=lambda dp: dp.read_reference_metric(data_type) if dp.read_reference_metric(data_type) is not None else -np.inf,
                    )
                    for idx in range(self.n_best_worst):
                        self.test_stacks[dataloader_idx][data_type]["worse"].try_to_add(
                            merged_buffer_and_current_lines[idx]
                        )
                        self.test_stacks[dataloader_idx][data_type]["best"].try_to_add(
                            merged_buffer_and_current_lines[-idx - 1]
                        )

                # when initialisation is done, start stacking here
                else:
                    for line in list_of_datapoints:
                        if not self.test_stacks[dataloader_idx][data_type][
                            "best"
                        ].try_to_add(line):
                            self.test_stacks[dataloader_idx][data_type][
                                "worse"
                            ].try_to_add(line)

    def on_test_end(self, trainer, pl_module):
        if not self.wandb_log or rank_zero_only.rank != 0:
            return

        for dataloader_idx in range(len(TEST_SETS_NAMES)):
            for data_type in self.data_type:
                
                def extract(dp, data_type):
                    return {
                        "sequence": dp.get("sequence"),
                        "reference": dp.get("reference"),
                        "length": dp.get("length"),
                        "true_{}".format(data_type): dp.get(data_type),
                        "pred_{}".format(data_type): dp.get(data_type, pred=True, true=False),
                        "score_{}".format(data_type): dp.read_reference_metric(data_type),
                    }
                
                for rank in ['best', 'worse']:
                    df_examples = pd.DataFrame(
                        [extract(dp, data_type) for dp in self.test_stacks[dataloader_idx][data_type][rank].vals] 
                    )

                    if len(df_examples) == 0:
                        continue
                    ## Log best and worst predictions of test dataset ##
                    df_examples.sort_values(by=["score_{}".format(data_type)], inplace=True)

                    true_outputs = df_examples["true_{}".format(data_type)].values
                    pred_outputs = df_examples["pred_{}".format(data_type)].values
                    sequences = df_examples["sequence"].values
                    references = df_examples["reference"].values
                    lengths = df_examples["length"].values

                    # Plot best and worst predictions
                    for true, pred, sequence, reference, length in zip(
                        true_outputs, pred_outputs, sequences, references, lengths
                    ):
                        for plot_data_type, plot_name in plot_factory.keys():
                            if plot_data_type != data_type:
                                continue
                            plot = plot_factory[(plot_data_type, plot_name)](
                                pred=pred,
                                true=true,
                                sequence=sequence,
                                reference=reference,
                                length=length,
                            )  # add arguments here if you want
                            self.logger.test_plot(
                                dataloader=TEST_SETS_NAMES[dataloader_idx],
                                data_type=data_type,
                                name=plot_name+ '_' + rank,
                                plot=plot,
                            )


class KaggleLogger(pl.Callback):
    def __init__(self, dm, push_to_kaggle=True) -> None:
        # prediction
        self.predictions = []
        self.dm = dm
        self.push_to_kaggle = push_to_kaggle

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        for dp in batch.to_list_of_datapoints():   
            d = {'reference': dp.get('reference')}
            for data_type in self.dm.data_type:
                d[data_type] = dp.get(data_type, pred=True, true=False, to_numpy=True)
            self.predictions.append(d)

    def on_predict_end(self, trainer, pl_module):
        # load data
        df = pd.DataFrame(self.predictions)

        sequence_ids = pd.read_csv(
            os.path.join(
                os.path.dirname(__file__), "../resources/test_sequences_ids.csv"
            )
        )
        df = pd.merge(sequence_ids, df, on="reference")

        # save predictions as csv
        dms, shape = np.concatenate(df["dms"].values), np.concatenate(
            df["shape"].values
        )
        dms, shape = np.clip(dms, 0, 1), np.clip(shape, 0, 1)
        pd.DataFrame(
            {"reactivity_DMS_MaP": dms, "reactivity_2A3_MaP": shape}
        ).reset_index().rename(columns={"index": "id"}).to_csv(
            "predictions.csv", index=False
        )

        # save predictions as pickle
        pickle.dump(df, open("predictions.pkl", "wb"))

        # upload to kaggle
        if self.push_to_kaggle:
            api = KaggleApi()
            api.authenticate()
            api.competition_submit(
                file_name="predictions.csv",
                message="from predict callback",
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


def compute_error_metrics_pack(batch:Batch):
    pred, true = batch.get_pairs("dms")
    return {
        data_type: {
            metric_name: metric_factory[metric_name](
                pred=pred,
                true=true,
                batch=len(batch.get_index(data_type)) > 1,
            )
            for metric_name in POSSIBLE_METRICS[data_type]
        }
        for data_type in DATA_TYPES
        if batch.contains(data_type)
    }
