import pytorch_lightning as pl
import torch
import numpy as np
from torch import Tensor, tensor
import wandb
from .metrics import compute_f1, r2_score
from .visualisation import plot_structure, plot_dms
from pytorch_lightning.utilities import rank_zero_only
import os
import pandas as pd
from rouskinhf import int2seq
import plotly.graph_objects as go
from ..config import TEST_SETS_NAMES


class PredictionLogger(pl.Callback):
    def __init__(self, data, n_best=5, wandb_log=True):
        self.wandb_log = wandb_log
        self.data = data

        self.n_best = n_best

        self.test_examples = {
            "seq": [],
            "pred": [],
            "true": [],
            "score": [],
        }  # Predictions on test set
        self.valid_examples = {
            "seq": [],
            "pred": [],
            "true": [],
            "score": [],
        }  # Predictions on one element of the valid set over epochs

        self.best_score = -np.inf

        self.model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "models", "trained_models"
        )

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        features, labels = batch
        predictions = outputs[0]

        # Only works because valid dataset is not shuffled !! -> should use indices retrieval
        if batch_idx == 0:
            len_seq = torch.count_nonzero(features[0]).item()

            self.valid_examples["seq"].append(features[0].cpu()[:len_seq])

            if self.data == "dms":
                self.valid_examples["pred"].append(predictions[0].cpu()[:len_seq])
                self.valid_examples["true"].append(labels[0].cpu()[:len_seq])
                self.valid_examples["score"].append(r2_score(labels[0], predictions[0]))

            else:
                self.valid_examples["pred"].append(
                    predictions[0].cpu()[:len_seq, :len_seq]
                )
                self.valid_examples["true"].append(
                    labels[0].cpu()[:len_seq, :len_seq].bool()
                )
                self.valid_examples["score"].append(
                    compute_f1(labels[0], predictions[0])
                )

    def on_validation_end(self, trainer, pl_module):
        # Save best model
        if rank_zero_only.rank == 0:
            if self.data.lower() == "dms":
                score = trainer.logged_metrics["valid/r2"].item()
            else:
                score = trainer.logged_metrics["valid/mFMI"].item()

            if score > self.best_score:
                self.best_score = score
                os.makedirs(self.model_path, exist_ok=True)
                torch.save(
                    pl_module.state_dict(),
                    os.path.join(
                        self.model_path, trainer.logger.experiment.name + ".pt"
                    ),
                )

            # Log a random example from the validation set. X is the true data, Y is the prediction, r2 is the score
            idx = np.random.randint(len(self.valid_examples["seq"]))
            fig = plot_dms(
                self.valid_examples["true"][idx],
                self.valid_examples["pred"][idx],
                r2=self.valid_examples["score"][idx],
                layout="scatter",
            )
            wandb.log(
                {
                    "valid/example": wandb.Image(fig),
                }
            )

    def on_test_start(self, trainer, pl_module):
        if not self.wandb_log or rank_zero_only.rank != 0:
            return

        # Load best model
        pl_module.load_state_dict(
            torch.load(
                os.path.join(self.model_path, trainer.logger.experiment.name + ".pt")
            )
        )

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        features, labels = batch
        predictions = outputs

        for i in range(predictions.shape[0]):
            len_seq = torch.count_nonzero(features[i]).item()

            self.test_examples["seq"].append(features[i].cpu()[:len_seq].numpy())

            if self.data == "dms":
                self.test_examples["pred"].append(predictions[i].cpu()[:len_seq])
                self.test_examples["true"].append(labels[i].cpu()[:len_seq])
                self.test_examples["score"].append(
                    r2_score(predictions[i], labels[i], features[i])
                )

            else:
                self.test_examples["pred"].append(
                    predictions[i].cpu()[:len_seq, :len_seq].numpy()
                )
                self.test_examples["true"].append(
                    labels[i].cpu()[:len_seq, :len_seq].bool().numpy()
                )
                self.test_examples["score"].append(
                    compute_f1(predictions[i], labels[i])
                )

    def on_test_end(self, trainer, pl_module):
        if not self.wandb_log or rank_zero_only.rank != 0:
            return

        self.log_valid_example()

        ## Log best and worst predictions of test dataset ##

        for key in self.test_examples.keys():
            self.test_examples[key] = np.array(self.test_examples[key], dtype=object)

        # Select best and worst predictions
        idx_sorted = np.argsort(self.test_examples["score"])[::-1]
        best_worst_idx = np.concatenate(
            (idx_sorted[-self.n_best :], idx_sorted[: self.n_best])
        )

        sequences = self.test_examples["seq"][best_worst_idx]
        true_outputs = self.test_examples["true"][best_worst_idx]
        pred_outputs = self.test_examples["pred"][best_worst_idx]
        scores = self.test_examples["score"][best_worst_idx]

        # Plot best and worst predictions
        fig_list = []
        for true_output, pred_output in zip(true_outputs, pred_outputs):
            if self.data == "dms":
                fig_list.append(plot_dms(true_output, pred_output))
            else:
                fig_list.append(plot_structure(true_output, pred_output))

        # Make list of figures into a wandb table
        df = pd.DataFrame(
            {
                "seq": [
                    "".join([int2seq[char] for char in seq[seq != 0]])
                    for seq in sequences
                ],
                "structures": fig_list,
                "score": scores,
            }
        )

        wandb.log({"final/best_worst": wandb.Table(dataframe=df)})

        ## Log correlation of F1 score and sequence lengths ##

        fig = go.Figure(
            data=go.Scatter(
                x=[len(seq[seq != 0]) for seq in self.test_examples["seq"]],
                y=self.test_examples["score"],
                mode="markers",
            )
        )
        score_type = "R2 score" if self.data == "dms" else "F1 score"
        fig.update_layout(
            title=f'Sequence lenght vs {score_type}. Mean score: {self.test_examples["score"].mean():.2f}',
            xaxis_title="Sequence length",
            yaxis_title=score_type,
        )

        wandb.log({"final/score_vs_lenght": fig})

    def log_valid_example(self):
        fig_list = []
        for true_output, pred_output in zip(
            self.valid_examples["true"], self.valid_examples["pred"]
        ):
            if self.data == "dms":
                fig_list.append(plot_dms(true_output, pred_output))
            else:
                fig_list.append(plot_structure(true_output, pred_output))

        df = pd.DataFrame(
            {"score": self.valid_examples["score"], "structures": fig_list}
        )

        wandb.log({"final/example": wandb.Table(dataframe=df)})


class ModelChecker(pl.Callback):
    def __init__(self, log_every_nstep=1000):
        self.step_number = 0

        self.log_every_nstep = log_every_nstep

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.step_number % self.log_every_nstep == 0:
            # Get all parameters
            params = []
            for param in pl_module.parameters():
                params.append(param.view(-1))
            params = torch.cat(params).cpu().detach().numpy()

            # Compute histogram
            if rank_zero_only.rank == 0:
                wandb.log({"model_params": wandb.Histogram(params)})

        self.step_number += 1
