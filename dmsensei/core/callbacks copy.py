import lightning.pytorch as pl
import torch
import numpy as np
from torch import Tensor, tensor
import wandb
from .metrics import f1, r2_score
from .visualisation import plot_structure, plot_dms, plot_dms_padding
from lightning.pytorch.utilities import rank_zero_only
import os
import pandas as pd
from rouskinhf import int2seq
import plotly.graph_objects as go
from ..config import TEST_SETS_NAMES
from .metrics import pearson_coefficient, f1, r2_score, mFMI


class PredictionLogger(pl.Callback):
    def __init__(self, data, n_best_worst=5, wandb_log=True):
        self.wandb_log = wandb_log
        self.data_type = data
        self.best_r2 = -np.inf

        self.n_best_worst = n_best_worst

        self.test_examples = [
            {
                "seq": [],
                "pred": [],
                "true": [],
                "score": [],
            }
            for _ in range(len(TEST_SETS_NAMES[self.data_type]))
        ]  # Predictions on test set
        self.valid_examples = [
            {
                "seq": [],
                "pred": [],
                "true": [],
                "score": [],
            }
            for _ in range(len(TEST_SETS_NAMES[self.data_type]))
        ]  # Predictions on one element of the valid set over epochs

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

            self.valid_examples[dataloader_idx]["seq"].append(
                features[0].cpu()[:len_seq]
            )

            if self.data_type == "dms":
                self.valid_examples[dataloader_idx]["pred"].append(
                    predictions[0].cpu()[:len_seq]
                )
                self.valid_examples[dataloader_idx]["true"].append(
                    labels[0].cpu()[:len_seq]
                )
                self.valid_examples[dataloader_idx]["score"].append(
                    r2_score(labels[0], predictions[0])
                )

            else:
                self.valid_examples[dataloader_idx]["pred"].append(
                    predictions[0].cpu()[:len_seq, :len_seq]
                )
                self.valid_examples[dataloader_idx]["true"].append(
                    labels[0].cpu()[:len_seq, :len_seq].bool()
                )
                self.valid_examples[dataloader_idx]["score"].append(
                    f1(labels[0], predictions[0])
                )

    def on_validation_end(self, trainer, pl_module, dataloader_idx=0):
        # Save best model
        if rank_zero_only.rank == 0:
            if self.data_type.lower() == "dms":
                score = trainer.logged_metrics["valid/r2"].item()
            else:
                score = trainer.logged_metrics["valid/mFMI"].item()

            if score > self.best_score:
                self.best_score = score
                os.makedirs(self.model_path, exist_ok=True)
                self.best_model_path = os.path.join(
                    self.model_path, trainer.logger.experiment.name + ".pt"
                )
                torch.save(
                    pl_module.state_dict(),
                    self.best_model_path,
                )
                if self.data_type.lower() == "dms":
                    wandb.log({"final/best_r2": score})

            # Log a random example from the validation set. X is the true data, Y is the prediction, r2 is the score
            fig = plot_dms(
                self.valid_examples[dataloader_idx]["true"][-1],
                self.valid_examples[dataloader_idx]["pred"][-1],
                r2=self.valid_examples[dataloader_idx]["score"][-1],
                layout="scatter",
            )
            wandb.log(
                {
                    "valid/example": wandb.Image(fig),
                }
            )

            # plot the embedding
            wandb.log(
                {
                    "valid/padding": wandb.Image(
                        plot_dms_padding(
                            self.valid_examples[dataloader_idx]["true"][-1],
                            self.valid_examples[dataloader_idx]["pred"][-1],
                        )
                    )
                }
            )

    # def on_test_start(self, trainer, pl_module):
    #     if not self.wandb_log or rank_zero_only.rank != 0:
    #         return

    #     # Load best model
    #     pl_module.load_state_dict(
    #         torch.load(
    #             os.path.join(self.model_path, trainer.logger.experiment.name + ".pt")
    #         )
    #     )

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        features, labels = batch
        predictions = outputs

        for i in range(predictions.shape[0]):
            len_seq = torch.count_nonzero(features[i]).item()

            self.test_examples[dataloader_idx]["seq"].append(
                features[i].cpu()[:len_seq].numpy()
            )

            if self.data_type == "dms":
                self.test_examples[dataloader_idx]["pred"].append(
                    predictions[i].cpu()[:len_seq].numpy()
                )
                self.test_examples[dataloader_idx]["true"].append(
                    labels[i].cpu()[:len_seq].numpy()
                )
                self.test_examples[dataloader_idx]["score"].append(
                    r2_score(labels[i], predictions[i])
                )

            else:
                self.test_examples[dataloader_idx]["pred"].append(
                    predictions[i].cpu()[:len_seq, :len_seq].numpy()
                )
                self.test_examples[dataloader_idx]["true"].append(
                    labels[i].cpu()[:len_seq, :len_seq].bool().numpy()
                )
                self.test_examples[dataloader_idx]["score"].append(
                    f1(predictions[i], labels[i])
                )

    def on_test_end(self, trainer, pl_module):
        if not self.wandb_log or rank_zero_only.rank != 0:
            return

        for dataloader_idx in range(len(TEST_SETS_NAMES[self.data_type])):
            test_examples = self.test_examples[dataloader_idx]
            test_set_name = TEST_SETS_NAMES[self.data_type][dataloader_idx]

            # self.log_valid_example()

            ## Log best and worst predictions of test dataset ##

            for key in test_examples.keys():
                test_examples[key] = np.array(test_examples[key], dtype=object)

            # Select best and worst predictions
            idx_sorted = np.argsort(test_examples["score"])
            best_worst_idx = np.concatenate(
                (idx_sorted[: self.n_best_worst], idx_sorted[-self.n_best_worst :])
            )

            sequences = test_examples["seq"][best_worst_idx]
            true_outputs = test_examples["true"][best_worst_idx]
            pred_outputs = test_examples["pred"][best_worst_idx]
            scores = test_examples["score"][best_worst_idx]

            # Plot best and worst predictions
            fig_list = []
            for true_output, pred_output, score in zip(
                true_outputs, pred_outputs, scores
            ):
                if self.data_type == "dms":
                    fig_list.append(
                        plot_dms(true_output, pred_output, score, layout="scatter")
                    )
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

            wandb.log({f"test/{test_set_name}/best_worst": wandb.Table(dataframe=df)})

            ## Log scatter of F1 score and sequence lengths ##

            fig = go.Figure(
                data=go.Scatter(
                    x=[len(seq[seq != 0]) for seq in test_examples["seq"]],
                    y=test_examples["score"],
                    mode="markers",
                )
            )
            score_type = "R2 score" if self.data_type == "dms" else "F1 score"
            fig.update_layout(
                title=f'Sequence lenght vs {score_type}. Mean score: {test_examples["score"].mean():.2f} ± {test_examples["score"].std():.2f} (1 std)',
                xaxis_title="Sequence length",
                yaxis_title=score_type,
            )

            wandb.log({f"test/{test_set_name}/score_vs_lenght": fig})

            ## Log scatter of F1 score vs pearson ##
            if self.data_type == "dms":
                fig = go.Figure(
                    data=go.Scatter(
                        # x=[len(seq[seq != 0]) for seq in test_examples["seq"]],
                        x=[
                            pearson_coefficient(
                                torch.tensor(y_true.astype(float)),
                                torch.tensor(y_pred.astype(float)),
                            )
                            for y_true, y_pred in zip(
                                test_examples["true"], test_examples["pred"]
                            )
                        ],
                        y=test_examples["score"],
                        mode="markers",
                    )
                )
                fig.update_layout(
                    title=f"Pearson vs R2 score",
                    xaxis_title="Pearson",
                    yaxis_title="R2 score",
                )

                wandb.log({f"test/{test_set_name}/r2_vs_pearson": fig})

        # plot the whole validation set
        # figs = []
        # for true_output, pred_output in zip(self.valid_examples["true"], self.valid_examples["pred"]):
        #     if self.data_type == "dms":
        #         figs.append(plot_dms(true_output, pred_output))
        #     else:
        #         figs.append(plot_structure(true_output, pred_output))
        # wandb.log({"valid/whole_set": wandb.Image(figs)})

    # def log_valid_example(self):
    #     fig_list = []
    #     for true_output, pred_output in zip(
    #         self.valid_examples["true"], self.valid_examples["pred"]
    #     ):
    #         if self.data_type == "dms":
    #             fig_list.append(plot_dms(true_output, pred_output))
    #         else:
    #             fig_list.append(plot_structure(true_output, pred_output))

    #     df = pd.DataFrame(
    #         {"score": self.valid_examples["score"], "structures": fig_list}
    #     )

    #     wandb.log({"final/example": wandb.Table(dataframe=df)})


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
