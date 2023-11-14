import wandb
from ..config import *


def train_loss(loss):
    wandb.log({"train/loss": torch.sqrt(loss).item()}, commit=False)


def valid_loss(loss):
    wandb.log({"valid/loss": torch.sqrt(loss).item()}, commit=False)
    
def log_metric(stage, data_type, metric, value):
    wandb.log({"{}/{}/{}".format(stage, data_type, metric): value}, commit=False)


def _plot(stage, data_type, metric, plot):
    wandb.log({"{}/{}/plot_{}".format(stage, data_type, metric): plot})


def valid_plot(data_type, plot):
    _plot("valid", data_type, REFERENCE_METRIC[data_type], plot)


def test_plot(data_type, plot):
    _plot("test", data_type, REFERENCE_METRIC[data_type], plot)


def final_score(data_type, average_score):
    wandb.log(
        {
            "final/{}/best_{}".format(
                data_type, REFERENCE_METRIC[data_type]
            ): average_score
        }
    )

def epoch(trainer):
    wandb.log({"epoch": trainer.current_epoch})
