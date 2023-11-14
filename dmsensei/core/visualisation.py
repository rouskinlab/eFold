from matplotlib import pyplot as plt
import numpy as np
import wandb
from .metrics import r2_score
from ..config import UKN

matplotlib_colors = [
    "red",
    "blue",
    "green",
    "yellow",
    "purple",
    "cyan",
    "magenta",
    "orange",
    "brown",
    "pink",
    "teal",
    "lavender",
    "olive",
    "maroon",
    "navy",
]


def plot_r2_distribution(r2_scores):
    fig, ax = plt.subplots()
    ax.hist(r2_scores, bins=20)
    ax.set_xlabel("R2 score")
    ax.set_ylabel("Count")
    ax.set_title("R2 score distribution")
    img = wandb.Image(fig)
    fig = plt.close(fig)

    return img


def plot_dms(pred, true, r2=None, layout="bar", interval=100, **kwargs):
    fig, ax = plt.subplots()

    # Base position with no coverage or G/U base are removed
    mask = true != UKN
    pred = pred[mask]
    true = true[mask]

    if layout == "bar":
        # Create and convert plot
        ax.bar(np.arange(len(true)), true, color="b", alpha=0.5, label="True DMS")
        ax.bar(
            np.arange(len(pred)),
            pred,
            color="r",
            alpha=0.5,
            label="Predicted DMS",
        )
        ax.set_xlabel("Position")
        ax.set_ylabel("DMS reactivity")

    if layout == "scatter":
        for i in range(int(np.ceil(len(true) / interval))):
            segment_true_dms = true[i * interval : (i + 1) * interval]
            segment_pred_dms = pred[i * interval : (i + 1) * interval]
            ax.scatter(
                segment_true_dms,
                segment_pred_dms,
                color=matplotlib_colors[i],
                alpha=0.5,
                label="{} - {}".format(i * interval, (i + 1) * interval),
            )
        ax.set_xlabel("True DMS")
        ax.set_ylabel("Predicted DMS")

    ax.legend()
    if r2 is not None:
        ax.set_title("R2 score: {:.3f}".format(r2))

    img = wandb.Image(fig)
    fig = plt.close(fig)

    return img


def plot_dms_padding(pred, true, **kwargs):
    fig, ax = plt.subplots()

    pred = pred[true == UKN]
    true = true[true == UKN]

    ax.scatter(true, pred)
    ax.set_xlabel("True DMS")
    ax.set_ylabel("Predicted DMS")

    img = wandb.Image(fig)
    fig = plt.close(fig)

    return img


def plot_structure(pred, true, **kwargs):
    fig, ax = plt.subplots()
    im = ax.imshow(pred, cmap="YlOrRd", interpolation="none", alpha=0.8, vmin=0, vmax=1)

    x, y = np.where(true == 1)

    plt.scatter(x, y, marker="+", color="black", s=2)

    ax.set_title("Predicted probability (color) and true structure (black)")
    plt.colorbar(im)
    fig.tight_layout()
    # plt.show()
    img = wandb.Image(fig)
    fig = plt.close(fig)

    return img


plot_factory = {"dms": plot_dms, "shape": plot_dms, "structure": plot_structure}  # TODO
