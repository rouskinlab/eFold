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


def plot_dms(true_dms, pred_dms, r2=None, layout="bar", interval=100):
    fig, ax = plt.subplots()

    # Base position with no coverage or G/U base are removed
    true_dms[true_dms == UKN] = 0
    pred_dms[true_dms == UKN] = 0

    if layout == "bar":
        # Create and convert plot
        ax.bar(
            np.arange(len(true_dms)), true_dms, color="b", alpha=0.5, label="True DMS"
        )
        ax.bar(
            np.arange(len(pred_dms)),
            pred_dms,
            color="r",
            alpha=0.5,
            label="Predicted DMS",
        )
        ax.set_xlabel("Position")
        ax.set_ylabel("DMS reactivity")

    if layout == "scatter":
        for i in range(int(np.ceil(len(true_dms) / interval))):
            segment_true_dms = true_dms[i * interval : (i + 1) * interval]
            segment_pred_dms = pred_dms[i * interval : (i + 1) * interval]
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


def plot_dms_padding(true_dms, pred_dms):
    fig, ax = plt.subplots()

    pred_dms = pred_dms[true_dms == UKN]
    true_dms = true_dms[true_dms == UKN]

    ax.scatter(true_dms, pred_dms)
    ax.set_xlabel("True DMS")
    ax.set_ylabel("Predicted DMS")

    img = wandb.Image(fig)
    fig = plt.close(fig)

    return img


def plot_structure(true_struct, pred_struct):
    fig, ax = plt.subplots()
    im = ax.imshow(
        pred_struct, cmap="YlOrRd", interpolation="none", alpha=0.8, vmin=0, vmax=1
    )

    x, y = np.where(true_struct == 1)

    plt.scatter(x, y, marker="+", color="black", s=2)

    ax.set_title("Predicted probability (color) and true structure (black)")
    plt.colorbar(im)
    fig.tight_layout()
    # plt.show()
    img = wandb.Image(fig)
    fig = plt.close(fig)

    return img
