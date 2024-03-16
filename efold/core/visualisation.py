from matplotlib import pyplot as plt
import numpy as np
import wandb
from .metrics import r2_score, mae_score, pearson_coefficient
from ..config import UKN
from rouskinhf import int2seq

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


def plot_signal(
    pred,
    true,
    data_type,
    sequence,
    length,
    reference=None,
    layout="scatter",
    interval=100,
    **kwargs,
):
    fig, ax = plt.subplots()

    # Compute metrics while you still have tensors
    r2 = r2_score(pred=pred, true=true)
    r = pearson_coefficient(pred=pred, true=true)
    mae = mae_score(pred=pred, true=true)

    # Base position with no coverage or G/U base are removed
    def chop_array(x):
        return x[:length].squeeze()

    def known_bases_to_list(x, mask):
        return x[mask].cpu().numpy()

    pred, true, sequence = chop_array(pred), chop_array(true), chop_array(sequence)
    mask = true != UKN
    true, pred, sequence = (
        known_bases_to_list(true, mask),
        known_bases_to_list(pred, mask),
        known_bases_to_list(sequence, mask),
    )

    if layout == "bar":
        # Create and convert plot
        ax.bar(
            np.arange(len(true)),
            true,
            color="b",
            alpha=0.5,
            label="True {}".format(data_type),
        )
        ax.bar(
            np.arange(len(pred)),
            pred,
            color="r",
            alpha=0.5,
            label="Predicted {}".format(data_type),
        )
        ax.set_xlabel("Position")
        ax.set_ylabel("{} reactivity".format(data_type))

    if layout == "scatter":
        for i in set(sequence):
            color = {
                "A": "red",
                "C": "blue",
                "G": "green",
                "U": "yellow",
                "T": "yellow",
            }[int2seq[i]]
            segment_true = true[sequence == i]
            segment_pred = pred[sequence == i]
            ax.scatter(
                segment_true,
                segment_pred,
                color=color,
                alpha=0.5,
                label=int2seq[i],
            )
        ax.set_xlabel("True {}".format(data_type))
        ax.set_ylabel("Predicted {}".format(data_type))

        # minimal window is 0, 1
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        ax.set_xlim(min(xmin, 0), max(xmax, 1))
        ax.set_ylim(min(ymin, 0), max(ymax, 1))

    ax.legend()
    title = "R2 = {:.2f}, pearson = {:.2f}, MAE = {:.2f}".format(r2, r, mae)
    if reference is not None:
        title = "{}: ".format(reference) + title
    ax.set_title(title)

    return fig


def plot_structure(pred, true, **kwargs):
    fig, ax = plt.subplots()
    pred, true = pred.cpu().numpy(), true.cpu().numpy()
    im = ax.imshow(pred, cmap="YlOrRd", interpolation="none", alpha=0.8, vmin=0, vmax=1)

    x, y = np.where(true == 1)

    plt.scatter(x, y, marker="+", color="black", s=2)

    ax.set_title("Predicted probability (color) and true structure (black)")
    plt.colorbar(im)
    fig.tight_layout()
    # plt.show()
    return fig


plot_factory = {
    ("dms", "scatter"): lambda *args, **kwargs: plot_signal(
        *args, **kwargs, data_type="DMS"
    ),
    ("shape", "scatter"): lambda *args, **kwargs: plot_signal(
        *args, **kwargs, data_type="SHAPE"
    ),
    ("structure", "heatmap"): plot_structure,
}
