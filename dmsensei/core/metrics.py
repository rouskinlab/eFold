import torch
from ..config import UKN


# make a decorator to run the metrics on the batch
def batch_mean(fn):
    def wrapper(pred, true, batch=False, *args, **kwargs):
        if not batch:
            return fn(pred=pred, true=true, *args, **kwargs)
        scores = []
        for p, t in zip(pred, true):
            scores.append(fn(pred=p.squeeze(), true=t.squeeze(), *args, **kwargs))
        avg = torch.nanmean(torch.tensor(scores))
        if ~torch.isnan(avg):
            return avg.item()
        else:
            return None

    return wrapper


@batch_mean
def f1(pred, true, batch=None, threshold=0.5):
    """
    Compute the F1 score of the predictions.

    :param pred: Predicted pairing matrix probability  (L,L)
    :param true: True binary pairing matrix (L,L)
    :return: F1 score for this RNA structure
    """
    mask = true != UKN
    pred = pred[mask]
    true = true[mask]

    pred = (pred > threshold).float()

    sum_pair = torch.sum(pred) + torch.sum(true)

    if sum_pair == 0:
        return 1.0
    else:
        return (2 * torch.sum(pred * true) / sum_pair).item()


@batch_mean
def mFMI(pred, true, batch=None, threshold=0.5):
    """
    Compute the mFMI score of the predictions.

    :param pred: Predicted pairing matrix probability  (L,L)
    :param true: True binary pairing matrix (L,L)
    :return: mFMI score for this RNA structure
    """

    mask = true != UKN
    pred = pred[mask]
    true = true[mask]

    pred = (pred > threshold).float()

    TP = torch.sum(pred * true)

    prod_true = torch.sum(pred) * torch.sum(true)
    if prod_true > 0:
        FMI = TP / torch.sqrt(prod_true)
    else:
        FMI = 0

    u = (
        torch.sum((~torch.sum(pred, dim=1).bool()) * (~torch.sum(true, dim=1).bool()))
        / pred.shape[-1]
    )

    mFMI = u + (1 - u) * FMI

    return mFMI.item()


@batch_mean
def r2_score(pred, true, batch=None):
    """
    Compute the R2 score of the predictions.

    :param true: True values
    :param pred: Predicted values
    :return: R2 score
    """

    mask = true != UKN
    pred = pred[mask]
    true = true[mask]

    return (
        1 - torch.sum((true - pred) ** 2) / torch.sum((true - torch.mean(true)) ** 2)
    ).item()


@batch_mean
def pearson_coefficient(pred, true, batch=None):
    """
    Compute the Pearson correlation coefficient of the predictions.

    :param true: True values
    :param pred: Predicted values
    :return: pearson coefficient
    """
    mask = true != UKN
    pred = pred[mask]
    true = true[mask]
    return torch.mean(
        (pred - torch.mean(pred))
        * (true - torch.mean(true))
        / (torch.std(pred) * torch.std(true))
    ).item()


@batch_mean
def mae_score(pred, true, batch=None):
    """
    Compute the Mean Average Error of the predictions.

    :param true: True values
    :param pred: Predicted values
    :return: MAE score
    """

    mask = true != UKN
    pred = pred[mask]
    true = true[mask]

    return torch.mean(torch.abs(true - pred)).item()


metric_factory = {
    "f1": f1,
    "mFMI": mFMI,
    "r2": r2_score,
    "pearson": pearson_coefficient,
    "mae": mae_score,
}
