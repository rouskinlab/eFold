import torch
from rouskinhf import seq2int
from ..config import UKN

from scipy import stats


def compute_f1(pred, true, threshold=0.5):
    """
    Compute the F1 score of the predictions.

    :param pred: Predicted pairing matrix probability  (L,L)
    :param true: True binary pairing matrix (L,L)
    :return: F1 score for this RNA structure
    """

    pred = (pred > threshold).float()

    sum_pair = torch.sum(pred) + torch.sum(true)

    if sum_pair == 0:
        return 1.0
    else:
        return (2 * torch.sum(pred * true) / sum_pair).item()


def compute_f1_batch(pred, true, threshold=0.5):
    """
    Compute the mean F1 score of the predictions for a batch.

    :param pred: Predicted pairing matrix probability  (L,L)
    :param true: True binary pairing matrix (L,L)
    :return: F1 score for this RNA structure
    """

    scores = []
    for pred, true in zip(pred, true):
        scores.append(
            compute_f1(
                pred=pred,
                true=true,
                threshold=threshold,
            )
        )
    return torch.mean(torch.tensor(scores)).item()


def compute_mFMI(pred, true, threshold=0.5):
    """
    Compute the mFMI score of the predictions.

    :param pred: Predicted pairing matrix probability  (L,L)
    :param true: True binary pairing matrix (L,L)
    :return: mFMI score for this RNA structure
    """

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


def compute_mFMI_batch(pred, true, threshold=0.5):
    """
    Compute the mean mFMI score of the predictions for a batch.

    :param pred: Predicted pairing matrix probability  (L,L)
    :param true: True binary pairing matrix (L,L)
    :return: mFMI score for this RNA structure
    """

    scores = []
    for pred, true in zip(pred, true):
        scores.append(
            compute_mFMI(
                pred=pred,
                true=true,
                threshold=threshold,
            )
        )
    return torch.mean(torch.tensor(scores)).item()


def r2_score(pred, true):
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


def r2_score_batch(pred, true):
    """
    Compute the mean R2 score of the predictions for a batch.

    :param true: True values
    :param pred: Predicted values
    :return: R2 score
    """

    scores = []
    for true, pred in zip(true, pred):
        scores.append(r2_score(pred=pred, true=true))
    return torch.mean(torch.tensor(scores)).item()


def pearson_coefficient(pred, true):
    """
    Compute the Pearson correlation coefficient of the predictions.

    :param true: True values
    :param pred: Predicted values
    :return: pearson coefficient
    """
    mask = true != UKN
    pred = pred[mask]
    true = true[mask]

    res = stats.pearsonr(
        true.detach().cpu().numpy().flatten(), pred.detach().cpu().numpy().flatten()
    )

    return res[0]


def pearson_coefficient_batch(pred, true):
    """
    Compute the mean Pearson correlation coefficient of the predictions for a batch.

    :param true: True values
    :param pred: Predicted values
    :return: pearson coefficient
    """

    scores = []
    for true, pred in zip(true, pred):
        scores.append(pearson_coefficient(pred=pred, true=true))
    return torch.mean(torch.tensor(scores)).item()


def mae_score(pred, true):
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


def mae_score_batch(pred, true):
    """
    Compute the mean Mean Average Error of the predictions for a batch.

    :param true: True values
    :param pred: Predicted values
    :return: MAE score
    """

    scores = []
    for true, pred in zip(true, pred):
        scores.append(mae_score(pred=pred, true=true))
    return torch.mean(torch.tensor(scores)).item()
