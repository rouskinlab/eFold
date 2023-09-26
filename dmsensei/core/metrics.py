import torch
from rouskinhf import seq2int
from ..config import UKN


def compute_f1(pred_matrix, target_matrix, threshold=0.5):
    """
    Compute the F1 score of the predictions.

    :param pred_matrix: Predicted pairing matrix probability  (L,L)
    :param target_matrix: True binary pairing matrix (L,L)
    :return: F1 score for this RNA structure
    """

    pred_matrix = (pred_matrix > threshold).float()

    sum_pair = torch.sum(pred_matrix) + torch.sum(target_matrix)

    if sum_pair == 0:
        return 1.0
    else:
        return (2 * torch.sum(pred_matrix * target_matrix) / sum_pair).item()


def compute_mFMI(pred_matrix, target_matrix, threshold=0.5):
    """
    Compute the mFMI score of the predictions.

    :param pred_matrix: Predicted pairing matrix probability  (L,L)
    :param target_matrix: True binary pairing matrix (L,L)
    :return: mFMI score for this RNA structure
    """

    pred_matrix = (pred_matrix > threshold).float()

    TP = torch.sum(pred_matrix * target_matrix)

    prod_true = torch.sum(pred_matrix) * torch.sum(target_matrix)
    if prod_true > 0:
        FMI = TP / torch.sqrt(prod_true)
    else:
        FMI = 0

    u = (
        torch.sum(
            (~torch.sum(pred_matrix, dim=1).bool())
            * (~torch.sum(target_matrix, dim=1).bool())
        )
        / pred_matrix.shape[-1]
    )

    mFMI = u + (1 - u) * FMI

    return mFMI.item()


def r2_score(y_true, y_pred):
    """
    Compute the R2 score of the predictions.

    :param y_true: True values
    :param y_pred: Predicted values
    :return: R2 score
    """

    mask = y_true != UKN
    y_pred = y_pred[mask]
    y_true = y_true[mask]

    return (
        1
        - torch.sum((y_true - y_pred) ** 2)
        / torch.sum((y_true - torch.mean(y_true)) ** 2).item()
    )
