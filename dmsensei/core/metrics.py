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
        / torch.sum((y_true - torch.mean(y_true)) ** 2)
    ).item()


def mae_score(y_true, y_pred):
    """
    Compute the Mean Average Error of the predictions.

    :param y_true: True values
    :param y_pred: Predicted values
    :return: MAE score
    """

    mask = y_true != UKN
    y_pred = y_pred[mask]
    y_true = y_true[mask]

    return torch.mean(torch.abs(y_true - y_pred)).item()


# def mae_score_ACGU(sequence, y_true, y_pred):
#     """
#     Compute the Mean Average Error of the predictions. Weighted by the number of A, C, G and U bases.

#     :param sequence: RNA sequence
#     :param y_true: True values
#     :param y_pred: Predicted values
#     :return: MAE score
#     """

#     MEAN_GU = 0.1
#     nGU = torch.sum(sequence == seq2int["G"]) + torch.sum(sequence == seq2int["U"])
#     nAC = torch.sum(sequence == seq2int["A"]) + torch.sum(sequence == seq2int["C"])
#     mae = mae_score(y_true, y_pred)
    
#     return (nGU * MEAN_GU + nAC * mae) / (nGU + nAC)


def mean_std_dms(y_pred):
    """
    Compute the mean and standard deviation of the predictions.

    :param y_pred: Predicted values
    :return: Mean and standard deviation
    """

    return torch.mean(y_pred), torch.std(y_pred)
