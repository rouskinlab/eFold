from sklearn.metrics import f1_score
import numpy as np
import torch
import matplotlib.pyplot as plt


def f1_between_bp_lists(pred, truth, L):
    """Get two lists of base pairs and return the F1 score between them."""
    pred_mat = bp_to_matrix(pred, L)
    truth_mat = bp_to_matrix(truth, L)
    if (pred_mat == truth_mat).all():
        return 1.
    return f1_score(truth_mat.flatten(), pred_mat.flatten())
    
def bp_to_matrix(bp, L):
    """Convert a list of base pairs to a matrix."""
    mat = np.zeros((L, L))
    if isinstance(bp, float):
        assert np.isnan(bp), "bp is a float but not NaN"
        return mat
    for i, j in bp:
        mat[i, j] = 1
        mat[j, i] = 1
    return mat


seq2int = {"X": 0, "A": 1, "C": 2, "G": 3, "U": 4}  # , 'S': 5, 'E': 6}
def one_hot_encode(seq):
    seq = np.array([seq2int[s] for s in seq])
    one_hot_embed = torch.zeros((5, 4))
    one_hot_embed[1:] = torch.eye(4)

    return one_hot_embed[seq].unsqueeze(0)