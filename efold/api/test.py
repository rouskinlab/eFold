from efold import inference
import torch
import json
import pandas as pd

def basepair_to_pairing_matrix(basepairs, L):
    """
    Convert a list of base pairs to a pairing matrix.

    :param basepairs: List of base pairs [(i, j), (k, l), ...]
    :param L: Length of the sequence
    :return: Pairing matrix of shape (L, L)
    """
    matrix = torch.zeros((L, L), dtype=torch.int)
    matrix[basepairs[:, 0], basepairs[:, 1]] = 1
    matrix[basepairs[:, 1], basepairs[:, 0]] = 1
    return matrix


def f1(pred, true, threshold=0.5):
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

# Load dataset and run batched inference
data_path = "/n/data1/hms/microbiology/rouskin/lab/projects/deep_learning/eFold/data/test_sets/archiveII.json"
with open(data_path) as f:
    dataset = json.load(f)

ids = list(dataset.keys())
sequences = [dataset[k]["sequence"] for k in ids]
predictions = inference(sequences, fmt='basepair')

# Compute F1 for each entry and collect as a DataFrame
rows = []
for k, seq in zip(ids, sequences):
    L = len(seq)
    pred_bp = predictions[seq]
    pred_bp = torch.tensor(pred_bp, dtype=torch.long) - 1 if pred_bp else torch.empty((0, 2), dtype=torch.long)
    true_bp = torch.tensor(dataset[k]["structure"], dtype=torch.long)

    pred_mat = basepair_to_pairing_matrix(pred_bp, L) if pred_bp.numel() else torch.zeros((L, L), dtype=torch.int)
    true_mat = basepair_to_pairing_matrix(true_bp, L) if true_bp.numel() else torch.zeros((L, L), dtype=torch.int)

    rows.append({"id": k, "sequence": seq, "f1": f1(pred_mat.float(), true_mat)})

df = pd.DataFrame(rows)
df.to_csv("/n/data1/hms/microbiology/rouskin/lab/projects/deep_learning/eFold/data/test_sets/archiveII_f1_baseline_avg_2.csv", index=False)
print(df.tail())