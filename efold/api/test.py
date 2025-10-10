from efold import inference
import torch
import json
import pandas as pd

# User defined
MODEL_NAME = "BASELINE"

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


def compute_f1(pred_matrix, target_matrix, threshold=0.5):
    """
    Compute the F1 score of the predictions.

    :param pred_matrix: Predicted pairing matrix probability  (L,L)
    :param target_matrix: True binary pairing matrix (L,L)
    :return: precision, recall F1 score for this RNA structure
    """

    pred_matrix = (pred_matrix > threshold).float()


    TP = torch.sum(pred_matrix*target_matrix)
    PP = torch.sum(pred_matrix)
    P = torch.sum(target_matrix)
    sum_pair = PP + P

    if sum_pair == 0:
        return (1.0, 1.0, 1.0)
    else:
        return (
                (TP / PP).item(),
                (TP / P).item(),
                (2 * TP / sum_pair).item()
                )

# Datasets to evaluate (dataset_name, absolute_path)
DATASETS = [
    ("archiveII_blast", "/n/data1/hms/microbiology/rouskin/lab/projects/deep_learning/eFold/tests/data/archiveII_blast/data.json"),
    ("lncRNA_nonFiltered", "/n/data1/hms/microbiology/rouskin/lab/projects/deep_learning/eFold/tests/data/lncRNA_nonFiltered/data.json"),
    ("PDB", "/n/data1/hms/microbiology/rouskin/lab/projects/deep_learning/eFold/tests/data/PDB/data.json"),
    ("viral_fragments", "/n/data1/hms/microbiology/rouskin/lab/projects/deep_learning/eFold/tests/data/viral_fragments/data.json"),
]

# Run batched inference per dataset and aggregate results
rows = []
for dataset_name, data_path in DATASETS:
    with open(data_path) as f:
        dataset = json.load(f)

    ids = list(dataset.keys())
    sequences = [dataset[k]["sequence"] for k in ids]
    predictions = inference(sequences, fmt='basepair')

    for k, seq in zip(ids, sequences):
        L = len(seq)
        pred_bp = predictions.get(seq, [])
        pred_bp = torch.tensor(pred_bp, dtype=torch.long) - 1 if pred_bp else torch.empty((0, 2), dtype=torch.long)
        true_bp = torch.tensor(dataset[k]["structure"], dtype=torch.long)

        pred_mat = basepair_to_pairing_matrix(pred_bp, L) if pred_bp.numel() else torch.zeros((L, L), dtype=torch.int)
        true_mat = basepair_to_pairing_matrix(true_bp, L) if true_bp.numel() else torch.zeros((L, L), dtype=torch.int)

        precision, recall, f1_score = compute_f1(pred_mat.float(), true_mat)
        rows.append({
            "dataset": dataset_name,
            "id": k,
            "sequence": seq,
            "precision": precision,
            "recall": recall,
            "f1": f1_score,
        })

df = pd.DataFrame(rows)
output_csv = f"/n/data1/hms/microbiology/rouskin/lab/projects/deep_learning/eFold/tests/eval_f1_{MODEL_NAME}.csv"
df.to_csv(output_csv, index=False)
print(f"Saved results to {output_csv}")