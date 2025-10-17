import efold.core as core

import pandas as pd
import numpy as np
from rouskinhf import get_dataset
import torch

import os

from tqdm import tqdm
from efold import inference

def ListofPairs2pairMatrix(pairs, length):
    matrix = torch.zeros((length, length))

    if len(pairs) == 0: return matrix
    matrix[pairs[:,0], pairs[:,1]] = 1
    matrix[pairs[:,1], pairs[:,0]] = 1

    return matrix.int()


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
        return [1.0, 1.0, 1.0]
    else:
        return [
                (TP / PP).item(),
                (TP / P).item(),
                (2 * TP / sum_pair).item()
                ]


data_path = "/n/data1/hms/microbiology/rouskin/lab/projects/deep_learning/eFold/tests/data"

for test_set in ["PDB", "archiveII", "viral_fragments", "lncRNA_nonFiltered"]:
    data = pd.read_json(os.path.join(data_path, test_set, "data.json")).T
        
    Precisions = []
    Recalls = []
    F1s = []

    for idx, row in tqdm(data.iterrows(), total=len(data)):
        true_structure = torch.tensor(row['structure'])
        sequence = row['sequence']

        prediction = torch.tensor(inference(sequence, fmt='bp')[sequence])-1

        precision, recall, f1 = compute_f1(ListofPairs2pairMatrix(prediction, len(sequence)), 
                                            ListofPairs2pairMatrix(true_structure, len(sequence)))

        Precisions.append(precision)
        Recalls.append(recall)
        F1s.append(f1)


    print(f"{test_set}: precision: {np.nanmean(Precisions):.2f}, recall: {np.nanmean(Recalls):.2f}, f1: {np.nanmean(F1s):.2f}")