from torch import nn
import torch
from ..config import DEFAULT_FORMAT
from rouskinhf.util import seq2int, dot2int, int2seq

NUM_BASES = len(set(seq2int.values()))


def base_pairs_to_int_dot_bracket(base_pairs, sequence_length, dtype=torch.int64):
    dot_bracket = ["."] * sequence_length
    for i, j in base_pairs:
        dot_bracket[i] = "("
        dot_bracket[j] = ")"
    return torch.tensor([dot2int[s] for s in dot_bracket], dtype=dtype)


def sequence_to_int(sequence: str):
    return torch.tensor([seq2int[s] for s in sequence], dtype=torch.int64)


# from dmsensei.core.embeddings import int_to_sequence
def int_to_sequence(sequence: torch.tensor):
    return "".join([int2seq[i.item()] for i in sequence])


def sequence_to_one_hot(sequence_batch: torch.tensor):
    """Converts a sequence to a one-hot encoding"""
    return nn.functional.one_hot(sequence_batch, NUM_BASES).type(DEFAULT_FORMAT)


def int_dot_bracket_to_one_hot(int_dot_bracket: torch.tensor):
    """Converts a sequence to a one-hot encoding"""
    return nn.functional.one_hot(int_dot_bracket, len(dot2int))


def base_pairs_to_pairing_matrix(base_pairs, sequence_length):
    # TODO #6 #Vectorize this function
    pairing_matrix = torch.zeros((sequence_length, sequence_length))
    for i, j in base_pairs:
        pairing_matrix[i.item(), j.item()] = 1
        pairing_matrix[j.item(), i.item()] = 1
    return pairing_matrix


def pairing_matrix_to_base_pairs(pairing_matrix):
    base_pairs = []
    for i in range(pairing_matrix.shape[0]):
        for j in range(pairing_matrix.shape[1]):
            if pairing_matrix[i, j] == 1:
                base_pairs.append([i, j])
                pairing_matrix[j, i] = 0
    return base_pairs
