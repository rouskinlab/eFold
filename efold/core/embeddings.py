import torch
from torch import nn

from efold.constants import config

NUM_BASES = len(set(config.tokens.seq2int.values()))


def sequence_to_int(sequence: str) -> torch.Tensor:
    return torch.tensor([config.tokens.seq2int[s] for s in sequence], dtype=torch.int64)


def int_to_sequence(sequence: torch.Tensor) -> str:
    return "".join([config.tokens.int2seq[i.item()] for i in sequence])


def sequence_to_one_hot(sequence_batch: torch.Tensor) -> torch.Tensor:
    return nn.functional.one_hot(sequence_batch, NUM_BASES).type(torch.float32)


def base_pairs_to_pairing_matrix(
    base_pairs: torch.Tensor,
    sequence_length: int,
    padding: int,
    pad_value: float = config.pytorch.unknown_value,
) -> torch.Tensor:
    pairing_matrix = torch.ones((padding, padding)) * pad_value
    if base_pairs is None:
        return pairing_matrix
    pairing_matrix[:sequence_length, :sequence_length] = 0.0
    if len(base_pairs) > 0 and base_pairs.shape[1] == 2 and base_pairs.shape[0] > 0:
        base_pairs = base_pairs.type(torch.long)
        pairing_matrix[base_pairs[:, 0], base_pairs[:, 1]] = 1.0
        pairing_matrix[base_pairs[:, 1], base_pairs[:, 0]] = 1.0
    return pairing_matrix
