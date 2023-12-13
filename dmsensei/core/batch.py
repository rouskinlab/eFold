import torch
from torch import tensor
import torch.nn.functional as F
from .embeddings import base_pairs_to_pairing_matrix, sequence_to_int
from ..config import device, POSSIBLE_METRICS, UKN
from .metrics import metric_factory
from typing import Dict
from .datatype import data_type_factory
from .util import split_data_type


def _pad(arr, L, data_type, accept_none=False):
    padding_values = {
        "sequence": 0,
        "dms": UKN,
        "shape": UKN,
    }
    assert (
        data_type in padding_values.keys()
    ), f"Unknown data type {data_type}. If you want to pad a structure, use base_pairs_to_pairing_matrix."
    if accept_none and arr is None:
        return tensor([padding_values[data_type]] * L)
    return F.pad(arr, (0, L - len(arr)), value=padding_values[data_type])


def get_padded_vector(dp, data_type, data_part, L):
    if getattr(dp, data_type) is None:
        return tensor([UKN] * L)
    if getattr(getattr(dp, data_type), data_part) is None:
        return tensor([UKN] * L)
    return _pad(getattr(getattr(dp, data_type), data_part), L, data_type)


class Batch:
    def __init__(
        self,
        reference,
        sequence,
        length,
        L,
        batch_size,
        data_types,
        dms=None,
        shape=None,
        structure=None,
    ):
        self.reference = reference
        self.sequence = sequence
        self.length = length
        self.dms = dms
        self.shape = shape
        self.structure = structure
        self.L = L
        self.batch_size = batch_size
        self.data_types = data_types

    @classmethod
    def from_dataset_items(cls, batch_data: list, data_type: str):
        reference = [dp["reference"] for dp in batch_data]
        length = [len(dp["sequence"]) for dp in batch_data]
        L = max(length)

        # move the conversion to the dataset
        sequence = torch.stack(
            [_pad(sequence_to_int(dp["sequence"]), L, "sequence") for dp in batch_data]
        ).to(device)
        batch_size = len(reference)

        data = {}
        for dt in data_type:
            if dt == "structure":
                data["structure"] = data_type_factory["batch"][dt](
                    true=torch.stack(
                        [
                            base_pairs_to_pairing_matrix(
                                dp["structure"]["true"], l, padding=L
                            )
                            for (dp, l) in zip(batch_data, length)
                        ]
                    ).to(device),
                    error=None,
                    pred=None,
                )
            else:
                true, error = [], []
                for dp in batch_data:
                    true.append(_pad(dp[dt]["true"], L, dt, accept_none=True))
                    if hasattr(dp[dt], "error"):
                        error.append(_pad(dp[dt]["error"], L, dt, accept_none=True))
                true = torch.stack(true).to(device)
                error = torch.stack(error).to(device) if len(error) else None
                data[dt] = data_type_factory["batch"][dt](true=true, error=error).to(
                    device
                )

        return cls(
            reference=reference,
            sequence=sequence,
            length=length,
            L=L,
            batch_size=batch_size,
            data_types=data_type,
            **data,
        )

    def get(self, data_type, index=None, to_numpy=False):
        if data_type in ["reference", "sequence", "length"]:
            out = getattr(self, data_type)
            data_part = None
        else:
            data_part, data_type = split_data_type(data_type)
            if not self.contains(data_type):
                raise ValueError(f"Batch does not contain {data_type}")
            out = getattr(getattr(self, data_type), data_part)

        if index is not None:
            out = out[index]
            if hasattr(out, "__len__"):
                out = out[: self.get("length")[index]]

        if to_numpy:
            if hasattr(out, "cpu"):
                out = out.squeeze().cpu().numpy()
        return out

    def integrate_prediction(self, prediction):
        for data_type, pred in prediction.items():
            if getattr(self, data_type) is not None:
                getattr(self, data_type).pred = pred
            else:
                setattr(
                    self,
                    data_type,
                    data_type_factory["batch"][data_type](true=None, pred=pred),
                )

    def get_pairs(self, data_type, to_numpy=False):
        idx = torch.tensor(
            [
                idx
                for idx, arr in enumerate(getattr(self, data_type).true)
                if arr is not None
            ]
        )
        return (
            self.get("pred_{}".format(data_type), to_numpy=to_numpy)[idx],
            self.get("true_{}".format(data_type), to_numpy=to_numpy)[idx],
        )

    def count(self, data_type):
        if not self.contains(data_type):
            return 0
        return len(
            [
                arr
                for arr in self.get(data_type=data_type)
                if arr is not None and len(torch.unique(arr)) > 1
            ]
        )  # TODO: remove the unique check since you could have a structure with only 0s for example

    def contains(self, data_type):
        if data_type in ["reference", "sequence", "length"]:
            return True
        data_part, data_type = split_data_type(data_type)
        if (
            not hasattr(self, data_type)
            or getattr(self, data_type) is None
            or data_type not in self.data_types
        ):
            return False
        if (
            not hasattr(getattr(self, data_type), data_part)
            or getattr(getattr(self, data_type), data_part) is None
        ):
            return False
        return True

    def __len__(self):
        return self.count("sequence")

    def compute_metrics(self) -> Dict[str, Dict[str, float]]:
        out = {}
        for data_type in self.data_types:
            if (
                not self.count(data_type)
                or data_type == "sequence"
                or not self.contains(f"pred_{data_type}")
            ):
                continue
            out[data_type] = {}
            pred, true = self.get_pairs(data_type)
            for metric in POSSIBLE_METRICS[data_type]:
                out[data_type][metric] = metric_factory[metric](
                    pred=pred, true=true, batch=self.count(data_type) > 1
                )
        return out
