import torch
import torch.nn.functional as F
from .embeddings import base_pairs_to_pairing_matrix
from ..config import device, POSSIBLE_METRICS, UKN
from .metrics import metric_factory
from typing import Dict
from .datatype import data_type_factory
from .util import split_data_type


def _pad(arr, L, data_type):
    padding_values = {
        "sequence": 0,
        "dms": UKN,
        "shape": UKN,
    }
    if data_type == "structure":
        return base_pairs_to_pairing_matrix(arr, L)
    else:
        return F.pad(arr, (0, L - len(arr)), value=padding_values[data_type])


def get_padded_vector(dp, data_type, data_part, L):
    if getattr(dp, data_type) is None:
        return None
    if getattr(getattr(dp, data_type), data_part) is None:
        return None
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
    def from_dataset_items(cls, datapoints: list, data_type: str):
        reference = [d["reference"] for d in datapoints]
        length = [d["length"] for d in datapoints]
        L = max(length)
        sequence = torch.stack([d["sequence"][:L]
                               for d in datapoints]).to(device)
        batch_size = len(datapoints)

        data = {}
        for dt in data_type:
            true, error, index = [], [], []
            for idx, dp in enumerate(datapoints):
                if dt in dp and dp[dt] is not None:
                    true.append(dp[dt]["true"])
                    if dt != "structure":
                        error.append(dp[dt]["error"])
                    index.append(idx)

            if len(true) == 0:
                continue

            if dt != "structure":
                data[dt] = data_type_factory["batch"][dt](
                    true=torch.stack(true)[:, :L].to(device),
                    pred=None,
                    error=torch.stack(error)[:, :L].to(device),
                    index=torch.tensor(index).to(device),
                )
            else:
                data[dt] = data_type_factory["batch"][dt](
                    true=torch.stack([_pad(t, L, "structure") for t in true]).to(
                        device
                    ),
                    pred=None,
                    index=torch.tensor(index).to(device),
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
            if data_part in ["true", "error"]:  # use the right length
                index = self.get(f"index_{data_type}")[index]
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
                setattr(self, data_type, data_type_factory['batch'][data_type](
                    true=None, pred=pred
                ))

    def get_pairs(self, data_type, to_numpy=False):
        index = self.get_index(data_type)
        return (
            self.get("pred_{}".format(data_type), to_numpy=to_numpy)[index],
            self.get("true_{}".format(data_type), to_numpy=to_numpy),
        )

    def count(self, data_type):
        if getattr(self, data_type) is None:
            return 0
        return len(self.get_index(data_type))

    def get_index(self, data_type):
        return self.get("index_{}".format(data_type))

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
