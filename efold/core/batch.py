import torch
from torch import tensor
import torch.nn.functional as F
from .embeddings import base_pairs_to_pairing_matrix, sequence_to_int
from ..config import device, POSSIBLE_METRICS, UKN
from typing import Dict
from .datatype import data_type_factory
from .util import split_data_type
from torch import cuda, backends


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
        use_error,
        batch_size,
        data_types,
        dt_count,
        dms=None,
        shape=None,
        structure=None,
        device = 'cpu'
    ):
        self.reference = reference
        self.sequence = sequence
        self.length = length
        self.use_error = use_error
        self.dms = dms
        self.shape = shape
        self.structure = structure
        self.L = L
        self.batch_size = batch_size
        self.data_types = data_types
        self.dt_count = dt_count
        self.device = device

    @classmethod
    def from_dataset_items(
        cls,
        batch_data: list,
        data_type: str,
        use_error: bool,
        structure_padding_value: float = UKN,
    ):
        reference = [dp["reference"] for dp in batch_data]
        length = [dp["length"] for dp in batch_data]
        L = max(length)

        # move the conversion to the dataset
        sequence = torch.stack(
            [_pad(sequence_to_int(dp["sequence"]), L, "sequence") for dp in batch_data]
        )
        batch_size = len(reference)

        data = {}
        dt_count = {
            dt: len(
                [
                    1
                    for dp in batch_data
                    if dt in dp and dp[dt] is not None and dp[dt]["true"] is not None
                ]
            )
            for dt in data_type
        }
        for dt in data_type:
            if dt == "structure":
                data[dt] = data_type_factory["batch"][dt](
                    true=torch.stack(
                        [
                            base_pairs_to_pairing_matrix(
                                dp["structure"]["true"],
                                l,
                                padding=L,
                                pad_value=structure_padding_value,
                            )
                            for (dp, l) in zip(batch_data, length)
                        ]
                    ),
                    error=None,
                    pred=None,
                )
            else:
                true, error = [], []
                for dp in batch_data:
                    true.append(_pad(dp[dt]["true"], L, dt, accept_none=True))
                true = torch.stack(true)

                # use error if there's a single non-None error and if the true signal is not None
                if use_error and len(
                    [1 for dp in batch_data if dp[dt]["error"] is not None]
                ):
                    for dp in batch_data:
                        error.append(_pad(dp[dt]["error"], L, dt, accept_none=True))
                    error = torch.stack(error)
                else:
                    error = [None] * batch_size

                data[dt] = data_type_factory["batch"][dt](true=true, error=error)

        return cls(
            reference=reference,
            sequence=sequence,
            length=length,
            use_error=use_error,
            L=L,
            batch_size=batch_size,
            data_types=data_type,
            dt_count=dt_count,
            **data,
        )

    def get(self, data_type, index=None, to_numpy=False):
        if data_type in ["reference", "sequence", "length"]:
            out = getattr(self, data_type)
            data_part = None
        else:
            data_part, data_type = split_data_type(data_type)

            # could be in the dataset but wasn't requested in the dm init
            if data_type not in self.data_types:
                return None

            # no data for this data_type
            if getattr(self, data_type) is None:
                return None

            # get data from the objects
            out = getattr(getattr(self, data_type), data_part)

        if index is not None:
            out = out[index]
            if hasattr(out, "__len__"):
                l = self.get("length")[index]
                if data_type == "structure":
                    out = out[:l, :l]
                else:
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
        return (
            self.get("pred_{}".format(data_type), to_numpy=to_numpy),
            self.get("true_{}".format(data_type), to_numpy=to_numpy),
        )

    def count(self, data_type):
        if data_type in ["reference", "sequence", "length"]:
            return self.batch_size
        if not data_type in self.dt_count or getattr(self, data_type) is None:
            return 0
        return self.dt_count[data_type]

    def contains(self, data_type):
        if data_type in ["reference", "sequence", "length"]:
            return True
        data_part, data_type = split_data_type(data_type)
        if not self.count(data_type):
            return False
        if (
            not hasattr(
                getattr(self, data_type), data_part
            )  # that's more of a sanity check
            or getattr(getattr(self, data_type), data_part) is None
        ):
            return False
        return True

    def __len__(self):
        return self.count("sequence")


    #     return out

    def __del__(self):
        del self.dms
        del self.shape
        del self.structure
        del self.reference
        del self.sequence
        del self.length
        del self.L
        del self.batch_size
        del self.data_types
        del self.dt_count
        del self
    
    @property
    def device(self):
        return self._device

    @device.getter
    def device(self):
        return self._device

    @device.setter
    def device(self, device):
        # assert device exists
        if device == 'mps' and not backends.mps.is_available():
            raise ValueError("MPS is not available on this device.")
        if device == 'cuda' and not cuda.is_available():
            raise ValueError("CUDA is not available on this device.")
        for attr in ['dms', 'shape', 'structure', 'sequence']:
            if getattr(self, attr) is not None:
                setattr(self, attr, getattr(self, attr).to(device))
        self._device = device

    def to(self, device):
        self.device = device
        return self
