import torch
from ..config import device, UKN, DTYPE_PER_DATA_TYPE
import torch.nn.functional as F
from .util import _pad


class DataType:
    attributes = ["true", "pred", "error"]

    def __init__(self, true: list, error: list = None, pred: list = None):
        self.true = true
        self.error = error
        self.pred = pred

    def to(self, device):
        for attr in DataType.attributes:
            if hasattr(getattr(self, attr), "to"):
                setattr(self, attr, getattr(self, attr).to(device))
        return self
    
    def __del__(self):
        del self.true
        del self.error
        del self.pred


class DataTypeBatch(DataType):
    def __init__(self, true, pred, error=None):
        super().__init__(true, pred=pred, error=error)


class DataTypeDataset(DataType):
    def __init__(self, true, error=None):
        super().__init__(true=true, error=error)

    def __len__(self):
        return len(self.true)

    def __getitem__(self, idx):
        out = {
            "true": self.true[idx],
        }
        if self.name != "structure":
            out["error"] = self.error[idx] if self.error is not None else None
        return out

    def __add__(self, other):
        if self.name != other.name:
            raise ValueError(
                f"Cannot concatenate {self.name} and {other.name} datasets."
            )

        if other is None:
            return self

        return data_type_factory["dataset"][self.name](
            true=self.true + other.true,
            error=self.error + other.error if self.name != "structure" else None,
        )

    def __radd__(self, other):
        if other is None:
            return self
        return other + self

    def __delitem__(self, idx):
        del self.true[idx]
        if self.error is not None:
            del self.error[idx]
        if self.pred is not None:
            del self.pred[idx]
            
    def sort(self, idx_sorted):
        self.true = [self.true[i] for i in idx_sorted]
        if self.error is not None:
            self.error = [self.error[i] for i in idx_sorted]
        if self.pred is not None:
            self.pred = [self.pred[i] for i in idx_sorted]

    @classmethod
    def from_data_json(cls, data_json: dict, L: int, refs: list):
        data_type = cls.name
        true, error = [], []
        for ref in refs:
            values = data_json[ref]
            if data_type in values:
                true.append(
                    torch.tensor(
                        values[data_type], dtype=DTYPE_PER_DATA_TYPE[data_type]
                    )
                )
                if data_type != "structure":
                    if "error_{}".format(data_type) in values:
                        error.append(
                            torch.tensor(
                                values["error_{}".format(data_type)],
                                dtype=DTYPE_PER_DATA_TYPE[data_type],
                            )
                        )
                    else:
                        error.append(None)
            else:
                true.append(None)
                if data_type != "structure":
                    error.append(None)

        if len(error):
            assert len(error) == len(true), "error and true must have the same length"

        return cls(
            **{
                "true": true,
                "error": error,
            }
        )


class DMSBatch(DataTypeBatch):
    name = "dms"

    def __init__(self, true, pred=None, error=None):
        super().__init__(true=true, pred=pred, error=error)


class SHAPEBatch(DataTypeBatch):
    name = "shape"

    def __init__(self, true, pred=None, error=None):
        super().__init__(true=true, pred=pred, error=error)


class StructureBatch(DataTypeBatch):
    name = "structure"

    def __init__(self, true, pred=None, error=None):
        super().__init__(true=true, pred=pred)


class DMSDataset(DataTypeDataset):
    name = "dms"

    def __init__(self, true, error=None):
        super().__init__(true=true, error=error)


class SHAPEDataset(DataTypeDataset):
    name = "shape"

    def __init__(self, true, error=None):
        super().__init__(true=true, error=error)


class StructureDataset(DataTypeDataset):
    name = "structure"

    def __init__(self, true, error=None):
        super().__init__(true=true)


data_type_factory = {
    "batch": {"dms": DMSBatch, "shape": SHAPEBatch, "structure": StructureBatch},
    "dataset": {
        "dms": DMSDataset,
        "shape": SHAPEDataset,
        "structure": StructureDataset,
    },
}
