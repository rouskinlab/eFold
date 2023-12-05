import torch
from ..config import device, UKN
import torch.nn.functional as F
from .util import _pad


class DataType:
    attributes = ["true", "pred", "error", "index"]

    def __init__(self, true, error=None, index=None, pred=None):
        self.true = true
        self.error = error
        self.index = index
        self.pred = pred

    def to(self, device):
        for attr in DataType.attributes:
            if hasattr(getattr(self, attr), "to"):
                setattr(self, attr, getattr(self, attr).to(device))
        return self


class DataTypeBatch(DataType):
    def __init__(self, true, pred, index, error=None):
        super().__init__(true, pred=pred, index=index, error=error)


class DataTypeDataset(DataType):
    def __init__(self, true, index, max_index, error=None):
        self.max_index = max_index
        super().__init__(true=true, error=error, index=index)

    def __len__(self):
        return self.max_index

    def __add__(self, other):
        if self.name != other.name:
            raise ValueError("Cannot add datasets of different types")

        L = max(self.true.shape[1], other.true.shape[1])

        return data_type_factory["dataset"][self.name](
            true=torch.cat(
                [_pad(self.true, L, self.name), _pad(other.true, L, self.name)], dim=0
            ),
            error=torch.cat(
                [_pad(self.error, L, self.name), _pad(other.error, L, self.name)], dim=0
            ),
            index=torch.cat([self.index, other.index + self.max_index]),
            max_index=self.max_index + other.max_index,
        )

    @classmethod
    def from_data_json(cls, data_json: dict, L: int, refs: list):
        data_type = cls.name
        index, true, error = [], [], []
        for idx, ref in enumerate(refs):
            values = data_json[ref]
            if data_type in values:
                index.append(idx)
                if data_type != "structure":
                    true.append(_pad(values[data_type], L, data_type))
                    error.append(
                        _pad(values["error_{}".format(data_type)], L, data_type)
                    )
                else:
                    true.append(values[data_type])

        if len(index) == 0:
            return None

        def post_process(x, dtype, dim=1):
            if cls.name == "structure" and dim == 2:
                return x
            if len(x) == 0:
                return None
            if dim == 1:
                return torch.tensor(x, dtype=dtype).to(device)
            if dim == 2:
                return torch.stack(x).to(dtype=dtype)
            raise ValueError(f"dim must be 1 or 2, got {dim}")

        return cls(
            **{
                "true": post_process(true, torch.float32, dim=2),
                "error": post_process(error, torch.float32, dim=2),
                "index": post_process(index, torch.int32, dim=1),
                "max_index": len(refs),
            }
        )


class DMSBatch(DataTypeBatch):
    name = "dms"

    def __init__(self, true, pred=None, error=None, index=None):
        super().__init__(true=true, pred=pred, index=index, error=error)


class SHAPEBatch(DataTypeBatch):
    name = "shape"

    def __init__(self, true, pred=None, error=None, index=None):
        super().__init__(true=true, pred=pred, index=index, error=error)


class StructureBatch(DataTypeBatch):
    name = "structure"

    def __init__(self, true, pred=None, index=None, error=None):
        super().__init__(true=true, pred=pred, index=index)


class DMSDataset(DataTypeDataset):
    name = "dms"

    def __init__(self, true, max_index, error=None, index=None):
        super().__init__(true=true, error=error, index=index, max_index=max_index)

    def __getitem__(self, idx):
        if idx not in self.index:
            return None
        local_idx = torch.where(self.index == idx)[0].item()
        return {
            "true": self.true[local_idx],
            "error": self.error[local_idx],
        }


class SHAPEDataset(DataTypeDataset):
    name = "shape"

    def __init__(self, true, max_index, error=None, index=None):
        super().__init__(true=true, error=error, index=index, max_index=max_index)

    def __getitem__(self, idx):
        if idx not in self.index:
            return None
        local_idx = torch.where(self.index == idx)[0].item()
        return {
            "true": self.true[local_idx],
            "error": self.error[local_idx],
        }


class StructureDataset(DataTypeDataset):
    name = "structure"

    def __init__(self, true, max_index, error=None, index=None):
        super().__init__(true=true, index=index, max_index=max_index)

    def __getitem__(self, idx):
        if idx not in self.index:
            return None
        local_idx = torch.where(self.index == idx)[0].item()
        return {
            "true": self.true[local_idx],
        }


data_type_factory = {
    "batch": {"dms": DMSBatch, "shape": SHAPEBatch, "structure": StructureBatch},
    "dataset": {
        "dms": DMSDataset,
        "shape": SHAPEDataset,
        "structure": StructureDataset,
    },
}
