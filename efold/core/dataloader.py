from torch.utils.data import DataLoader as _DataLoader
import torch
from .batch import Batch

class DataLoader(_DataLoader):
    def __init__(self, *args, **kwargs):
        if "to_device" in kwargs:
            self.to_device = kwargs.pop("to_device")
        super().__init__(*args, **kwargs)

    def transfer_batch_to_device(self, batch: Batch, device: torch.device, dataloader_idx: int) -> Batch:
        if self.to_device:
            return batch.to(device)
        return batch
