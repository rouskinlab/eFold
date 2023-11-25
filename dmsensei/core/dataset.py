from torch.utils.data import Dataset as TorchDataset
from typing import List
from .batch import Batch
from ..huggingface import get_dataset
from ..config import device


class Dataset(TorchDataset):
    def __init__(
        self,
        name: str,
        data_type: List[str],
        force_download: bool,
        tqdm=True,
    ) -> None:
        super().__init__()
        self.name = name
        self.data_type = data_type
        self.list_of_datapoints = get_dataset(
            name=name,
            force_download=force_download,
            tqdm=tqdm,
        )
        for dp in self.list_of_datapoints:
            dp.to(device=device)

    def __len__(self) -> int:
        return len(self.list_of_datapoints)

    def __getitem__(self, index) -> tuple:
        return self.list_of_datapoints[index]

    def collate_fn(self, batch_data):
        batch = Batch.from_list_of_datapoints(batch_data, self.data_type)
        return batch
