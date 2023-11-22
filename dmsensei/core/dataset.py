from rouskinhf import import_dataset
from torch.utils.data import Dataset as TorchDataset
from typing import List
from .batch import Batch
from .listofdatapoints import ListOfDatapoints


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
        data = import_dataset(name, force_download=force_download)
        self.data_type = data_type + ["sequence"] if "sequence" not in data_type else data_type
        self.list_of_datapoints = ListOfDatapoints.from_rouskinhf(
            data, data_type, name=name, tqdm=tqdm
        )

    def __len__(self) -> int:
        return len(self.list_of_datapoints)

    def __getitem__(self, index) -> tuple:
        return self.list_of_datapoints[index]

    def collate_fn(self, batch_data):
        batch = Batch.from_list_of_datapoints(batch_data, self.data_type)
        return batch
