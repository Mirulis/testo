import os
import torch
import numpy as np
import pandas as pd

from typing import Tuple, Union
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class WineDataset(Dataset):
    @classmethod
    def from_csv(
        cls, path: str, name: str, test_size: Union[int, float], device: str = "cpu"
    ) -> Tuple["WineDataset", "WineDataset"]:
        def z_score(arr: np.ndarray) -> torch.Tensor:
            mean = np.mean(arr, axis=0)
            std = np.std(arr, axis=0)
            mid_arr = torch.from_numpy((arr - mean) / std)
            mid_arr_max = torch.max(mid_arr, dim=0).values
            mid_arr_min = torch.min(mid_arr, dim=0).values
            return (mid_arr - mid_arr_min) / (mid_arr_max - mid_arr_min) * 2 - 1

        src = pd.read_csv(os.path.join(path, name), sep=";").values
        data = z_score(src[:, :-1])
        label = torch.zeros(
            (src.shape[0], 6),
            dtype=torch.float32,
        )
        for i, v in enumerate(src[:, -1]):
            label[i, int(v) - 3] = 1
        train_data, test_data, train_label, test_label = map(
            torch.as_tensor, train_test_split(data, label, test_size=test_size)
        )
        return cls(train_data, train_label, device), cls(test_data, test_label, device)

    def __init__(self, data: torch.Tensor, label: torch.Tensor, device: str = "cpu"):
        super().__init__()
        self._data: torch.Tensor = data.float().to(device)
        self._label: torch.Tensor = label.float().to(device)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._data[index], self._label[index]

    def __len__(self):
        return len(self._label)

    def to_dataloader(self, **kwargs) -> DataLoader:
        kwargs.pop("dataset", None)
        return DataLoader(self, **kwargs)
