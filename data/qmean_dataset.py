import pandas as pd
import pyarrow
import pyarrow.dataset as ds
import torch
from torch.utils.data import (Dataset)


class QmeanDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, qmin: float, qmax: float):
        self.table = ds.dataset(pyarrow.Table.from_pandas(dataframe))


        self.qmin = qmin
        self.qmax = qmax

        self.targets = self.table.column("avg_local_score").to_pylist()

        self.sequences = self.table.column("sequence").to_pylist()

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        target = self.targets[idx]
        target = 2 * (target - self.qmin) / (self.qmax - self.qmin) - 1.0
        return target, torch.tensor(target, dtype=torch.float32)
