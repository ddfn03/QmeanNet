import pyarrow.parquet as pq
import os
import torch
from logging import getLogger
import dask.dataframe as dd
from torch.utils.data import (Dataset)

logger = getLogger(__name__)
class QmeanDataset(Dataset):
    def __init__(self, csv_path: str , parquet_dir: str , force_rebuild: bool = False):


        if not os.path.exists(parquet_dir) or force_rebuild:
            logger.info("Creating Parquet Directory...")
            ddf = dd.read_csv(csv_path, usecols=["name", "sequence", "avg_local_score"])
            ddf.to_parquet(parquet_dir, engine="pyarrow", write_index=False)
            logger.info("Parquet Created")

        self.parquet_files = sorted(
            os.path.join(parquet_dir, f)
            for f in os.listdir(parquet_dir)
            if f.endswith(".parquet")
        )

        # Build row counts per shard
        self.rows_per_file = []
        for f in self.parquet_files:
            pf = pq.ParquetFile(f)
            self.rows_per_file.append(pf.metadata.num_rows)

        # Prefix sum for global indexing
        self.cum_rows = torch.tensor(self.rows_per_file).cumsum(0)

        # Cache for speed
        self._current_file_idx = None
        self._current_table = None

    def __len__(self):
        return int(self.cum_rows[-1])

    def _load_file(self, file_idx):
        if file_idx != self._current_file_idx:
            self._current_table = pq.read_table(self.parquet_files[file_idx])
            self._current_file_idx = file_idx

    def __getitem__(self, idx):
        # Find which parquet file this index belongs to
        file_idx = int(torch.searchsorted(self.cum_rows, idx, right=True))

        if file_idx == 0:
            row_idx = idx
        else:
            row_idx = idx - int(self.cum_rows[file_idx - 1])

        # Load shard if needed
        self._load_file(file_idx)

        # Extract row
        row = self._current_table.slice(row_idx, 1).to_pydict()

        x = row["sequence"]
        y = torch.tensor(row["avg_local_score"][0], dtype=torch.long)

        return x, y
