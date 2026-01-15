import logging
import os

import dask.dataframe as dd
import pandas as pd
import lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .qmean_dataset import QmeanDataset

logger = logging.getLogger(__name__)


class QmeanDataModule(pl.LightningDataModule):
    def __init__(self, csv_path: str = "qmean_global_scores.csv", parquet_dir: str = "qmean_global_scores.parquet",
                 batch_size: int = 4, max_sequence_len: int = 512, train_split: float = 0.8, val_split: float = 0.1,
                 test_split: float = 0.1,
                 random_state: int = 42):
        super().__init__()

        if not os.path.exists(parquet_dir):
            logger.info("Creating Parquet Directory...")
            ddf = dd.read_csv(csv_path, usecols=["name", "sequence", "avg_local_score"])
            ddf.to_parquet(parquet_dir, engine="pyarrow", write_index=False)
            logger.info("Parquet Created")

        self.proteins = pd.read_parquet(parquet_dir, engine="pyarrow")
        self.q = self.proteins["avg_local_score"]
        self.qmin, self.qmax = self.q.min(), self.q.max()
        logger.info(f"Dataset: {len(self.proteins)} rows")

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

        self.train_size = train_split
        self.val_size = val_split
        self.test_size = test_split
        self.random_state = random_state
        if sum([train_split, test_split, val_split]) != 1.0:
            raise ValueError("train_split , val_split  and test_split sum must be equal to one")

        self.batch_size = batch_size
        self.max_sequence_len = max_sequence_len
        self.tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert")

    def setup(self, stage: str):
        train_p, test_p = train_test_split(self.proteins, test_size=self.test_size, random_state=self.random_state)
        train_p, val_p = train_test_split(train_p, test_size=self.val_size / self.train_size,
                                          random_state=self.random_state)
        if stage == "fit":
            self.train_ds = QmeanDataset(train_p, self.qmin, self.qmax)
            self.val_ds = QmeanDataset(val_p, self.qmin, self.qmax)
        else:
            self.test_ds = QmeanDataset(test_p, self.qmin, self.qmax)

    def collate_fn(self, batch):
        seqs, ys = zip(*batch)
        seps_spaced = [" ".join(list(s)) for s in seqs]  # formato in cui ProtBert prende gli input

        enc = self.tokenizer(
            seps_spaced,
            padding=True,
            truncation=True,
            max_length=self.max_sequence_len,
            return_tensors="pt"
        )
        y = torch.stack(ys)
        return enc["input_ids"], enc["attention_mask"], y

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size,
                          shuffle=True, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size,
                          shuffle=False, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size,
                          shuffle=False, collate_fn=self.collate_fn)
