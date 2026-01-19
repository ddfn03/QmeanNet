import logging

import lightning as pl
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .qmean_dataset import QmeanDataset

logger = logging.getLogger(__name__)


class QmeanDataModule(pl.LightningDataModule):
    def __init__(self, train_path: str = "_dataset/train.csv", val_path: str = "_dataset/val.csv",
                 test_path: str = "_dataset/test.csv", parquet_dir: str = "_dataset/parquet_shards",
                 batch_size: int = 4, max_sequence_len: int = 512, tokenizer: str = "Rostlab/prot_bert",
                 num_workers: int = 8):
        super().__init__()
        self.parquet_dir = parquet_dir

        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

        self.batch_size = batch_size
        self.max_sequence_len = max_sequence_len
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.num_workers = num_workers

    def setup(self, stage: str):
        if stage == "fit":
            self.train_ds = QmeanDataset(self.train_path, self.parquet_dir + "/train")
            self.val_ds = QmeanDataset(self.val_path, self.parquet_dir + "/val")
        else:
            self.test_ds = QmeanDataset(self.test_path, self.parquet_dir + "/test")

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
                          shuffle=True, collate_fn=self.collate_fn, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size,
                          shuffle=False, collate_fn=self.collate_fn, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size,
                          shuffle=False, collate_fn=self.collate_fn, num_workers=self.num_workers)
