import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from QmeanDataset import QmeanDataset


class QmeanDataModule(pl.LightningDataModule):
    def __init__(self, df_train , df_val , df_test , batch_size = 4 , max_len = 512):
        super().__init__()
        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test
        self.batch_size = batch_size
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert")


    def setup(self , stage=None):
        self.train_ds = QmeanDataset(self.df_train)
        self.val_ds = QmeanDataset(self.df_val)
        self.test_ds = QmeanDataset(self.df_test)


    def collate_fn(self, batch):
        seqs , ys = zip(*batch)
        seps_spaced = [" ".join(list(s)) for s in seqs] #formato in cui ProtBert prende gli input

        enc = self.tokenizer(
            seps_spaced,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        y = torch.stack(ys)
        return enc["input_ids"] , enc["attention_mask"] , y

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size,
                          shuffle=True, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size,
                          shuffle=False, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size,
                          shuffle=False, collate_fn=self.collate_fn)