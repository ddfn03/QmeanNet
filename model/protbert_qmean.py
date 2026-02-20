from typing import Literal

import lightning as pl
import torch
import torch.nn as nn
from transformers import AutoModel


class ProtBerQmean(pl.LightningModule):
    def __init__(self, lr: float = 1e-4, weight_decay: float = 1e-2, model_name: str = 'Rostlab/prot_bert',
                 freeze_bert: bool = True, dropout_rate: float = 0.1, n_regressor_layers: int = 1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)

        self.freeze_bert = freeze_bert
        if self.freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

        hidden_size = self.bert.config.hidden_size

        self.regressor = nn.Sequential()

        for i in range(n_regressor_layers-1):
            self.regressor.append(nn.Linear(hidden_size, hidden_size))
            self.regressor.append(nn.ReLU())
            self.regressor.append(nn.Dropout(dropout_rate))

        self.regressor.append(nn.Linear(hidden_size, 1))
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        self.loss = nn.MSELoss()
        self.mae = torch.nn.L1Loss()
        self.lr = lr

        self.save_hyperparameters()

    def forward(self, input_ids, attention_mask):
        # output di Pbert
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq_embeddings = out.pooler_output
        pred = self.regressor(seq_embeddings).squeeze(-1)

        pred = torch.tanh(pred)  # valori fra -1 e 1
        return pred

    def _common_step(self, batch, prefix: Literal["train", "val", "test"]):
        input_ids, attention_mask, y, _ = batch
        y_hat = self(input_ids, attention_mask)
        loss = self.loss(y_hat, y)
        mae = self.mae(y_hat, y).item()
        self.log_dict({f"{prefix}/loss": loss, f"{prefix}/mae": mae}, prog_bar=True, batch_size=input_ids.shape[0])
        return loss

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, prefix="train")

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, prefix="val")

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, prefix="test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                               T_max=self.trainer.estimated_stepping_batches)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                 "monitor": "val/loss",
            }
        }
