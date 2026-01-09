import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import  AutoModel

class ProtBerQmean(pl.LightningModule):
    def __init__(self, lr=1e-4):
        super().__init__()
        self.bert = AutoModel.from_pretrained('Rostlab/prot_bert')

        for p in self.bert.parameters():
            p.requires_grad = False
        hidden_size = self.bert.config.hidden_size

        #regression head per un solo ouput
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1)
        )

        #loss , usiamo MSE
        self.loss = nn.MSELoss()

        self.lr = lr


    def forward(self , input_ids , attention_mask):
        #output di Pbert
        out = self.bert(input_ids=input_ids , attention_mask=attention_mask)

        seq_embeddings = out.pooler_output

        pred = self.regressor(seq_embeddings).squeeze(-1)

        pred = torch.tanh(pred) #valori fra -1 e 1
        return pred

    def on_train_start(self):
        self.bert.train()

    def training_step(self, batch, batch_idx):
        if batch_idx == 0:
            print("bert in train mode?", self.bert.training)

        input_ids , attention_mask , y = batch
        y_hat = self(input_ids , attention_mask)
        loss = self.loss(y_hat , y)
        self.log("train_loss" , loss , prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, y = batch
        y_hat = self(input_ids, attention_mask)
        loss = self.loss(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss # FROCIO

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer