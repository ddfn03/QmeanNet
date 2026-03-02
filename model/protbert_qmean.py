from typing import Literal, Optional

import lightning as pl
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
from transformers import AutoModel


class ProtBerQmean(pl.LightningModule):
    def __init__(
            self,
            lr: float = 1e-4,
            weight_decay: float = 1e-2,
            model_name: str = "Rostlab/prot_bert",
            freeze_bert: bool = True,
            dropout_rate: float = 0.1,
            n_regressor_layers: int = 1,
            use_gnn: Optional[str] = None,
            gnn_in_channels: Optional[int] = None,
            gnn_hidden_dim: int = 128,
            gnn_num_layers: int = 1,
    ):
        super().__init__()

        self.use_gnn = use_gnn
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        self.lr = lr

        if self.use_gnn == "GCN":
            if gnn_in_channels is None:
                raise ValueError(
                    "gnn_in_channels must be specified when use_gnn parameter is used. "
                    "it is equal to the data features  (data.x.shape[1])."
                )

            hidden_dim = gnn_hidden_dim


            self.gcn_convs = nn.ModuleList()
            self.gcn_convs.append(GCNConv(gnn_in_channels, hidden_dim))
            
            self.gcn_blocks = nn.ModuleList()
            for i in range(gnn_num_layers - 1):
                self.gcn_blocks.append(
                    nn.Sequential(
                        GCNConv(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout_rate),
                    )
                )

            self.gcn_regressor = nn.Linear(hidden_dim, 1)
            

        else:
            # Modello ProtBert
            self.bert = AutoModel.from_pretrained(model_name)

            self.freeze_bert = freeze_bert
            if self.freeze_bert:
                for p in self.bert.parameters():
                    p.requires_grad = False

            hidden_size = self.bert.config.hidden_size

            self.regressor = nn.Sequential()


            for i in range(n_regressor_layers - 1):
                self.regressor.append(nn.Linear(hidden_size, hidden_size))
                self.regressor.append(nn.ReLU())
                self.regressor.append(nn.Dropout(dropout_rate))

            #output
            self.regressor.append(nn.Linear(hidden_size, 1))

        self.loss = nn.MSELoss()
        self.mae = torch.nn.L1Loss()

        self.save_hyperparameters()

    def forward(self, *args, **kwargs):
        #uso con gnn
        if self.use_gnn == "GCN":
            data = args[0] if args else kwargs.get("data")
            x, edge_index, batch = data.x, data.edge_index, data.batch

            for conv in self.gcn_convs:
                x = conv(x, edge_index)
                x = torch.relu(x)
                x = self.gcn_dropout(x)


            graph_embeddings = global_mean_pool(x, batch)
            pred = self.gcn_regressor(graph_embeddings).squeeze(-1)
            pred = torch.tanh(pred)  # valori fra -1 e 1
            return pred

        #protbert classico
        input_ids, attention_mask = args
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq_embeddings = out.pooler_output
        pred = self.regressor(seq_embeddings).squeeze(-1)
        pred = torch.tanh(pred)  # valori fra -1 e 1
        return pred

    def _common_step(self, batch, prefix: Literal["train", "val", "test"]):
        if self.use_gnn == "GCN":
            # batch è un oggetto Batch di torch_geometric, con y già dentro
            data = batch
            y = data.y.view(-1)
            y_hat = self(data)
            batch_size = y.shape[0]
        else:
            input_ids, attention_mask, y, _ = batch
            y_hat = self(input_ids, attention_mask)
            batch_size = input_ids.shape[0]

        loss = self.loss(y_hat, y)
        mae = self.mae(y_hat, y).item()
        self.log_dict(
            {f"{prefix}/loss": loss, f"{prefix}/mae": mae},
            prog_bar=True,
            batch_size=batch_size,
        )
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
