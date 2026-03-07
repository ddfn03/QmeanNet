from typing import Literal, Optional

import lightning as pl
import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.models import GAT, GCN, GIN, GraphSAGE
from transformers import AutoConfig, AutoModel

# Mappa use_gnn -> classe modello PyG (stessa interfaccia: in_channels, hidden_channels, num_layers, out_channels, dropout)
GNN_MODELS = {
    "GCN": GCN,
    "GraphSAGE": GraphSAGE,
    "GIN": GIN,
    "GAT": GAT
}


def _build_regressor(
    embedding_dim: int,
    n_regressor_layers: int,
    dropout_rate: float,
) -> nn.Sequential:


    regressor = nn.Sequential()
    for _ in range(n_regressor_layers - 1):
        regressor.append(nn.Linear(embedding_dim, embedding_dim))
        regressor.append(nn.ReLU())
        regressor.append(nn.Dropout(dropout_rate))
    regressor.append(nn.Linear(embedding_dim, 1))
    return regressor


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

        # Dimensione embedding condivisa (da config BERT così il regressore è identico per entrambi gli approcci)
        config = AutoConfig.from_pretrained(model_name)
        self.embedding_dim = config.hidden_size

        # Regressore unico per entrambi i modelli
        self.regressor = _build_regressor(
            self.embedding_dim, n_regressor_layers, dropout_rate
        )

        if self.use_gnn is not None:
            if self.use_gnn not in GNN_MODELS:
                raise ValueError(
                    f"use_gnn must be part of {list(GNN_MODELS.keys())}, received: {self.use_gnn!r}"
                )
            if gnn_in_channels is None:
                raise ValueError(
                    "gnn_in_channels must be specified when use_gnn parameter is used."
                    "it is equal to the data features  (data.x.shape[1])."
                )

            gnn_cls = GNN_MODELS[self.use_gnn]

            self.gnn_encoder = gnn_cls(
                in_channels=gnn_in_channels,
                hidden_channels=gnn_hidden_dim,
                num_layers=gnn_num_layers,
                out_channels=self.embedding_dim,
                dropout=dropout_rate,
            )
        else:
            self.bert = AutoModel.from_pretrained(model_name)
            self.freeze_bert = freeze_bert
            if self.freeze_bert:
                for p in self.bert.parameters():
                    p.requires_grad = False

        self.loss = nn.MSELoss()
        self.mae = nn.L1Loss()

        self.save_hyperparameters()

    def forward(self, *args, **kwargs):
        if self.use_gnn is not None:
            data = args[0] if args else kwargs.get("data")
            x, edge_index, batch = data.x, data.edge_index, data.batch
            # Encoder PyG (GCN / GraphSAGE / GIN / GAT) -> embedding per nodo
            node_emb = self.gnn_encoder(x, edge_index, batch=batch)
            graph_emb = global_mean_pool(node_emb, batch)
            pred = self.regressor(graph_emb).squeeze(-1)
            return torch.tanh(pred)

        input_ids, attention_mask = args[0], args[1]
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq_embeddings = out.pooler_output
        pred = self.regressor(seq_embeddings).squeeze(-1)
        return torch.tanh(pred)

    def _common_step(self, batch, prefix: Literal["train", "val", "test"]):
        if self.use_gnn is not None:
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
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=self.trainer.estimated_stepping_batches,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
            },
        }
