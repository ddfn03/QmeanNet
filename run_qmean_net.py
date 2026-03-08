import os
import sys
import torch
from lightning.pytorch.cli import LightningCLI

from model.protbert_qmean import ProtBerQmean
from data.qmean_datamodule import QmeanDataModule, QmeanGraphDataModule


def _use_gnn_config() -> bool:
    """True se in argv c'è un config per GNN (es. config_gnn.yaml)."""
    argv = sys.argv
    if "--config" not in argv:
        return False
    idx = argv.index("--config")
    if idx + 1 >= len(argv):
        return False
    path = argv[idx + 1].lower()
    return "gnn" in path or "config_gnn" in path


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    use_gnn = _use_gnn_config()
    datamodule_class = QmeanGraphDataModule if use_gnn else QmeanDataModule

    LightningCLI(
        model_class=ProtBerQmean,
        datamodule_class=datamodule_class,
        save_config_kwargs={"overwrite": True},
    )


