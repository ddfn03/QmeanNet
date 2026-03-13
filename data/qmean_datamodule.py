import logging
import os
from typing import Iterable, Optional, List

import lightning as pl
import torch
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GraphDataLoader
from transformers import AutoTokenizer

from .qmean_dataset import (
    PROCESSED_META_FILES,
    QmeanDataset,
    QmeanGraphDataset,
    QmeanGraphProcessedDataset,
    _is_data_pt,
)

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

    def _normalize_names(self, names: Iterable, fallback_prefix: Optional[str] = "unknown") -> List[str]:
        out: List[str] = []
        for i, n in enumerate(names):
            try:
                if isinstance(n, torch.Tensor):
                    try:
                        val = n.item()
                    except Exception:
                        val = n.tolist()
                    s = str(val)
                elif isinstance(n, (bytes, bytearray)):
                    s = n.decode("utf-8", errors="replace")
                elif n is None:
                    s = f"{fallback_prefix}_{i}"
                else:
                    s = str(n)
            except Exception:
                s = f"{fallback_prefix}_{i}"
            out.append(s.strip())
        return out

    def collate_fn(self, batch):
        seqs, ys, names = zip(*batch)
        seps_spaced = [" ".join(list(s)) for s in seqs]  # formato in cui ProtBert prende gli input

        enc = self.tokenizer(
            seps_spaced,
            padding=True,
            truncation=True,
            max_length=self.max_sequence_len,
            return_tensors="pt"
        )
        y = torch.stack(ys)
        norm_names = self._normalize_names(names)
        return enc["input_ids"], enc["attention_mask"], y, norm_names

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size,
                          shuffle=True, collate_fn=self.collate_fn, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size,
                          shuffle=False, collate_fn=self.collate_fn, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size,
                          shuffle=False, collate_fn=self.collate_fn, num_workers=self.num_workers)


class QmeanGraphDataModule(pl.LightningDataModule):
    """
    DataModule per i grafi che:
    - costruisce tutti i grafi in `root/processed/*.pt` usando QmeanGraphDataset
    - effettua lo splitting in train/val/test creando:
        root/processed/train/*.pt
        root/processed/val/*.pt
        root/processed/test/*.pt
      (file disgiunti tra loro)
    - usa QmeanGraphProcessedDataset per caricare le tre split.
    """

    def __init__(
        self,
        root: str = "_graph_dataset",
        target_csv: Optional[str] = None,
        batch_size: int = 4,
        num_workers: int = 8,
        train_fraction: float = 0.8,
        val_fraction: float = 0.1,
        test_fraction: float = 0.1,
        seed: int = 42,
    ):
        super().__init__()

        self.root = root
        self.target_csv = target_csv

        self.train_fraction = train_fraction
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.seed = seed

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.processed_dir: Optional[str] = None

    def _prepare_splits(self) -> tuple[str, List[str]]:
        """
        Costruisce, se necessario, tutti i grafi in `root/processed/*.pt`
        usando QmeanGraphDataset, e restituisce la lista completa dei file `.pt`.
        Nessuno split fisico su disco: lo split train/val/test viene fatto
        in memoria tramite indici.
        """
        processed_dir = os.path.join(self.root, "processed")

        # Se non esistono ancora file .pt di dati, crea i grafi con QmeanGraphDataset
        if not os.path.isdir(processed_dir) or not any(
            _is_data_pt(f) for f in os.listdir(processed_dir)
        ):
            base_ds = QmeanGraphDataset(self.root, target_csv=self.target_csv)
            processed_dir = base_ds.processed_dir

        # Solo .pt dati in processed/ (escludi pre_filter.pt, pre_transform.pt)
        all_files: List[str] = [
            os.path.join(processed_dir, f)
            for f in os.listdir(processed_dir)
            if _is_data_pt(f) and os.path.isfile(os.path.join(processed_dir, f))
        ]

        if not all_files:
            logger.warning(
                "Nessun file .pt trovato in %s. "
                "Assicurati che QmeanGraphDataset abbia processato i grafi.",
                processed_dir,
            )

        return processed_dir, all_files

    def setup(self, stage: str):
        processed_dir, all_files = self._prepare_splits()
        self.processed_dir = processed_dir

        n = len(all_files)
        if n == 0:
            logger.warning("Nessun grafo disponibile in %s", processed_dir)
            return

        # Shuffle deterministico basato su seed
        g = torch.Generator()
        g.manual_seed(self.seed)
        perm = torch.randperm(n, generator=g).tolist()

        n_train = int(n * self.train_fraction)
        n_val = int(n * self.val_fraction)
        n_test = n - n_train - n_val

        train_idx = perm[:n_train]
        val_idx = perm[n_train : n_train + n_val]
        test_idx = perm[n_train + n_val :]

        if stage == "fit":
            self.train_ds = QmeanGraphProcessedDataset(
                processed_dir, files=all_files, indices=train_idx
            )
            self.val_ds = QmeanGraphProcessedDataset(
                processed_dir, files=all_files, indices=val_idx
            )
        else:
            self.test_ds = QmeanGraphProcessedDataset(
                processed_dir, files=all_files, indices=test_idx
            )

    def train_dataloader(self):
        return GraphDataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return GraphDataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return GraphDataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
