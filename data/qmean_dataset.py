import glob
import pickle
from pathlib import Path
from typing import List, Optional

import dask.dataframe as dd
import graphein.molecule as gm
import networkx as nx
import os
import pandas as pd
import pyarrow.parquet as pq
import shutil
import torch
import torch.nn.functional as F
from graphein.ml import GraphFormatConvertor
from logging import getLogger
from rdkit import Chem
from rdkit.Chem import PropertyMol
from torch_geometric.data import Data
from torch_geometric.data import Dataset as GraphDataset
from torch_geometric.utils import from_networkx
from torch.utils.data import Dataset

logger = getLogger(__name__)

# File di metadati PyG in processed/: non sono grafi e non vanno in train/val/test
PROCESSED_META_FILES = frozenset({"pre_filter.pt", "pre_transform.pt"})


def _is_data_pt(filename: str) -> bool:
    """True se il file è un .pt di dato (grafo), non pre_filter/pre_transform."""
    return filename.endswith(".pt") and filename not in PROCESSED_META_FILES


class QmeanDataset(Dataset):
    def __init__(self,
                 csv_path: str,
                 parquet_dir: str,
                 force_rebuild: bool = False):

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

        x = row["sequence"][0]
        y = torch.tensor(row["avg_local_score"][0])

        if "name" in row and row["name"]:
            name = row["name"][0]
        elif "id" in row and row["id"]:
            name = row["id"][0]
        else:
            name = str(idx)

        return x, y, name


class QmeanGraphDataset(GraphDataset):
    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 target_csv: str | None = None):
        self.targets = None
        if target_csv is not None:
            df = pd.read_csv(target_csv)
            self.targets = dict(zip(df['name'], df['avg_local_score']))
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    # file che vengono controllati per vedere se far partire il download
    def raw_file_names(self):
        return list(glob.glob(os.path.join(self.root, "*.p2smi")))

    @property
    def processed_file_names(self):
        # Se la cartella processed non esiste ancora, non ci sono file .pt
        if not os.path.isdir(self.processed_dir):
            return []
        # Solo .pt numerici (0.pt, 1.pt, ...); escludi pre_filter.pt / pre_transform.pt
        names = [
            f
            for f in os.listdir(self.processed_dir)
            if f.endswith(".pt")
            and f not in PROCESSED_META_FILES
            and os.path.isfile(os.path.join(self.processed_dir, f))
        ]
        # Ordine per indice numerico
        def _key(f):
            base = f.replace(".pt", "")
            return int(base) if base.isdigit() else -1

        return [os.path.join(self.processed_dir, f) for f in sorted(names, key=_key)]

    def process(self):
        # Read data into huge `Data` list.
        # Assicura che la cartella processed esista
        os.makedirs(self.processed_dir, exist_ok=True)

        files_to_process = os.listdir(self.root)
        for idx, file in enumerate(files_to_process):
            if not file.endswith(".p2smi"):
                continue
            with open(os.path.join(self.root, file)) as f:
                data = f.read()

            smiles = data.split(": ")[1]


            stem = Path(file).stem
            parts = stem.split("-")
            graph_name = parts[1] if len(parts) > 1 else stem

            config = gm.MoleculeGraphConfig()
            nx_graph = gm.construct_graph(smiles=smiles, config=config)
            # Conversione nx -> PyG:
            convertor = GraphFormatConvertor(
                src_format="nx",
                dst_format="pyg",
                columns=[
                    "edge_index",
                    "coords",
                    "name",
                    "node_id",
                    "atom_type_one_hot",
                ],
            )
            pyg_graph = convertor(nx_graph)
            # Costruiamo data.x a partire da atom_type_one_hot (tensor float)
            pyg_graph.x = torch.as_tensor(
                pyg_graph.atom_type_one_hot, dtype=torch.float
            )
            # Rimuoviamo l'attributo ridondante per avere un Data "pulito"
            if hasattr(pyg_graph, "atom_type_one_hot"):
                del pyg_graph.atom_type_one_hot

            # Nome proteina/grafo per CSV e log
            pyg_graph.name = graph_name

            target_value = None
            if self.targets is not None:
                namefile = graph_name
                for name, score in self.targets.items():
                    if name in namefile:
                        target_value = score
                        break
                if target_value is not None:
                    pyg_graph.y = torch.tensor([float(target_value)])
                else:
                    logger.warning("Target not found for %s", namefile)

            if self.pre_filter is not None:
                pyg_graph = self.pre_filter(pyg_graph)

            if self.pre_transform is not None:
                pyg_graph = self.pre_transform(pyg_graph)

            torch.save(pyg_graph, os.path.join(self.processed_dir, str(idx) + ".pt"))

    def __len__(self):
        return len(self.processed_file_names)

    def __getitem__(self, idx):
        data = torch.load(self.processed_file_names[idx], weights_only=False)
        return data


class QmeanGraphProcessedDataset(Dataset):
    """
    Dataset semplice che legge i grafi già processati da `processed_dir`.
    Può essere ristretto a un sottoinsieme tramite `files` o `indices`.
    """

    def __init__(
        self,
        processed_dir: str,
        files: Optional[List[str]] = None,
        indices: Optional[List[int]] = None,
    ):
        self.processed_dir = processed_dir

        if files is None:
            base_files = sorted(
                os.path.join(processed_dir, f)
                for f in os.listdir(processed_dir)
                if _is_data_pt(f)
            )
        else:
            base_files = list(files)

        if indices is not None:
            self.files = [base_files[i] for i in indices]
        else:
            self.files = base_files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], weights_only=False)
        return data


if __name__ == "__main__":
    qmean_graph_dataset = QmeanGraphDataset(root="../paolos",
                                            target_csv="../qmean_global_scores_clean.csv")



    print(qmean_graph_dataset.__getitem__(0))

    for idx, data in enumerate(qmean_graph_dataset):
        print(qmean_graph_dataset.__getitem__(idx).y)