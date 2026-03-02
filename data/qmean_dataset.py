import glob
import pickle
from pathlib import Path
import pandas as pd
import networkx as nx
import pyarrow.parquet as pq
import os
import torch
import shutil
from logging import getLogger
import dask.dataframe as dd
import graphein.molecule as gm
from graphein.ml import GraphFormatConvertor
from rdkit import Chem
from rdkit.Chem import PropertyMol
from torch_geometric.data import Dataset as GraphDataset
from torch.utils.data import (Dataset)
from torch_geometric.utils import from_networkx

logger = getLogger(__name__)
class QmeanDataset(Dataset):
    def __init__(self, csv_path: str , parquet_dir: str , force_rebuild: bool = False):


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

        return x, y , name



class QmeanGraphDataset(GraphDataset):
        def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, target_csv: str | None = None):
            self.targets = None
            if target_csv is not None:
                df = pd.read_csv(target_csv)
                self.targets = dict(zip(df['name'], df['avg_local_score']))
            super().__init__(root, transform, pre_transform, pre_filter)

        @property
        #file che vengono controllati per vedere se far partire il download
        def raw_file_names(self):
            return list(glob.glob(os.path.join(self.root, "*.p2smi")))

        @property
        #after processing
        def processed_file_names(self):
            return list(glob.glob(os.path.join(self.processed_dir, "*.pt")))


        def process(self):
            # Read data into huge `Data` list.
            #ggiungere splitting dei grafi
            for idx, file in enumerate(os.listdir(self.root)):
                if not file.endswith(".p2smi"):
                    continue
                with open(os.path.join(self.root , file)) as f:
                    data = f.read()

                data = data.split(": ")
                data = data[1]

                config = gm.MoleculeGraphConfig()
                nx_graph = gm.construct_graph(smiles=data, config=config)
                #restituisce non un grafo pyg ma nx

                convertor = GraphFormatConvertor(src_format="nx", dst_format="pyg")
                pyg_graph = convertor(nx_graph)


                # attach target by substring match between CSV name and file stem
                target_value = None
                if self.targets is not None:
                    namefile = Path(file).stem
                    for name , score in self.targets.items():
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

                torch.save(pyg_graph, os.path.join(self.processed_dir , str(idx) + ".pt"))

        def __len__(self):
            return len(self.processed_file_names)

        def __getitem__(self, idx):
            data = torch.load(os.path.join(self.processed_dir , str(idx) +".pt"), weights_only=False)
            return data

if __name__ == "__main__":
    qmean_graph_dataset = QmeanGraphDataset(root="/Users/davidedelfranconatale/QmeanNet/paolos" , target_csv="../qmean_global_scores_clean.csv")

    print(qmean_graph_dataset.__getitem__(0))

    for idx, data in enumerate(qmean_graph_dataset):
        print(qmean_graph_dataset.__getitem__(idx).y)