import csv
import os
from argparse import ArgumentParser

import torch
from lightning import seed_everything
from tqdm import tqdm

from data.qmean_datamodule import QmeanDataModule, QmeanGraphDataModule
from model.protbert_qmean import ProtBerQmean


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _test_bert(args):
    seed_everything(args.seed)

    model = ProtBerQmean.load_from_checkpoint(args.ckpt_path)
    model = model.to(device=get_device())
    model.eval()
    model_name = args.model_name or "_".join(args.ckpt_path.split(os.path.sep)[-1].split("-")[:2])

    datamodule = QmeanDataModule(test_path=args.test_path, parquet_dir=args.parquet_dir, batch_size=args.batch_size,
                                 num_workers=args.num_workers, tokenizer=args.tokenizer,
                                 max_sequence_len=args.max_sequence_len)
    datamodule.setup(stage="test")
    dataloader = datamodule.test_dataloader()

    os.makedirs(args.scores_dir, exist_ok=True)
    scores_path = os.path.join(args.scores_dir, args.scores_file)
    file_exists = os.path.exists(scores_path)

    fieldnames = ["model", "protein_name", "pred", "qmean", "mae"]

    with open(scores_path, "a", newline="") as f_scores:
        writer_scores = csv.DictWriter(f_scores, fieldnames=fieldnames)
        if not file_exists:
            writer_scores.writeheader()

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing Batches (BERT)")):
            input_ids, attention_mask, y, names = batch  # noqa

            input_ids = input_ids.to(device=model.device)
            attention_mask = attention_mask.to(device=model.device)
            y = y.to(device=model.device)

            with torch.no_grad():
                y_hat = model(input_ids, attention_mask)

            batch_size = y.size(0)
            for i in range(batch_size):
                pred = y_hat[i].item()
                true = y[i].item()

                n = names[i]
                if isinstance(n, torch.Tensor):
                    try:
                        prot_name = str(n.item())
                    except Exception:  # noqa
                        prot_name = str(n.tolist())
                elif isinstance(n, (bytes, bytearray)):
                    prot_name = n.decode("utf-8", errors="replace")
                else:
                    prot_name = str(n)

                row = {
                    "protein_name": prot_name,
                    "pred": float(pred),
                    "qmean": float(true),
                    "mae": torch.nn.functional.l1_loss(y_hat[i], y[i]).item(),
                    "model": model_name,
                }
                writer_scores.writerow(row)


def _test_gnn(args):
    seed_everything(args.seed)

    model = ProtBerQmean.load_from_checkpoint(args.ckpt_path)
    model = model.to(device=get_device())
    model.eval()
    model_name = args.model_name or "_".join(args.ckpt_path.split(os.path.sep)[-1].split("-")[:2])

    datamodule = QmeanGraphDataModule(
        root=args.gnn_root,
        target_csv=args.gnn_target_csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    datamodule.setup(stage="test")
    dataloader = datamodule.test_dataloader()

    os.makedirs(args.scores_dir, exist_ok=True)
    scores_path = os.path.join(args.scores_dir, args.scores_file)
    file_exists = os.path.exists(scores_path)

    fieldnames = ["model", "protein_name", "pred", "qmean", "mae"]

    with open(scores_path, "a", newline="") as f_scores:
        writer_scores = csv.DictWriter(f_scores, fieldnames=fieldnames)
        if not file_exists:
            writer_scores.writeheader()

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing Batches (GNN)")):
            data = batch.to(get_device())
            y = data.y.view(-1)

            with torch.no_grad():
                y_hat = model(data)

            batch_size = y.size(0)
            names = getattr(data, "name", None)
            for i in range(batch_size):
                if names is not None and i < len(names):
                    prot_name = names[i] if isinstance(names[i], str) else str(names[i])
                else:
                    prot_name = f"graph_{i}"
                row = {
                    "protein_name": prot_name,
                    "pred": float(y_hat[i].item()),
                    "qmean": float(y[i].item()),
                    "mae": torch.nn.functional.l1_loss(y_hat[i], y[i]).item(),
                    "model": model_name,
                }
                writer_scores.writerow(row)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--test_path", type=str, required=False)

    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default=None)

    parser.add_argument("--scores_dir", type=str, default="./_qmean_scores")
    parser.add_argument("--scores_file", type=str, default="scores.csv")
    parser.add_argument("--tokenizer", type=str, default="Rostlab/prot_bert", help="Model tokenizer")
    parser.add_argument("--parquet_dir", type=str, default="parquet_shards", help="Path to the parquet directory")
    parser.add_argument("--max_sequence_len", "--max-sequence-len", type=int, default=512, dest="max_sequence_len",
                        help="Maximum sequence length for tokenizer")

    # GNN:
    parser.add_argument("--gnn_type", type=str, default=None,
                        choices=["GCN", "GraphSAGE", "GIN", "GAT"],
                        help="Use GNN model(specify the type of GNN)")
    parser.add_argument("--gnn_root", type=str, default="smiles", help="Root directory with graph data (.p2smi)")
    parser.add_argument("--gnn_target_csv", type=str, default="qmean_global_scores_clean.csv",
                        help="CSV with targets for graphs")
    parser.add_argument("--gnn_in_channels", type=int, default=11, help="Node feature dimension (saved in ckpt, per coerenza CLI)")
    parser.add_argument("--gnn_hidden_dim", type=int, default=128, help="GNN hidden dim (saved in ckpt, per coerenza CLI)")
    parser.add_argument("--gnn_num_layers", type=int, default=2, help="GNN layers (saved in ckpt, per coerenza CLI)")

    args = parser.parse_args()

    gnn_mode = args.gnn_type is not None and args.gnn_type in ("GCN", "GraphSAGE", "GIN", "GAT")
    if gnn_mode:
        _test_gnn(args)
    else:
        if not args.test_path:
            raise SystemExit("--test_path is required when not using --gnn_type")
        _test_bert(args)
