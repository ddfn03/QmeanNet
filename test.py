import csv
import os
from argparse import ArgumentParser

import torch
from lightning import seed_everything
from tqdm import tqdm

from data.qmean_datamodule import QmeanDataModule
from model import ProtBerQmean


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def test(args):
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
    fold_csv_dir = os.path.join(args.scorse_dir, f"predictions")

    os.makedirs(fold_csv_dir, exist_ok=True)

    fold_csv_path = os.path.join(fold_csv_dir, f"fold_{model_name}.csv")

    global_scores = {}

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing Batches")):
        input_ids, attention_mask, y = batch

        input_ids = input_ids.to(device=model.device)
        attention_mask = attention_mask.to(device=model.device)
        y = y.to(device=model.device)

        with torch.no_grad():
            y_hat = model(input_ids, attention_mask)

        mae = torch.nn.functional.l1_loss(y_hat, y).item()

        global_scores["mae"] = mae


    scores_path = os.path.join(args.scores_dir, args.scores_file)
    file_exists = os.path.exists(scores_path)


    global_scores = {"model": model_name, **global_scores}

    with open(scores_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=global_scores.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(global_scores)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--test_path", type=str, required=True)

    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default=None)

    parser.add_argument("--scores_dir", type=str, default="./_qmean_scores")
    parser.add_argument("--scores_file", type=str, default="scores.csv")
    parser.add_argument("--tokenizer", type=str, default="Rostlab/prot_bert", help="Model tokenizer")
    parser.add_argument("--parquet_dir", type=str, default="parquet_shards", help="Path to the parquet directory")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of workers for dataloaders")
    parser.add_argument("--max-sequence-len", type=int, default=512, help="Maximum sequence length for tokenizer")

    test(parser.parse_args())
