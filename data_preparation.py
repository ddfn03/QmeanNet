import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser(description="Split train/val/test")
    parser.add_argument("--test-size", type=float, default=0.1, help="Values for the test set (0-1, of total)")
    parser.add_argument("--val-size", type=float, default=0.1, help="Values for the val set (0-1, of total))")
    parser.add_argument("--train-size", type=float, default=0.8, help="Values for the train set (0-1, of total))")
    parser.add_argument("--output-split-dir", type=str, default="_dataset", help="Path to directory containing the train, val and test csv")
    parser.add_argument("--csv-path", type=str, default="qmean_global_scores.csv", help="Path to the input CSV file")
    parser.add_argument("--random-state", type=int, default=42, help="Seed for random splitting")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    os.makedirs(args.output_split_dir, exist_ok=True)

    test_size = args.test_size
    val_size = args.val_size
    train_size = args.train_size

    if sum([train_size, test_size, val_size]) != 1.0:
        raise ValueError("train_split , val_split  and test_split sum must be equal to one")

    train_path = os.path.join(args.output_split_dir, "train.csv")
    val_path = os.path.join(args.output_split_dir, "val.csv")
    test_path = os.path.join(args.output_split_dir, "test.csv")

    random_state = args.random_state
    csv_path = args.csv_path

    csv = pd.read_csv(csv_path)

    train_p, test_p = train_test_split(csv, test_size=test_size, random_state=random_state)
    train_p, val_p = train_test_split(train_p, test_size=val_size / train_size,
                                      random_state=random_state)

    train_p.to_csv(train_path, index=False)
    val_p.to_csv(val_path, index=False)
    test_p.to_csv(test_path, index=False)

    '''if not os.path.exists(parquet_dir):
        logger.info("Creating Parquet Directory...")
        ddf = dd.read_csv(csv_path, usecols=["name", "sequence", "avg_local_score"])
        ddf.to_parquet(parquet_dir, engine="pyarrow", write_index=False)
        logger.info("Parquet Created")
    
    self.proteins = pd.read_parquet(parquet_dir, engine="pyarrow")'''
