import logging
import os
import random
import subprocess
import time
from argparse import ArgumentParser

import pandas as pd
import wandb
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint
from scipy.conftest import devices
from sklearn.model_selection import train_test_split
from wandb.integration.lightning.fabric import WandbLogger

from data.qmean_datamodule import QmeanDataModule
from model import ProtBerQmean

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def train(model, datamodule, model_prefix, project, entity, offline, group, patience, k, max_epochs, min_delta,
          lr_log_interval, devices , default_root_dir,):
    callbacks = [
        LearningRateMonitor(logging_interval=lr_log_interval),
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            min_delta=min_delta,
            verbose=False,
            mode='min'
        ),
        ModelCheckpoint(
            filename=f"{model_prefix}-fold-{k + 1}-{{epoch:02d}}-{{val_loss:.2f}}",
            monitor='val_loss',
            mode='min',
            save_top_k=1,
        )
    ]

    wandb_logger = WandbLogger(
        project=project,
        entity=entity,
        offline=offline,
        group=group,
        name=f"{model_prefix}-fold-{k + 1}"
    )

    trainer = Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        default_root_dir=default_root_dir,
        logger=wandb_logger,
        devices=devices,
    )

    trainer.fit(model, datamodule=datamodule)
    wandb.finish()

    return trainer.logger.experiment.id


def main(args):
    if sum(args.split) != 100:
        raise ValueError("train_split , val_split  and test_split sum must be equal to 100")

    logger.info("=" * 80)
    logger.info(f"Starting Cross-Validation with {args.k} folds - Model: {args.model_prefix}")

    seed_everything(args.seed)
    random.seed(args.seed)

    logger.info(f"Random seed set to {args.seed}")
    logger.info("=" * 80)

    df = pd.read_csv(args.csv_path)

    logger.info(f"Fold distribution - Train: {args.split[0]}%, Val: {args.split[1]}%, Test: {args.split[2]}%")

    for k in range(args.k):
        logger.info("=" * 80)
        logger.info(f"[Fold {k + 1}/{args.k}] Starting training and evaluation...")
        logger.info("=" * 80)

        train_split, test_split = train_test_split(df, args.split, test_size=args.split[2], random_state=args.seed + k)
        train_split, val_split = train_test_split(df, args.split, test_size=args.split[1] / args.split[0],
                                                  random_state=args.seed + k)

        unique_id = time.time()

        data_dir = os.path.join(args.data_dir, str(unique_id))
        os.makedirs(data_dir, exist_ok=True)

        train_path = os.path.join(data_dir, "train.csv")
        val_path = os.path.join(data_dir, "val.csv")
        test_path = os.path.join(data_dir, "test.csv")

        train_split.to_csv(train_path, index=False)
        val_split.to_csv(val_path, index=False)
        test_split.to_csv(test_path, index=False)

        datamodule = QmeanDataModule(train_path, val_path, test_path, args.parquet_dir, args.batch_size,
                                     args.num_workers, args.tokenizer, args.max_sequence_len)
        model = ProtBerQmean(args.lr, args.weight_decay, args.model_name, args.freeze_bert, args.dropout,
                             args.n_regressor_layers)

        logger.info(f"[Fold {k + 1}/{args.k}] Starting QmeanNet Training...")

        experiment_id = train(model=model, datamodule=datamodule, model_prefix=args.model_prefix, project=args.project,
                              entity=args.entity, offline=args.offline,
                              group=args.group, patience=args.patience, min_delta=args.min_delta, k=k,
                              lr_log_interval=args.lr_logging_interval, devices=args.devices,
                              max_epochs=args.max_epochs, defaul_root_dir=args.default_root_dir,
                              )

        logger.info(f"[Fold {k + 1}/{args.k}] QmeanNet Training Ended...")

        logger.info(f"[Fold {k + 1}/{args.k}] QmeanNet Starting Testing...")

        # get last model checkpoint from the current run
        ckpt_dir = os.path.join(args.rpn_default_root_dir, experiment_id, "checkpoints")  # noqa
        ckpt_path = [os.path.join(ckpt_dir, ckpt) for ckpt in os.listdir(ckpt_dir) if ckpt.endswith(".ckpt")][
            -1]  # noqa

        cmd = [
            "python", "test.py",
            "--seed", str(args.seed),
            "--batch_size", str(args.rpn_batch_size),
            "--ckpt_path", ckpt_path,
            "--model_name", f"{args.model_prefix}-{k + 1}-fold",
            "--scores_dir", "./_qmean_scores",
            "--scores_file", "scores.csv",
        ]


        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    parser = ArgumentParser(description="Perform k-fold Cross-validation for QmeanNet")

    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--k", type=int, default=5, help="Number of folds")
    parser.add_argument("--split", nargs=3, type=int, default=(80, 10, 10),
                        help="Percentage split for train, val and test sets")
    parser.add_argument("--data_dir", default="_dataset_cross", type=str, help="Path to the data directory")

    # dataloader
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for dataloaders")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of workers for dataloaders")
    parser.add_argument("--max-sequence-len", type=int, default=512, help="Maximum sequence length for tokenizer")
    parser.add_argument("--csv_path", type=str, default="qmean_global_scores_clean-.csv", help="Path to the csv file")
    parser.add_argument("--tokenizer", type=str, default="Rostlab/prot_bert", help="Model tokenizer")
    parser.add_argument("--model_name", type=str, default="Rostlab/prot_bert", help="Model name")
    parser.add_argument("--parquet_dir", type=str, default="parquet_shards", help="Path to the parquet directory")

    #wandb
    parser.add_argument("--model-prefix", type=str, default="qmean", help="Prefix per i checkpoint e run name")
    parser.add_argument("--project", type=str, default="qmean_project", help="WandB project name")
    parser.add_argument("--entity", type=str, default=None, help="WandB entity (team/user)")
    parser.add_argument("--offline", action="store_true", help="Launch wandb in offline mode")
    parser.add_argument("--group", type=str, default=None, help="WandB group name")

    #model_parameters
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay for optimizer")
    parser.add_argument("--freeze-bert", action="store_true", help="Freeze BERT encoder weights")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate for the model")
    parser.add_argument("--n-regressor-layers", type=int, default=1, help="Number of regressor layers")

    parser.add_argument("--patience", type=int, default=10, help="EarlyStopping patience")
    parser.add_argument("--min-delta", type=float, default=0.0, help="EarlyStopping min_delta")
    parser.add_argument("--lr-logging-interval", type=str, choices=["step", "epoch"], default="epoch",
                        help="Logging interval for LearningRateMonitor")

    parser.add_argument("--devices", type=int, default=1, help="Number of devices for Trainer")
    parser.add_argument("--max-epochs", type=int, default=50, help="Maximum number of training epochs")
    parser.add_argument("--default-root-dir", type=str, default=".", help="Root dir for Trainer outputs")

