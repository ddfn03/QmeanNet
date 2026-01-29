import logging
import random
import time
import pandas as pd
from argparse import ArgumentParser

import wandb
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint
from scipy.conftest import devices
from wandb.integration.lightning.fabric import WandbLogger

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def train(model, datamodule, model_prefix, project, entity, offline, group, patience, k, max_epochs, min_delta,
          lr_log_interval, devices):
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
        logger=wandb_logger,
        devices=devices,
    )

    trainer.fit(model, datamodule=datamodule)
    wandb.finish()

    return trainer.logger.experiment.id


def main(args):
    if sum(args.split) != 100:
        raise ValueError("train_split , val_split  and test_split sum must be equal to 100")

    logger.info("="*80)
    logger.info(f"Starting Cross-Validation with {args.k} folds - Model: {args.model_prefix}")

    seed_everything(args.seed)
    random.seed(args.seed)

    logger.info(f"Random seed set to {args.seed}")
    logger.info("="*80)

    df = pd.read_csv(args.csv_path)




    logger.info(f"Fold distribution - Train: {args.split[0]}%, Val: {args.split[1]}%, Test: {args.split[2]}%")

    for k in range(args.k):
        logger.info("="*80)
        logger.info(f"[Fold {k +1}/{args.k}] Starting training and evaluation...")
        logger.info("="*80)

        unique_id = time.time()







if __name__ == "__main__":
    parser = ArgumentParser(description="Perform k-fold Cross-validation for QmeanNet")

    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--k", type=int, default=5, help="Number of folds")
    parser.add_argument("--split", nargs=3, type=int, default=(80, 10, 10),
                        help="Percentage split for train, val and test sets")

    # dataloader
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for dataloaders")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of workers for dataloaders")
    parser.add_argument("--max-sequence-len", type=int, default=512, help="Maximum sequence length for tokenizer")
    parser.add_argument("--csv_path",type=str, default="qmean_global_scores_clean-.csv" , help="Path to the csv file")