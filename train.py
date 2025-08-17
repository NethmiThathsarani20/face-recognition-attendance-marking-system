#!/usr/bin/env python3
"""
Unified training script: runs both CNN training and the embedding-based classifier.

Usage examples:
  python train.py --epochs 30 --validation-split 0.2
  python train.py --only embedding
    python train.py --only cnn --early-stopping-patience 0
    python train.py --only custom-embedding --embedding-dim 128
"""
from __future__ import annotations

import argparse
import sys

from src.cnn_trainer import CNNTrainer
from src.embedding_trainer import EmbeddingTrainer, EmbeddingTrainingConfig
from src.custom_embedding_trainer import CustomEmbeddingTrainer, CustomEmbeddingConfig


ess = """
Training modes:
- cnn: end-to-end lightweight CNN classifier
- embedding: InsightFace embedding + multinomial logistic regression
- custom-embedding: independent embedding model (no InsightFace embeddings)
"""


def train_cnn(args) -> bool:
    trainer = CNNTrainer()
    trainer.prepare_training_data()

    es_patience = args.early_stopping_patience if args.early_stopping_patience and args.early_stopping_patience > 0 else None
    rlrop_patience = args.reduce_lr_patience if args.reduce_lr_patience and args.reduce_lr_patience > 0 else None

    result = trainer.train_model(
        epochs=args.epochs,
        validation_split=args.validation_split,
        monitor=args.monitor,
        early_stopping_patience=es_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        reduce_lr_patience=rlrop_patience,
        reduce_lr_factor=args.reduce_lr_factor,
    )
    return bool(result.get("success"))


def train_embedding(args) -> bool:
    cfg = EmbeddingTrainingConfig(
        validation_split=args.validation_split,
    )
    et = EmbeddingTrainer(config=cfg)
    et.prepare_training_data()
    result = et.train()
    return bool(result.get("success"))


def train_custom_embedding(args) -> bool:
    cfg = CustomEmbeddingConfig(
        validation_split=args.validation_split,
        epochs=args.epochs,
    embedding_dim=args.embedding_dim,
    )
    cet = CustomEmbeddingTrainer(config=cfg)
    cet.prepare_training_data()
    result = cet.train()
    return bool(result.get("success"))


def main():
    parser = argparse.ArgumentParser(description=ess)
    parser.add_argument("--only", choices=["cnn", "embedding", "custom-embedding"], help="Train only a specific method")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--embedding-dim", type=int, default=128, help="Embedding dimension for custom-embedding training")
    # CNN callback controls
    parser.add_argument("--monitor", type=str, default="val_categorical_accuracy")
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--reduce-lr-patience", type=int, default=4)
    parser.add_argument("--reduce-lr-factor", type=float, default=0.5)
    args = parser.parse_args()

    ok = True
    if args.only in (None, "cnn"):
        ok = train_cnn(args) and ok
    if args.only in (None, "embedding"):
        ok = train_embedding(args) and ok
    if args.only in (None, "custom-embedding"):
        ok = train_custom_embedding(args) and ok

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
