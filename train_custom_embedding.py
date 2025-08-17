#!/usr/bin/env python3
"""
Entrypoint: Train custom embedding model (independent from InsightFace embeddings).
"""
from __future__ import annotations

import argparse

from src.custom_embedding_trainer import CustomEmbeddingTrainer, CustomEmbeddingConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--embedding-dim", type=int, default=128)
    args = parser.parse_args()

    cfg = CustomEmbeddingConfig(
        validation_split=args.validation_split,
        embedding_dim=args.embedding_dim,
        epochs=args.epochs,
    )

    trainer = CustomEmbeddingTrainer(config=cfg)
    trainer.prepare_training_data()
    result = trainer.train()
    if not result.get("success"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
