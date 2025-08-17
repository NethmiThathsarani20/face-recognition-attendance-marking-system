#!/usr/bin/env python3
"""
Entrypoint: Train embedding-based classifier on current database.
"""
from __future__ import annotations

import argparse

from src.embedding_trainer import EmbeddingTrainer, EmbeddingTrainingConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--solver", type=str, default="saga")
    parser.add_argument("--penalty", type=str, default="l2")
    args = parser.parse_args()

    cfg = EmbeddingTrainingConfig(
        validation_split=args.validation_split,
        max_iter=args.max_iter,
        C=args.C,
        solver=args.solver,
        penalty=args.penalty,
    )

    trainer = EmbeddingTrainer(config=cfg)
    trainer.prepare_training_data()
    result = trainer.train()
    if not result.get("success"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
