"""
Minimal entrypoint to train the custom CNN model in CI.
- Prepares training data using InsightFace via CNNTrainer
- Trains the model
- Overwrites existing artifacts in cnn_models/
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

# Make sure we can import from src/
HERE = os.path.dirname(os.path.abspath(__file__))
SYS_PATHS = [HERE, os.path.join(HERE, "src")]
for p in SYS_PATHS:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.cnn_trainer import CNNTrainer  # type: ignore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--validation-split", type=float, default=0.2)
    args = parser.parse_args()

    # Remove all existing model artifacts for a fresh train (as requested)
    models_dir = os.path.join(HERE, "cnn_models")
    try:
        shutil.rmtree(models_dir)
    except FileNotFoundError:
        pass
    except OSError:
        # Best-effort cleanup; continue
        pass
    os.makedirs(models_dir, exist_ok=True)

    # Do not load any existing model/state
    trainer = CNNTrainer()

    # Always (re)prepare data and retrain
    trainer.prepare_training_data()
    result = trainer.train_model(epochs=args.epochs, validation_split=args.validation_split)

    if not result.get("success"):
        print(f"Training failed: {result}")
        sys.exit(1)
    else:
        print("Training completed successfully.")


if __name__ == "__main__":
    main()
