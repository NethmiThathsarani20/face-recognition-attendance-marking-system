"""
Minimal entrypoint to train the custom CNN model in CI.
- Prepares training data using InsightFace via CNNTrainer
- Trains the model
- Overwrites existing artifacts in cnn_models/
"""
from __future__ import annotations

import argparse
import os
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

    trainer = CNNTrainer()

    # Remove existing model artifacts for a fresh train (as requested)
    models_dir = os.path.join(HERE, "cnn_models")
    to_remove = [
        os.path.join(models_dir, "custom_face_model.h5"),
        os.path.join(models_dir, "label_encoder.pkl"),
        os.path.join(models_dir, "training_log.json"),
    ]
    for path in to_remove:
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass

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
