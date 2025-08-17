"""
Embedding-based training using InsightFace features and a scikit-learn classifier.

Modern baseline: extract frozen face embeddings (InsightFace) and train a simple
linear classifier (Logistic Regression, multinomial) with class weighting.
This is robust for small datasets and often outperforms end-to-end CNNs trained
from scratch with limited data.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import joblib

# Local imports
try:
    from .config import DATABASE_DIR
    from .exceptions import InsufficientDataError
    from .face_manager import FaceManager
except ImportError:
    from config import DATABASE_DIR
    from exceptions import InsufficientDataError
    from face_manager import FaceManager


EMBED_MODEL_DIRNAME = "embedding_models"
MODEL_FILENAME = "embedding_classifier.joblib"
ENCODER_FILENAME = "label_encoder.pkl"
LOG_FILENAME = "training_log.json"


@dataclass
class EmbeddingTrainingConfig:
    validation_split: float = 0.2
    max_iter: int = 2000
    C: float = 1.0
    solver: str = "saga"  # supports multinomial + l1/l2
    penalty: str = "l2"
    n_jobs: int = -1
    random_state: int = 42


class EmbeddingTrainer:
    """Trainer that builds a classifier on top of InsightFace embeddings."""

    def __init__(self, models_root: Optional[str] = None, config: Optional[EmbeddingTrainingConfig] = None):
        self.face_manager = FaceManager()
        self.label_encoder = LabelEncoder()
        self.X: List[np.ndarray] = []  # embeddings
        self.y: List[str] = []  # labels
        self.config = config or EmbeddingTrainingConfig()

        # Save artifacts inside the repository root (next to src/, templates/, etc.)
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.models_dir = models_root or os.path.join(repo_root, EMBED_MODEL_DIRNAME)
        os.makedirs(self.models_dir, exist_ok=True)
        self.model_path = os.path.join(self.models_dir, MODEL_FILENAME)
        self.encoder_path = os.path.join(self.models_dir, ENCODER_FILENAME)
        self.training_log_path = os.path.join(self.models_dir, LOG_FILENAME)

        self.model: Optional[LogisticRegression] = None

    def load_if_available(self) -> bool:
        """Load saved model and encoder if present."""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.encoder_path):
                self.model = joblib.load(self.model_path)
                self.label_encoder = joblib.load(self.encoder_path)
                return True
        except Exception:
            pass
        return False

    def _iter_user_images(self, user_dir: str) -> List[str]:
        files: List[str] = []
        for name in os.listdir(user_dir):
            if name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                files.append(os.path.join(user_dir, name))
        return files

    def prepare_training_data(self) -> bool:
        """Extract embeddings for all images in DATABASE_DIR by user folders."""
        self.X = []
        self.y = []

        if not os.path.exists(DATABASE_DIR):
            raise InsufficientDataError("Database directory not found")

        users = [d for d in os.listdir(DATABASE_DIR) if os.path.isdir(os.path.join(DATABASE_DIR, d))]
        if not users:
            raise InsufficientDataError("No users found in database")

        print(f"📊 Preparing embeddings for {len(users)} users...")
        per_user_counts: Dict[str, int] = {}

        for user in sorted(users):
            user_dir = os.path.join(DATABASE_DIR, user)
            paths = self._iter_user_images(user_dir)
            count = 0
            for p in paths:
                img = cv2.imread(p)
                if img is None:
                    continue
                faces = self.face_manager.detect_faces(img)
                if not faces:
                    continue
                emb = faces[0].normed_embedding
                if isinstance(emb, np.ndarray) and emb.ndim == 1:
                    self.X.append(emb.astype(np.float32))
                    self.y.append(user)
                    count += 1
            if count:
                per_user_counts[user] = count
                print(f"   {user}: {count} embeddings")

        total = len(self.X)
        classes = len(set(self.y))
        print(f"✅ Prepared {total} embeddings from {classes} users")

        if total == 0:
            raise InsufficientDataError("No embeddings extracted from database images")

        return True

    def train(self) -> Dict[str, Any]:
        if not self.X:
            return {"success": False, "message": "No data. Call prepare_training_data() first."}

        X = np.stack(self.X)
        y = np.array(self.y)
        y_enc = self.label_encoder.fit_transform(y)

        # Split (stratified if possible)
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y_enc,
                test_size=self.config.validation_split,
                random_state=self.config.random_state,
                stratify=y_enc,
            )
            stratified = True
        except ValueError:
            # Fall back without stratify if class counts too small
            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y_enc,
                test_size=self.config.validation_split,
                random_state=self.config.random_state,
                stratify=None,
            )
            stratified = False

        # Class weights to mitigate imbalance
        classes = np.unique(y_enc)
        weights_vec = compute_class_weight(class_weight="balanced", classes=classes, y=y_enc)
        class_weight = {int(c): float(w) for c, w in zip(classes, weights_vec)}

        # Multinomial Logistic Regression on L2-normalized embeddings
        self.model = LogisticRegression(
            C=self.config.C,
            penalty=self.config.penalty,
            solver=self.config.solver,
            max_iter=self.config.max_iter,
            n_jobs=self.config.n_jobs,
            multi_class="multinomial",
            class_weight=class_weight,
            random_state=self.config.random_state,
        )

        print("🚀 Training embedding classifier (LogisticRegression)...")
        self.model.fit(X_train, y_train)

        # Evaluate
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)
        train_acc = float(accuracy_score(y_train, train_pred))
        val_acc = float(accuracy_score(y_val, val_pred))

        # Top-3 accuracy if probabilities available
        top3_val_acc = None
        if hasattr(self.model, "predict_proba"):
            val_proba = self.model.predict_proba(X_val)
            top3 = np.argsort(val_proba, axis=1)[:, -3:]
            top3_hits = sum(1 for i, t in enumerate(y_val) if t in top3[i])
            top3_val_acc = float(top3_hits / len(y_val)) if len(y_val) else 0.0

        # Persist artifacts
        joblib.dump(self.model, self.model_path)
        with open(self.encoder_path, "wb") as f:
            joblib.dump(self.label_encoder, f)

        training_log: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "num_classes": int(len(self.label_encoder.classes_)),
            "num_samples": int(len(X)),
            "validation_split": float(self.config.validation_split),
            "solver": self.config.solver,
            "penalty": self.config.penalty,
            "C": float(self.config.C),
            "max_iter": int(self.config.max_iter),
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "val_top3_accuracy": top3_val_acc if top3_val_acc is not None else None,
            "stratified_split": stratified,
            "classes": [str(c) for c in self.label_encoder.classes_],
        }
        with open(self.training_log_path, "w") as f:
            json.dump(training_log, f, indent=2)

        print("✅ Embedding classifier training completed!")
        print(f"   Training accuracy: {train_acc:.4f}")
        print(f"   Validation accuracy: {val_acc:.4f}")
        if top3_val_acc is not None:
            print(f"   Top-3 Val accuracy: {top3_val_acc:.4f}")

        return {"success": True, "training_log": training_log}

    def predict(self, image: np.ndarray, threshold: float = 0.0) -> Optional[Tuple[str, float]]:
        if self.model is None:
            return None

        faces = self.face_manager.detect_faces(image)
        if not faces:
            return None
        emb = faces[0].normed_embedding
        if not isinstance(emb, np.ndarray) or emb.ndim != 1:
            return None

        emb = emb.reshape(1, -1)
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(emb)[0]
            idx = int(np.argmax(proba))
            conf = float(proba[idx])
        else:
            pred = self.model.predict(emb)[0]
            idx = int(pred)
            conf = 1.0

        if conf >= threshold:
            label = self.label_encoder.classes_[idx]
            return str(label), conf
        return None
