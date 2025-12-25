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
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.utils.class_weight import compute_class_weight
import joblib

# Use non-interactive backend for CI / headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

    def _balance_dataset(self) -> None:
        """Balance the embedding dataset by oversampling minority classes."""
        if not self.X or not self.y:
            return

        labels = np.array(self.y)
        counts = Counter(labels)
        max_count = max(counts.values())

        balanced_X: List[np.ndarray] = []
        balanced_y: List[str] = []

        for cls, _ in counts.items():
            idx = np.where(labels == cls)[0]
            if not len(idx):
                continue
            if len(idx) < max_count:
                extra = np.random.choice(idx, size=max_count - len(idx), replace=True)
                all_idx = np.concatenate([idx, extra])
            else:
                all_idx = idx
            for i in all_idx:
                balanced_X.append(self.X[int(i)])
                balanced_y.append(cls)

        self.X = balanced_X
        self.y = balanced_y

        print(
            f"\U0001F4CA Balanced embedding dataset to {max_count} samples per class "
            f"across {len(counts)} classes (total {len(self.X)} samples).",
        )

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

        # Balance dataset to handle imbalanced databases
        self._balance_dataset()

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

        # ---------------- Evaluation curves and confusion matrices ----------------
        if hasattr(self.model, "predict_proba"):
            val_proba = self.model.predict_proba(X_val)
        else:
            # Fallback: use one-hot based on predictions
            num_classes = len(self.label_encoder.classes_)
            val_proba = np.zeros((len(y_val), num_classes), dtype=np.float32)
            for i, p in enumerate(val_pred):
                val_proba[i, int(p)] = 1.0

        num_classes = len(self.label_encoder.classes_)

        # Confusion matrices
        cm = confusion_matrix(y_val, val_pred, labels=list(range(num_classes)))
        with np.errstate(invalid="ignore", divide="ignore"):
            cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
            cm_norm = np.nan_to_num(cm_norm)

        class_names = list(self.label_encoder.classes_)

        def _plot_confusion(m, title: str, normalize: bool, filename: str) -> None:
            """Plot confusion matrix with improved readability for many classes."""
            n_classes = len(class_names)
            # Larger figures for better readability with many classes
            base_size = max(14, 0.25 * n_classes)
            fig, ax = plt.subplots(figsize=(base_size, base_size))

            im = ax.imshow(m, interpolation="nearest", cmap=plt.cm.Blues, aspect='auto')
            cbar = ax.figure.colorbar(im, ax=ax, pad=0.02)
            cbar.set_label('Count' if not normalize else 'Normalized Value', rotation=270, labelpad=20)
            
            # Set ticks for all classes
            ax.set_xticks(np.arange(n_classes))
            ax.set_yticks(np.arange(n_classes))
            ax.set_xticklabels(class_names, rotation=90, ha="right", fontsize=7)
            ax.set_yticklabels(class_names, fontsize=7)
            
            ax.set_ylabel("True label", fontsize=12, fontweight='bold')
            ax.set_xlabel("Predicted label", fontsize=12, fontweight='bold')
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

            # Add gridlines for better visibility
            ax.set_xticks(np.arange(n_classes) - 0.5, minor=True)
            ax.set_yticks(np.arange(n_classes) - 0.5, minor=True)
            ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

            # Add cell text values
            fmt = ".2f" if normalize else "d"
            thresh = m.max() / 2.0 if m.size else 0.0
            
            for i in range(m.shape[0]):
                for j in range(m.shape[1]):
                    value = m[i, j]
                    # Only show non-zero values for normalized matrices to reduce clutter
                    if normalize and value < 0.01:
                        continue
                    
                    text_color = "white" if value > thresh else "black"
                    fontsize = 6 if n_classes > 40 else 7 if n_classes > 25 else 8
                    
                    ax.text(
                        j, i,
                        format(value, fmt),
                        ha="center", va="center",
                        color=text_color,
                        fontsize=fontsize,
                        fontweight='bold' if value > thresh else 'normal'
                    )

            plt.tight_layout()
            fig.savefig(os.path.join(self.models_dir, filename), dpi=300, bbox_inches='tight')
            plt.close(fig)

        _plot_confusion(cm, "Embedding Confusion Matrix", False, "embedding_confusion_matrix.png")
        _plot_confusion(
            cm_norm,
            "Embedding Confusion Matrix (Normalized)",
            True,
            "embedding_confusion_matrix_normalized.png",
        )

        # Precision-recall curve (micro-averaged)
        y_true_bin = label_binarize(y_val, classes=list(range(num_classes)))
        precision_pr, recall_pr, _ = precision_recall_curve(
            y_true_bin.ravel(), val_proba.ravel(),
        )
        fig_pr, ax_pr = plt.subplots(figsize=(6, 5))
        ax_pr.plot(recall_pr, precision_pr, label="micro-average")
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.set_title("Embedding Precision-Recall Curve (micro-avg)")
        ax_pr.grid(True)
        ax_pr.legend(loc="best")
        fig_pr.savefig(
            os.path.join(self.models_dir, "embedding_precision_recall_curve.png"),
            dpi=150,
        )
        plt.close(fig_pr)

        # Precision / recall vs confidence threshold
        max_proba = np.max(val_proba, axis=1)
        thresholds = np.linspace(0.0, 1.0, 50)
        precisions_thr: List[float] = []
        recalls_thr: List[float] = []
        for thr in thresholds:
            mask = max_proba >= thr
            if not np.any(mask):
                precisions_thr.append(1.0)
                recalls_thr.append(0.0)
                continue
            y_sel_true = y_val[mask]
            y_sel_pred = val_pred[mask]
            correct = np.sum(y_sel_true == y_sel_pred)
            precisions_thr.append(float(correct) / float(len(y_sel_pred)))
            recalls_thr.append(float(correct) / float(len(y_val)))

        fig_thr, ax_thr = plt.subplots(figsize=(6, 5))
        ax_thr.plot(thresholds, precisions_thr, label="Precision")
        ax_thr.plot(thresholds, recalls_thr, label="Recall")
        ax_thr.set_xlabel("Confidence threshold")
        ax_thr.set_ylabel("Score")
        ax_thr.set_title("Embedding Precision / Recall vs Confidence Threshold")
        ax_thr.grid(True)
        ax_thr.legend(loc="best")
        fig_thr.savefig(
            os.path.join(self.models_dir, "embedding_precision_confidence_curve.png"),
            dpi=150,
        )
        plt.close(fig_thr)

        # Confidence histogram
        fig_hist, ax_hist = plt.subplots(figsize=(6, 4))
        ax_hist.hist(max_proba, bins=20, range=(0.0, 1.0), alpha=0.7, color="tab:blue")
        ax_hist.set_xlabel("Predicted max confidence")
        ax_hist.set_ylabel("Count")
        ax_hist.set_title("Embedding Confidence Distribution (validation)")
        fig_hist.savefig(
            os.path.join(self.models_dir, "embedding_confidence_curve.png"),
            dpi=150,
        )
        plt.close(fig_hist)

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
