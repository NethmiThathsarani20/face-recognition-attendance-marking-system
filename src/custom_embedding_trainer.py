"""
Custom embedding-based face recognizer independent of InsightFace embeddings.

This trainer builds a small CNN to produce L2-normalized embeddings and a
classification head trained with cross-entropy. At inference, it uses cosine
similarity between embeddings and per-class centroids for recognition.

Face detection uses OpenCV Haar cascade by default (no InsightFace needed).
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
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight

# Use non-interactive backend for CI / headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers, callbacks  # type: ignore
    from tensorflow.keras import metrics as keras_metrics  # type: ignore
except Exception:
    # Fallback to tf-keras package name if needed
    import tf_keras as tf  # type: ignore
    from tf_keras import layers, models, optimizers, callbacks  # type: ignore
    from tf_keras import metrics as keras_metrics  # type: ignore

# Local imports
try:
    from .config import DATABASE_DIR
    from .exceptions import InsufficientDataError
except ImportError:
    from config import DATABASE_DIR
    from exceptions import InsufficientDataError


CUSTOM_EMBED_MODEL_DIRNAME = "custom_embedding_models"
MODEL_FILENAME = "custom_embedding_model.keras"
ENCODER_FILENAME = "label_encoder.pkl"
CENTROIDS_FILENAME = "class_centroids.npy"
LOG_FILENAME = "training_log.json"


@dataclass
class CustomEmbeddingConfig:
    validation_split: float = 0.2
    embedding_dim: int = 128
    image_size: Tuple[int, int] = (112, 112)
    batch_size: int = 32
    epochs: int = 30
    learning_rate: float = 1e-3
    random_state: int = 42
    similarity_threshold: float = 0.5  # cosine similarity threshold for recognition


class HaarFaceDetector:
    """Lightweight face detector using OpenCV Haar cascade (no InsightFace)."""

    def __init__(self):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(cascade_path)

    def detect_first_face(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        if image is None or image.size == 0:
            return None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        if len(faces) == 0:
            return None
        x, y, w, h = faces[0]
        return (int(x), int(y), int(x + w), int(y + h))


class CustomEmbeddingTrainer:
    """Trainer that learns its own embeddings and performs recognition via cosine similarity."""

    def __init__(self, models_root: Optional[str] = None, config: Optional[CustomEmbeddingConfig] = None):
        self.config = config or CustomEmbeddingConfig()
        self.detector = HaarFaceDetector()
        self.label_encoder = LabelEncoder()
        self.X: List[np.ndarray] = []
        self.y: List[str] = []

        # Artifacts directory
        # Save artifacts inside the repository root (next to src/, templates/, etc.)
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.models_dir = models_root or os.path.join(repo_root, CUSTOM_EMBED_MODEL_DIRNAME)
        os.makedirs(self.models_dir, exist_ok=True)
        self.model_path = os.path.join(self.models_dir, MODEL_FILENAME)
        self.encoder_path = os.path.join(self.models_dir, ENCODER_FILENAME)
        self.centroids_path = os.path.join(self.models_dir, CENTROIDS_FILENAME)
        self.training_log_path = os.path.join(self.models_dir, LOG_FILENAME)

        self.model: Optional[models.Model] = None
        self._embedding_model: Optional[models.Model] = None
        self.class_centroids: Optional[np.ndarray] = None

    # --------------------------- Data prep ---------------------------
    def _iter_user_images(self, user_dir: str) -> List[str]:
        files: List[str] = []
        for name in os.listdir(user_dir):
            if name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                files.append(os.path.join(user_dir, name))
        return files

    def _extract_face(self, image: np.ndarray, padding: int = 20) -> Optional[np.ndarray]:
        bbox = self.detector.detect_first_face(image)
        if bbox is None:
            return None
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(image.shape[1], x2 + padding)
        y2 = min(image.shape[0], y2 + padding)
        face = image[y1:y2, x1:x2]
        if face.size == 0:
            return None
        return face

    def _preprocess(self, face_img: np.ndarray) -> np.ndarray:
        h, w = self.config.image_size
        face_resized = cv2.resize(face_img, (w, h))
        face_resized = face_resized.astype(np.float32) / 255.0
        return face_resized

    def _balance_dataset(self) -> None:
        """Balance the dataset by oversampling minority classes based on labels."""
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
            f"\U0001F4CA Balanced custom-embedding dataset to {max_count} samples per class "
            f"across {len(counts)} classes (total {len(self.X)} samples).",
        )

    def prepare_training_data(self) -> bool:
        self.X = []
        self.y = []

        if not os.path.exists(DATABASE_DIR):
            raise InsufficientDataError("Database directory not found")

        users = [d for d in os.listdir(DATABASE_DIR) if os.path.isdir(os.path.join(DATABASE_DIR, d))]
        if not users:
            raise InsufficientDataError("No users found in database")

        # Use proper Unicode code point for 📊 to avoid surrogate pair issues on some terminals
        print(f"\U0001F4CA Preparing custom embeddings for {len(users)} users (no InsightFace embeddings)...")
        per_user_counts: Dict[str, int] = {}

        for user in sorted(users):
            user_dir = os.path.join(DATABASE_DIR, user)
            paths = self._iter_user_images(user_dir)
            count = 0
            for p in paths:
                img = cv2.imread(p)
                if img is None:
                    continue
                face = self._extract_face(img)
                if face is None:
                    continue
                tensor = self._preprocess(face)
                self.X.append(tensor)
                self.y.append(user)
                count += 1
            if count:
                per_user_counts[user] = count
                print(f"   {user}: {count} faces")

        total = len(self.X)
        classes = len(set(self.y))
        print(f"\u2705 Prepared {total} face crops from {classes} users")

        if total == 0:
            raise InsufficientDataError("No faces extracted from database images")

        # Balance dataset to handle imbalanced databases
        self._balance_dataset()

        return True

    # --------------------------- Model ---------------------------
    def _build_model(self, num_classes: int) -> Tuple[models.Model, models.Model]:
        """Returns (full_model, embedding_model)."""
        input_shape = (self.config.image_size[0], self.config.image_size[1], 3)

        l2 = tf.keras.regularizers.l2(1e-4)  # type: ignore
        inputs = layers.Input(shape=input_shape)

        x = layers.Rescaling(1.0, offset=0.0)(inputs)  # no-op; images already scaled
        # Simple CNN backbone
        x = layers.SeparableConv2D(32, 3, padding="same", activation="relu", depthwise_regularizer=l2, pointwise_regularizer=l2)(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
        x = layers.SeparableConv2D(64, 3, padding="same", activation="relu", depthwise_regularizer=l2, pointwise_regularizer=l2)(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
        x = layers.SeparableConv2D(128, 3, padding="same", activation="relu", depthwise_regularizer=l2, pointwise_regularizer=l2)(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation="relu", kernel_regularizer=l2)(x)
        x = layers.Dropout(0.4)(x)
        embed = layers.Dense(self.config.embedding_dim, activation=None, name="embedding", kernel_regularizer=l2)(x)
        # L2 normalize embeddings using a Keras layer (compatible with Keras 3)
        try:
            embed = layers.UnitNormalization(axis=-1, name="embedding_norm")(embed)
        except Exception:
            # Fallback for older TF/Keras combos
            embed = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=-1), name="embedding_norm")(embed)

        cls_outputs = layers.Dense(num_classes, activation="softmax", name="cls")(embed)

        full_model = models.Model(inputs=inputs, outputs=cls_outputs, name="custom_embed_model")
        embedding_model = models.Model(inputs=inputs, outputs=embed, name="custom_embed_encoder")

        full_model.compile(
            optimizer=optimizers.Adam(learning_rate=self.config.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=[keras_metrics.SparseCategoricalAccuracy(name="accuracy")],
        )

        return full_model, embedding_model

    # --------------------------- Train ---------------------------
    def train(self) -> Dict[str, Any]:
        if not self.X:
            return {"success": False, "message": "No data. Call prepare_training_data() first."}

        X = np.asarray(self.X, dtype=np.float32)
        y = np.asarray(self.y)
        y_enc = self.label_encoder.fit_transform(y)

        # Split train/val
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y_enc,
                test_size=self.config.validation_split,
                random_state=self.config.random_state,
                stratify=y_enc,
            )
        except ValueError:
            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y_enc,
                test_size=self.config.validation_split,
                random_state=self.config.random_state,
                stratify=None,
            )

        # Class weights
        classes = np.unique(y_enc)
        weights_vec = compute_class_weight(class_weight="balanced", classes=classes, y=y_enc)
        class_weight = {int(c): float(w) for c, w in zip(classes, weights_vec)}

        # Model
        num_classes = len(self.label_encoder.classes_)
        self.model, self._embedding_model = self._build_model(num_classes)

        cb: List[callbacks.Callback] = []
        try:
            cb = [
                callbacks.EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True, mode="max"),
                callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=3, mode="max"),
                callbacks.ModelCheckpoint(filepath=self.model_path, monitor="val_accuracy", save_best_only=True, mode="max"),
            ]
        except Exception:
            cb = []

        # Use proper Unicode code point for 🚀 to avoid surrogate pair issues on some terminals
        print("\U0001F680 Training custom embedding model (no InsightFace embeddings)...")
        history = self.model.fit(
            X_train,
            y_train,
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            validation_data=(X_val, y_val),
            class_weight=class_weight,
            callbacks=cb,
            verbose=1,  # type: ignore
        )

        # Save best model if checkpoint not present
        try:
            if not os.path.exists(self.model_path):
                self.model.save(self.model_path)
        except Exception:
            self.model.save(self.model_path)

        # Persist encoder
        import pickle

        with open(self.encoder_path, "wb") as f:
            pickle.dump(self.label_encoder, f)

        # Compute per-class centroids on train set
        self.class_centroids = self._compute_centroids(X_train, y_train)
        np.save(self.centroids_path, self.class_centroids)

        # Evaluate
        train_eval = self.model.evaluate(X_train, y_train, verbose=0)
        val_eval = self.model.evaluate(X_val, y_val, verbose=0)

        # ---------------- Evaluation and plots on validation set ----------------
        y_val_true = y_val
        y_val_proba = self.model.predict(X_val, verbose=0)
        y_val_pred = np.argmax(y_val_proba, axis=1)

        num_classes = len(self.label_encoder.classes_)
        cm = confusion_matrix(y_val_true, y_val_pred, labels=list(range(num_classes)))
        with np.errstate(invalid="ignore", divide="ignore"):
            cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
            cm_norm = np.nan_to_num(cm_norm)

        class_names = list(self.label_encoder.classes_)

        def _plot_confusion(m, title: str, normalize: bool, filename: str) -> None:
            """Plot confusion matrix with settings that stay readable for many classes.

            For normalized matrices with many classes we hide the per-cell text and rely
            on colors + colorbar so the image is not messy.
            """
            n_classes = len(class_names)
            # Make the figure larger when there are many classes.
            base_size = 8.0 if n_classes <= 20 else min(0.3 * n_classes, 20.0)
            fig, ax = plt.subplots(figsize=(base_size, base_size))

            im = ax.imshow(m, interpolation="nearest", cmap=plt.cm.Blues)
            ax.figure.colorbar(im, ax=ax)
            ax.set(xticks=np.arange(m.shape[1]), yticks=np.arange(m.shape[0]))
            ax.set_xticklabels(class_names, rotation=90, ha="center")
            ax.set_yticklabels(class_names)
            ax.set_ylabel("True label")
            ax.set_xlabel("Predicted label")
            ax.set_title(title)

            # Smaller tick labels for big matrices
            if n_classes <= 10:
                label_size = 10
            elif n_classes <= 25:
                label_size = 8
            else:
                label_size = 5
            ax.tick_params(axis="x", labelsize=label_size)
            ax.tick_params(axis="y", labelsize=label_size)

            # For large, normalized matrices we skip cell text to avoid clutter.
            add_text = not (normalize and n_classes > 25)

            if add_text:
                fmt = ".2f" if normalize else "d"
                thresh = m.max() / 2.0 if m.size else 0.0
                for i in range(m.shape[0]):
                    for j in range(m.shape[1]):
                        ax.text(
                            j,
                            i,
                            format(m[i, j], fmt),
                            ha="center",
                            va="center",
                            color="white" if m[i, j] > thresh else "black",
                            fontsize=6 if n_classes > 25 else 8,
                        )

            plt.tight_layout()
            fig.savefig(os.path.join(self.models_dir, filename), dpi=200)
            plt.close(fig)

        _plot_confusion(
            cm,
            "Custom-Embedding Confusion Matrix",
            False,
            "custom_embedding_confusion_matrix.png",
        )
        _plot_confusion(
            cm_norm,
            "Custom-Embedding Confusion Matrix (Normalized)",
            True,
            "custom_embedding_confusion_matrix_normalized.png",
        )

        # Precision-recall curve (micro-averaged)
        y_true_bin = label_binarize(y_val_true, classes=list(range(num_classes)))
        precision_pr, recall_pr, _ = precision_recall_curve(
            y_true_bin.ravel(), y_val_proba.ravel(),
        )
        fig_pr, ax_pr = plt.subplots(figsize=(6, 5))
        ax_pr.plot(recall_pr, precision_pr, label="micro-average")
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.set_title("Custom-Embedding Precision-Recall Curve (micro-avg)")
        ax_pr.grid(True)
        ax_pr.legend(loc="best")
        fig_pr.savefig(
            os.path.join(self.models_dir, "custom_embedding_precision_recall_curve.png"),
            dpi=150,
        )
        plt.close(fig_pr)

        # Precision / recall vs confidence threshold
        max_proba = np.max(y_val_proba, axis=1)
        thresholds = np.linspace(0.0, 1.0, 50)
        precisions_thr: List[float] = []
        recalls_thr: List[float] = []
        for thr in thresholds:
            mask = max_proba >= thr
            if not np.any(mask):
                precisions_thr.append(1.0)
                recalls_thr.append(0.0)
                continue
            y_sel_true = y_val_true[mask]
            y_sel_pred = y_val_pred[mask]
            correct = np.sum(y_sel_true == y_sel_pred)
            precisions_thr.append(float(correct) / float(len(y_sel_pred)))
            recalls_thr.append(float(correct) / float(len(y_val_true)))

        fig_thr, ax_thr = plt.subplots(figsize=(6, 5))
        ax_thr.plot(thresholds, precisions_thr, label="Precision")
        ax_thr.plot(thresholds, recalls_thr, label="Recall")
        ax_thr.set_xlabel("Confidence threshold")
        ax_thr.set_ylabel("Score")
        ax_thr.set_title("Custom-Embedding Precision / Recall vs Confidence Threshold")
        ax_thr.grid(True)
        ax_thr.legend(loc="best")
        fig_thr.savefig(
            os.path.join(self.models_dir, "custom_embedding_precision_confidence_curve.png"),
            dpi=150,
        )
        plt.close(fig_thr)

        # Confidence histogram
        fig_hist, ax_hist = plt.subplots(figsize=(6, 4))
        ax_hist.hist(max_proba, bins=20, range=(0.0, 1.0), alpha=0.7, color="tab:blue")
        ax_hist.set_xlabel("Predicted max confidence")
        ax_hist.set_ylabel("Count")
        ax_hist.set_title("Custom-Embedding Confidence Distribution (validation)")
        fig_hist.savefig(
            os.path.join(self.models_dir, "custom_embedding_confidence_curve.png"),
            dpi=150,
        )
        plt.close(fig_hist)

        # Log
        training_log: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "num_classes": int(num_classes),
            "num_samples": int(len(X)),
            "validation_split": float(self.config.validation_split),
            "embedding_dim": int(self.config.embedding_dim),
            "epochs": int(len(history.history.get("loss", [])) or self.config.epochs),
            "train_accuracy": float(train_eval[1]) if isinstance(train_eval, (list, tuple)) and len(train_eval) > 1 else None,
            "val_accuracy": float(val_eval[1]) if isinstance(val_eval, (list, tuple)) and len(val_eval) > 1 else None,
            "classes": [str(c) for c in self.label_encoder.classes_],
        }
        with open(self.training_log_path, "w") as f:
            json.dump(training_log, f, indent=2)

        print("\u2705 Custom embedding training completed!")
        if training_log["train_accuracy"] is not None:
            print(f"   Training accuracy: {training_log['train_accuracy']:.4f}")
        if training_log["val_accuracy"] is not None:
            print(f"   Validation accuracy: {training_log['val_accuracy']:.4f}")

        return {"success": True, "training_log": training_log}

    # --------------------------- Inference ---------------------------
    def load_if_available(self) -> bool:
        """Load saved model, encoder, and centroids if present."""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.encoder_path) and os.path.exists(self.centroids_path):
                self.model = models.load_model(self.model_path)
                # Recreate embedding submodel: prefer normalized output if available
                try:
                    norm_layer = self.model.get_layer("embedding_norm")
                    emb_out = norm_layer.output
                except Exception:
                    # Older models may not have a named normalization layer. Normalize on the fly.
                    base_emb = self.model.get_layer("embedding").output
                    try:
                        emb_out = layers.UnitNormalization(axis=-1, name="embedding_norm_runtime")(base_emb)
                    except Exception:
                        emb_out = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=-1), name="embedding_norm_runtime")(base_emb)
                self._embedding_model = models.Model(inputs=self.model.input, outputs=emb_out)
                import pickle

                with open(self.encoder_path, "rb") as f:
                    self.label_encoder = pickle.load(f)
                self.class_centroids = np.load(self.centroids_path)
                return True
        except Exception as e:
            print(f"Failed to load custom embedding model: {e}")
        return False

    def _compute_centroids(self, X: np.ndarray, y_enc: np.ndarray) -> np.ndarray:
        if self._embedding_model is None:
            raise RuntimeError("Embedding model not available")
        embeds = self._embedding_model.predict(X, verbose=0)
        num_classes = len(self.label_encoder.classes_)
        centroids = np.zeros((num_classes, embeds.shape[1]), dtype=np.float32)
        for c in range(num_classes):
            cls_embeds = embeds[y_enc == c]
            if len(cls_embeds) == 0:
                continue
            centroids[c] = np.mean(cls_embeds, axis=0)
            # L2 normalize centroid
            norm = np.linalg.norm(centroids[c]) + 1e-8
            centroids[c] = centroids[c] / norm
        return centroids

    def predict(self, image: np.ndarray, threshold: Optional[float] = None) -> Optional[Tuple[str, float]]:
        if self._embedding_model is None or self.class_centroids is None:
            return None
        if image is None or image.size == 0:
            return None
        face = self._extract_face(image)
        if face is None:
            return None
        x = self._preprocess(face)
        x = np.expand_dims(x, axis=0)
        emb = self._embedding_model.predict(x, verbose=0)[0]
        # Cosine similarity against centroids (embeddings are L2 normalized)
        sims = np.dot(self.class_centroids, emb)
        idx = int(np.argmax(sims))
        score = float(sims[idx])
        use_thr = self.config.similarity_threshold if threshold is None else float(threshold)
        if score >= use_thr:
            label = self.label_encoder.classes_[idx]
            return str(label), score
        return None
