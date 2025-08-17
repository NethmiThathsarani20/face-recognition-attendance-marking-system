"""
CNN Training Module for Custom Face Recognition.

Provides functionality to train a lightweight CNN for specific face recognition tasks.
"""

import json
import os
import pickle
import random
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, List

import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

try:
    from tensorflow.keras import layers, models, optimizers, callbacks  # type: ignore
    from tensorflow.keras.utils import to_categorical  # type: ignore
    from tensorflow.keras import metrics as keras_metrics  # type: ignore
except ImportError:
    # Fallback for different TensorFlow versions
    from tf_keras import layers, models, optimizers, callbacks  # type: ignore
    from tf_keras.utils import to_categorical  # type: ignore
    from tf_keras import metrics as keras_metrics  # type: ignore

# Handle both relative and absolute imports
try:
    from .config import DATABASE_DIR
    from .exceptions import InsufficientDataError, ModelTrainingError, VideoProcessingError
    from .face_manager import FaceManager
except ImportError:
    from config import DATABASE_DIR
    from exceptions import InsufficientDataError, ModelTrainingError, VideoProcessingError
    from face_manager import FaceManager


# Constants
TARGET_SIZE = (112, 112)  # Standard face size
DEFAULT_CONFIDENCE_THRESHOLD = 0.7
DEFAULT_EPOCHS = 50
DEFAULT_VALIDATION_SPLIT = 0.2
DEFAULT_FRAME_INTERVAL = 30
PADDING = 20
# Use native Keras format to avoid HDF5 legacy warnings
MODEL_FILENAME = "custom_face_model.keras"
ENCODER_FILENAME = "label_encoder.pkl"
LOG_FILENAME = "training_log.json"


class CNNTrainer:
    """CNN Training class for custom face recognition model."""

    def __init__(self):
        """Initialize CNN trainer with InsightFace for data preparation (always fresh)."""
        # Reproducibility
        np.random.seed(42)
        random.seed(42)
        try:
            tf.random.set_seed(42)  # type: ignore
        except Exception:
            pass
        self.face_manager = FaceManager()
        self.model = None
        self.label_encoder = LabelEncoder()
        self.training_data = []
        self.training_labels = []
        self.unknown_counter = 1

        # Model paths
        self.models_dir = os.path.join(os.path.dirname(DATABASE_DIR), "cnn_models")
        os.makedirs(self.models_dir, exist_ok=True)
        self.model_path = os.path.join(self.models_dir, MODEL_FILENAME)
        self.encoder_path = os.path.join(self.models_dir, ENCODER_FILENAME)
        self.training_log_path = os.path.join(self.models_dir, LOG_FILENAME)

        # Training configuration
        self.target_size = TARGET_SIZE

    # Note: We intentionally do not load any existing model here to avoid confusion.

    def create_model(self, num_classes: int) -> models.Model:
        """
        Create a lightweight CNN model for face recognition.

        Args:
            num_classes: Number of face classes to recognize

        Returns:
            Compiled Keras model
        """
        # Data augmentation (applied only during training)
        data_augmentation = models.Sequential(
            [
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.08),
                layers.RandomZoom(0.08),
                layers.RandomContrast(0.2),
            ],
            name="augmentation",
        )

        l2 = tf.keras.regularizers.l2(1e-4)  # type: ignore

        model = models.Sequential(
            [
                layers.Input(shape=(112, 112, 3)),
                data_augmentation,
                # First block (separable conv to reduce params, improve generalization)
                layers.SeparableConv2D(32, (3, 3), padding="same", activation="relu", depthwise_regularizer=l2, pointwise_regularizer=l2),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.2),
                # Second block
                layers.SeparableConv2D(64, (3, 3), padding="same", activation="relu", depthwise_regularizer=l2, pointwise_regularizer=l2),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.3),
                # Third block
                layers.SeparableConv2D(128, (3, 3), padding="same", activation="relu", depthwise_regularizer=l2, pointwise_regularizer=l2),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.4),
                # Global pooling + dense
                layers.GlobalAveragePooling2D(),
                layers.Dense(256, activation="relu", kernel_regularizer=l2),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                layers.Dense(128, activation="relu", kernel_regularizer=l2),
                layers.Dropout(0.4),
                layers.Dense(num_classes, activation="softmax"),
            ]
        )

        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss="categorical_crossentropy",
            metrics=[
                keras_metrics.CategoricalAccuracy(name="categorical_accuracy"),
                keras_metrics.TopKCategoricalAccuracy(k=3, name="top3_accuracy"),
            ],
        )

        return model

    def prepare_training_data(self) -> bool:
        """
        Prepare training data using InsightFace for face detection and alignment.

        Returns:
            True if data preparation successful, False otherwise
        """
        self.training_data = []
        self.training_labels = []

        # Enumerate all users directly from database folder (ignore precomputed embeddings)
        if not os.path.exists(DATABASE_DIR):
            msg = f"Database directory not found at: {DATABASE_DIR}"
            print(f"❌ {msg}")
            raise InsufficientDataError(msg)

        users = [
            d
            for d in sorted(os.listdir(DATABASE_DIR))
            if os.path.isdir(os.path.join(DATABASE_DIR, d)) and not d.startswith(".")
        ]
        if not users:
            msg = "No users found in database"
            print(f"❌ {msg}")
            raise InsufficientDataError(msg)

        print(f"📊 Preparing training data for {len(users)} users...")

        for user_name in users:
            user_folder = os.path.join(DATABASE_DIR, user_name)
            if not os.path.exists(user_folder):
                continue

            user_faces = []
            for image_file in os.listdir(user_folder):
                if image_file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    image_path = os.path.join(user_folder, image_file)
                    face_image = self._extract_and_align_face(image_path)
                    if face_image is not None:
                        user_faces.append(face_image)

            if user_faces:
                self.training_data.extend(user_faces)
                self.training_labels.extend([user_name] * len(user_faces))
                print(f"   {user_name}: {len(user_faces)} face images")

        print(
            f"✅ Prepared {len(self.training_data)} training samples from {len(set(self.training_labels))} users"
        )
        
        if len(self.training_data) == 0:
            msg = "No training data could be extracted from user images"
            print(f"❌ {msg}")
            raise InsufficientDataError(msg)
            
        return True

    def _extract_and_align_face(self, image_path: str) -> Optional[np.ndarray]:
        """
        Extract and align face from image using InsightFace.

        Args:
            image_path: Path to the image file

        Returns:
            Aligned face image or None if no face found
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None

            # Use InsightFace to detect faces
            faces = self.face_manager.detect_faces(image)
            if not faces:
                return None

            # Get the first face
            face = faces[0]

            # Extract face region using bounding box
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox

            # Add padding
            padding = PADDING
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(image.shape[1], x2 + padding)
            y2 = min(image.shape[0], y2 + padding)

            # Extract face region
            face_region = image[y1:y2, x1:x2]

            # Resize to target size
            face_aligned = cv2.resize(face_region, self.target_size)

            # Normalize pixel values
            return face_aligned.astype(np.float32) / 255.0

        except Exception as e:
            print(f"⚠️ Error processing {image_path}: {e}")
            return None

    def train_model(
        self,
        epochs: int = DEFAULT_EPOCHS,
        validation_split: float = DEFAULT_VALIDATION_SPLIT,
        # Callback configuration (aligned with train.py CLI)
        monitor: str = "val_categorical_accuracy",
        early_stopping_patience: Optional[int] = 8,
        early_stopping_min_delta: float = 0.0,
        reduce_lr_patience: Optional[int] = 4,
        reduce_lr_factor: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Train the CNN model with prepared data.

        Args:
            epochs: Number of training epochs
            validation_split: Fraction of data for validation
            monitor: Metric name to monitor for callbacks (e.g., 'val_loss' or 'val_categorical_accuracy')
            early_stopping_patience: Patience for EarlyStopping; set None or <=0 to disable
            early_stopping_min_delta: Minimum change to qualify as improvement for EarlyStopping
            reduce_lr_patience: Patience for ReduceLROnPlateau; set None or <=0 to disable
            reduce_lr_factor: Factor by which the learning rate will be reduced

        Returns:
            Training results dictionary
        """
        # Always start fresh: remove any existing artifacts
        for _path in (self.model_path, self.encoder_path, self.training_log_path):
            try:
                if os.path.exists(_path):
                    os.remove(_path)
            except OSError:
                pass

        if not self.training_data:
            return {"success": False, "message": "No training data available"}

        print(f"🚀 Starting CNN training with {len(self.training_data)} samples...")

        # Convert to numpy arrays
        X = np.array(self.training_data)
        y = self.training_labels

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        y_categorical = to_categorical(y_encoded)

        # Create model
        num_classes = len(self.label_encoder.classes_)
        self.model = self.create_model(num_classes)

        print(f"📋 Model architecture: {num_classes} classes")
        print(f"   Input shape: {X.shape}")
        print(f"   Classes: {list(self.label_encoder.classes_)}")

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y_categorical,
            test_size=validation_split,
            random_state=42,
            stratify=y_encoded,
        )

        # Compute class weights to offset class imbalance
        classes: np.ndarray = np.unique(y_encoded)
        class_weights_vec = compute_class_weight(class_weight="balanced", classes=classes, y=y_encoded)
        class_weight = {int(c): float(w) for c, w in zip(classes, class_weights_vec)}

        # Training callbacks
        try:
            # Decide direction based on monitor string
            mode = "min" if ("loss" in monitor.lower()) else "max"
            callback_list = []  # type: List[Any]

            if early_stopping_patience and early_stopping_patience > 0:
                callback_list.append(
                    callbacks.EarlyStopping(
                        monitor=monitor,
                        patience=int(early_stopping_patience),
                        min_delta=float(early_stopping_min_delta),
                        restore_best_weights=True,
                        mode=mode,
                    )
                )

            if reduce_lr_patience and reduce_lr_patience > 0:
                callback_list.append(
                    callbacks.ReduceLROnPlateau(
                        monitor=monitor,
                        factor=float(reduce_lr_factor),
                        patience=int(reduce_lr_patience),
                        mode=mode,
                    )
                )

            # Always checkpoint the best model
            callback_list.append(
                callbacks.ModelCheckpoint(
                    filepath=self.model_path,
                    monitor=monitor,
                    save_best_only=True,
                    save_weights_only=False,
                    mode=mode,
                )
            )
        except (AttributeError, NameError):
            # Fallback for different TensorFlow/Keras variants
            callback_list = []

        # Train model
        start_time = datetime.now()
        history = self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=callback_list,
            class_weight=class_weight,
            verbose=1,  # type: ignore
        )

        training_time = (datetime.now() - start_time).total_seconds()

        # Evaluate model with named metrics (handle Keras/TF variations)
        # Newer Keras versions can return a dict; older versions return a list aligned with metrics_names.
        try:
            train_eval = self.model.evaluate(X_train, y_train, verbose=0, return_dict=True)  # type: ignore[arg-type]
            val_eval = self.model.evaluate(X_val, y_val, verbose=0, return_dict=True)  # type: ignore[arg-type]
        except TypeError:
            # return_dict not supported, fall back to list + metrics_names
            eval_names: List[str] = self.model.metrics_names  # type: ignore
            train_eval = self.model.evaluate(X_train, y_train, verbose=0)  # type: ignore
            val_eval = self.model.evaluate(X_val, y_val, verbose=0)  # type: ignore

            def _to_list(x: Any) -> List[float]:
                if isinstance(x, (list, tuple)):
                    return list(x)
                try:
                    import numpy as _np  # type: ignore
                    if isinstance(x, _np.ndarray):
                        return x.tolist()
                except Exception:
                    pass
                # Last resort: wrap scalar
                try:
                    return [float(x)]
                except Exception:
                    return [x]  # type: ignore

            def _eval_list_to_dict(names: List[str], values_any: Any) -> Dict[str, float]:
                values = _to_list(values_any)
                return {str(n): float(v) for n, v in zip(names, values)}

            train_metrics: Dict[str, float] = _eval_list_to_dict(eval_names, train_eval)
            val_metrics: Dict[str, float] = _eval_list_to_dict(eval_names, val_eval)
        else:
            # Dict path
            def _cast_dict(d: Dict[str, Any]) -> Dict[str, float]:
                out: Dict[str, float] = {}
                for k, v in d.items():
                    try:
                        out[str(k)] = float(v)  # type: ignore[arg-type]
                    except Exception:
                        # In case value is a 0-dim array or other numeric type
                        try:
                            import numpy as _np  # type: ignore
                            if isinstance(v, _np.ndarray):
                                out[str(k)] = float(v.item())
                                continue
                        except Exception:
                            pass
                        # Fallback to NaN
                        out[str(k)] = float("nan")
                return out

            train_metrics = _cast_dict(train_eval)  # type: ignore[arg-type]
            val_metrics = _cast_dict(val_eval)  # type: ignore[arg-type]

        # Ensure best model is saved even if checkpoint is unavailable
        try:
            if not os.path.exists(self.model_path):
                self.model.save(self.model_path)
        except Exception:
            self.model.save(self.model_path)
        with open(self.encoder_path, "wb") as f:
            pickle.dump(self.label_encoder, f)

        # Log training results
        training_log = {
            "timestamp": datetime.now().isoformat(),
            "num_classes": num_classes,
            "num_samples": len(X),
            "epochs": len(history.history["loss"]) if history and hasattr(history, "history") else epochs,
            "training_time_seconds": training_time,
            # Metrics
            "final_train_loss": float(train_metrics.get("loss", np.nan)),
            "final_val_loss": float(val_metrics.get("loss", np.nan)),
            "final_train_accuracy": float(train_metrics.get("categorical_accuracy", np.nan)),
            "final_val_accuracy": float(val_metrics.get("categorical_accuracy", np.nan)),
            "final_train_top3_accuracy": float(train_metrics.get("top3_accuracy", np.nan)),
            "final_val_top3_accuracy": float(val_metrics.get("top3_accuracy", np.nan)),
            "classes": list(self.label_encoder.classes_),
        }

        with open(self.training_log_path, "w") as f:
            json.dump(training_log, f, indent=2)

        print("✅ Training completed!")
        print(f"   Training accuracy: {training_log['final_train_accuracy']:.4f}")
        print(f"   Validation accuracy: {training_log['final_val_accuracy']:.4f}")
        if not np.isnan(training_log.get("final_val_top3_accuracy", float("nan"))):
            print(f"   Top-3 Val accuracy: {training_log['final_val_top3_accuracy']:.4f}")
        print(f"   Training time: {training_time:.2f} seconds")

        return {
            "success": True,
            "message": "Model trained successfully",
            "training_log": training_log,
            "history": {key: [float(val) for val in values] for key, values in history.history.items()} if history and hasattr(history, "history") else {},
        }

    def predict_face(
        self,
        image: np.ndarray,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> Optional[Tuple[str, float]]:
        """
        Predict face using trained CNN model.

        Args:
            image: Input image
            confidence_threshold: Minimum confidence for positive identification

        Returns:
            Tuple of (user_name, confidence) or None if no confident match
        """
        if self.model is None:
            return None

        # Extract and align face
        faces = self.face_manager.detect_faces(image)
        if not faces:
            return None

        # Get first face
        face = faces[0]
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox

        # Add padding
        padding = PADDING
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(image.shape[1], x2 + padding)
        y2 = min(image.shape[0], y2 + padding)

        # Extract and prepare face
        face_region = image[y1:y2, x1:x2]
        face_aligned = cv2.resize(face_region, self.target_size)
        face_normalized = face_aligned.astype(np.float32) / 255.0
        face_batch = np.expand_dims(face_normalized, axis=0)

        # Predict
        predictions = self.model.predict(face_batch, verbose=0)  # type: ignore
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])

        if confidence >= confidence_threshold:
            predicted_user = self.label_encoder.classes_[predicted_class_idx]
            return (predicted_user, confidence)

        return None

    def add_unknown_user(self, image: np.ndarray) -> str:
        """
        Add an unknown user to the training data for future training.

        Args:
            image: Image containing unknown face

        Returns:
            Generated user name for the unknown person
        """
        unknown_name = f"unknown_user_{self.unknown_counter:03d}"
        self.unknown_counter += 1

        # Create directory for unknown user
        unknown_dir = os.path.join(DATABASE_DIR, unknown_name)
        os.makedirs(unknown_dir, exist_ok=True)

        # Save the image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = os.path.join(unknown_dir, f"{timestamp}.jpg")
        cv2.imwrite(image_path, image)

        print(f"💾 Saved unknown user as: {unknown_name}")


        return unknown_name


    def get_training_status(self) -> Dict[str, Any]:
        """
        Get current training status and model information.

        Returns:
            Dictionary with training status information
        """
        status = {
            "model_exists": self.model is not None,
            "training_data_count": len(self.training_data),
            "unknown_counter": self.unknown_counter,
        }

        if self.model is not None:
            status["num_classes"] = len(self.label_encoder.classes_)
            status["classes"] = list(self.label_encoder.classes_)

        # Load training log if available
        if os.path.exists(self.training_log_path):
            try:
                with open(self.training_log_path, "r") as f:
                    status["last_training"] = json.load(f)
            except (OSError, json.JSONDecodeError):
                pass

        return status


    def add_training_data_from_video(
        self,
        video_path: str,
        user_name: str,
        frame_interval: int = DEFAULT_FRAME_INTERVAL,
    ) -> Dict[str, Any]:
        """
        Extract training data from a video file.

        Args:
            video_path: Path to the video file
            user_name: Name of the user in the video
            frame_interval: Extract every nth frame

        Returns:
            Result dictionary with extraction statistics
        """
        if not os.path.exists(video_path):
            return {"success": False, "message": "Video file not found"}

        # Create user directory
        user_dir = os.path.join(DATABASE_DIR, user_name)
        os.makedirs(user_dir, exist_ok=True)

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"success": False, "message": "Could not open video file"}

        frame_count = 0
        extracted_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    # Try to extract face
                    face_image = self._extract_and_align_face_from_frame(frame)
                    if face_image is not None:
                        # Save extracted face
                        filename = f"video_frame_{extracted_count:04d}.jpg"
                        save_path = os.path.join(user_dir, filename)
                        cv2.imwrite(save_path, (face_image * 255).astype(np.uint8))
                        extracted_count += 1

                frame_count += 1

            cap.release()

            return {
                "success": True,
                "message": f"Extracted {extracted_count} face images from video",
                "frames_processed": frame_count,
                "faces_extracted": extracted_count,
            }

        except Exception as e:
            cap.release()
            return {"success": False, "message": f"Error processing video: {e}"}

    def _extract_and_align_face_from_frame(
        self, frame: np.ndarray
    ) -> Optional[np.ndarray]:
        """Extract face from video frame."""
        try:
            faces = self.face_manager.detect_faces(frame)
            if not faces:
                return None

            # Get the first face
            face = faces[0]
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox

            # Add padding
            padding = PADDING
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(frame.shape[1], x2 + padding)
            y2 = min(frame.shape[0], y2 + padding)

            # Extract face region
            face_region = frame[y1:y2, x1:x2]
            face_aligned = cv2.resize(face_region, self.target_size)
            return face_aligned.astype(np.float32) / 255.0

        except Exception:
            return None
