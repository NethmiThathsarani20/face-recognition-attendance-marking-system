"""
CNN Training Module for Custom Face Recognition.

Provides functionality to train a lightweight CNN for specific face recognition tasks.
"""

import json
import os
import pickle
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

try:
    from tensorflow.keras import layers, models, optimizers, callbacks  # type: ignore
    from tensorflow.keras.utils import to_categorical  # type: ignore
except ImportError:
    # Fallback for different TensorFlow versions
    from tf_keras import layers, models, optimizers, callbacks  # type: ignore
    from tf_keras.utils import to_categorical  # type: ignore

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
MODEL_FILENAME = "custom_face_model.h5"
ENCODER_FILENAME = "label_encoder.pkl"
LOG_FILENAME = "training_log.json"


class CNNTrainer:
    """CNN Training class for custom face recognition model."""

    def __init__(self):
        """Initialize CNN trainer with InsightFace for data preparation."""
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
        self.auto_training_enabled = True

        self._load_existing_model()

    def _load_existing_model(self):
        """Load existing trained model if available."""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.encoder_path):
                self.model = tf.keras.models.load_model(self.model_path)  # type: ignore
                with open(self.encoder_path, "rb") as f:
                    self.label_encoder = pickle.load(f)
                print(
                    f"✅ Loaded existing CNN model with {len(self.label_encoder.classes_)} classes"
                )
            else:
                print("📝 No existing CNN model found")
        except Exception as e:
            print(f"⚠️ Error loading existing model: {e}")

    def create_model(self, num_classes: int) -> models.Model:
        """
        Create a lightweight CNN model for face recognition.

        Args:
            num_classes: Number of face classes to recognize

        Returns:
            Compiled Keras model
        """
        model = models.Sequential(
            [
                # Input layer
                layers.Input(shape=(112, 112, 3)),
                # First convolutional block
                layers.Conv2D(32, (3, 3), activation="relu"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                # Second convolutional block
                layers.Conv2D(64, (3, 3), activation="relu"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                # Third convolutional block
                layers.Conv2D(128, (3, 3), activation="relu"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                # Flatten and dense layers
                layers.Flatten(),
                layers.Dense(256, activation="relu"),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                layers.Dense(128, activation="relu"),
                layers.Dropout(0.5),
                # Output layer
                layers.Dense(num_classes, activation="softmax"),
            ]
        )

        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
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

        # Get all users from database
        users = self.face_manager.get_user_list()
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
    ) -> Dict[str, Any]:
        """
        Train the CNN model with prepared data.

        Args:
            epochs: Number of training epochs
            validation_split: Fraction of data for validation

        Returns:
            Training results dictionary
        """
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

        # Training callbacks
        try:
            callback_list = [
                callbacks.EarlyStopping(
                    patience=10, restore_best_weights=True
                ),
                callbacks.ReduceLROnPlateau(
                    factor=0.5, patience=5
                ),
            ]
        except (AttributeError, NameError):
            # Fallback for different TensorFlow versions
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
            verbose=1,  # type: ignore
        )

        training_time = (datetime.now() - start_time).total_seconds()

        # Evaluate model
        train_loss, train_acc = self.model.evaluate(X_train, y_train, verbose=0)  # type: ignore
        val_loss, val_acc = self.model.evaluate(X_val, y_val, verbose=0)  # type: ignore

        # Save model and encoder
        self.model.save(self.model_path)
        with open(self.encoder_path, "wb") as f:
            pickle.dump(self.label_encoder, f)

        # Log training results
        training_log = {
            "timestamp": datetime.now().isoformat(),
            "num_classes": num_classes,
            "num_samples": len(X),
            "epochs": len(history.history["loss"])
            if history and hasattr(history, "history")
            else epochs,
            "training_time_seconds": training_time,
            "final_train_accuracy": float(train_acc),
            "final_val_accuracy": float(val_acc),
            "final_train_loss": float(train_loss),
            "final_val_loss": float(val_loss),
            "classes": list(self.label_encoder.classes_),
        }

        with open(self.training_log_path, "w") as f:
            json.dump(training_log, f, indent=2)

        print("✅ Training completed!")
        print(f"   Training accuracy: {train_acc:.4f}")
        print(f"   Validation accuracy: {val_acc:.4f}")
        print(f"   Training time: {training_time:.2f} seconds")

        return {
            "success": True,
            "message": "Model trained successfully",
            "training_log": training_log,
            "history": {
                key: [float(val) for val in values]
                for key, values in history.history.items()
            }
            if history and hasattr(history, "history")
            else {},
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

        # If auto-training is enabled, trigger retraining
        if self.auto_training_enabled:
            self._trigger_auto_training()

        return unknown_name

    def _trigger_auto_training(self):
        """Trigger automatic model retraining when new data is added."""
        print("🔄 Auto-training triggered...")
        try:
            if self.prepare_training_data():
                result = self.train_model(
                    epochs=20
                )  # Shorter training for auto-updates
                if result["success"]:
                    print("✅ Auto-training completed successfully")
                else:
                    print(f"❌ Auto-training failed: {result['message']}")
        except Exception as e:
            print(f"❌ Auto-training error: {e}")

    def get_training_status(self) -> Dict[str, Any]:
        """
        Get current training status and model information.

        Returns:
            Dictionary with training status information
        """
        status = {
            "model_exists": self.model is not None,
            "auto_training_enabled": self.auto_training_enabled,
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

    def toggle_auto_training(self, enabled: bool):
        """
        Enable or disable automatic training.

        Args:
            enabled: Whether to enable auto-training
        """
        self.auto_training_enabled = enabled
        print(f"🔧 Auto-training {'enabled' if enabled else 'disabled'}")

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
