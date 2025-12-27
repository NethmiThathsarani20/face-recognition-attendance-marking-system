"""Face detection and recognition using InsightFace.
Simple implementation with default settings.
"""

import os
import pickle
import shutil
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from insightface.app import FaceAnalysis
import onnxruntime as ort

# Handle both relative and absolute imports
try:
    from .config import (
        DATABASE_DIR,
        DETECTION_SIZE,
        DETECTION_THRESHOLD,
        EMBEDDINGS_FILE,
        FACE_MODEL_NAME,
        SIMILARITY_THRESHOLD,
    )
except ImportError:
    from config import (
        DATABASE_DIR,
        DETECTION_SIZE,
        DETECTION_THRESHOLD,
        EMBEDDINGS_FILE,
        FACE_MODEL_NAME,
        SIMILARITY_THRESHOLD,
    )

# Performance optimization: Limit images per user for faster initial loading
# This balances accuracy (need multiple images per person) with speed
# 5 images provides good face representation while keeping load time reasonable
DEFAULT_MAX_IMAGES_PER_USER = 5


class FaceManager:
    """Simple face manager using InsightFace defaults.
    Handles face detection, recognition, and embedding storage.
    """

    def __init__(self):
        """Initialize face analysis app with default settings."""
        import sys
        print("🔧 Initializing FaceManager...", file=sys.stderr)
        
        # Configure ONNX Runtime providers to avoid CUDA warning on systems without CUDA
        print("⚙️  Configuring ONNX Runtime providers...", file=sys.stderr)
        available = ort.get_available_providers()
        providers = [p for p in ("CoreMLExecutionProvider", "CPUExecutionProvider") if p in available]
        if not providers:
            providers = ["CPUExecutionProvider"]
        print(f"✅ Using providers: {providers}", file=sys.stderr)
        
        print(f"📦 Loading InsightFace model: {FACE_MODEL_NAME}", file=sys.stderr)
        print("   (This may take 30-60 seconds on first run if models need to be downloaded)", file=sys.stderr)
        self.app = FaceAnalysis(name=FACE_MODEL_NAME, providers=providers)
        print("✅ InsightFace model loaded successfully", file=sys.stderr)
        
        print(f"🔧 Preparing face analysis app (det_size={DETECTION_SIZE}, det_thresh={DETECTION_THRESHOLD})...", file=sys.stderr)
        self.app.prepare(
            ctx_id=0, det_size=DETECTION_SIZE, det_thresh=DETECTION_THRESHOLD,
        )
        print("✅ Face analysis app prepared successfully", file=sys.stderr)
        
        print("📂 Loading face database from disk...", file=sys.stderr)
        self.face_database = self._load_face_database()
        print(f"✅ Face database loaded with {len(self.face_database)} entries", file=sys.stderr)
        print("✅ FaceManager initialization complete!", file=sys.stderr)

    def detect_faces(self, image: np.ndarray) -> List[Any]:
        """Detect faces in an image using InsightFace defaults.

        Args:
            image: Input image as numpy array

        Returns:
            List of face objects with embeddings
        """
        return self.app.get(image)

    def get_face_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Get face embedding from image using InsightFace.

        Args:
            image: Input image as numpy array

        Returns:
            Face embedding as numpy array or None if no face found
        """
        faces = self.detect_faces(image)
        if faces:
            # Return embedding of the first (most confident) face
            return faces[0].normed_embedding
        return None

    def add_user_images(self, user_name: str, image_paths: List[str]) -> bool:
        """Add a new user to the database with multiple images.

        Args:
            user_name: Name of the user
            image_paths: List of image file paths

        Returns:
            True if successfully added, False otherwise
        """
        embeddings = []

        for image_path in image_paths:
            if os.path.exists(image_path):
                image = cv2.imread(image_path)
                if image is not None:
                    embedding = self.get_face_embedding(image)
                    if embedding is not None:
                        embeddings.append(embedding)

        if embeddings:
            # Store average embedding for better recognition
            avg_embedding = np.mean(embeddings, axis=0)
            self.face_database[user_name] = avg_embedding
            self._save_face_database()
            return True

        return False

    def add_user_from_database_folder(self, user_name: str, max_images: int = DEFAULT_MAX_IMAGES_PER_USER) -> bool:
        """Add user from existing database folder structure.

        Args:
            user_name: Name of the user (folder name in database)
            max_images: Maximum number of images to process per user (default: 5 for fast loading)

        Returns:
            True if successfully added, False otherwise
        """
        import sys
        
        # Check if user is already loaded in memory
        if user_name in self.face_database:
            print(f"   User '{user_name}' already loaded, skipping...", file=sys.stderr)
            return True
        
        user_folder = os.path.join(DATABASE_DIR, user_name)
        if not os.path.exists(user_folder):
            return False

        # Get all image paths and limit to max_images for faster loading
        all_image_paths = []
        for filename in os.listdir(user_folder):
            if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
                all_image_paths.append(os.path.join(user_folder, filename))
        
        # Limit images per user for faster initial load
        # Select evenly distributed images if we have more than max_images
        if len(all_image_paths) > max_images:
            step = len(all_image_paths) // max_images
            image_paths = [all_image_paths[i * step] for i in range(max_images)]
        else:
            image_paths = all_image_paths

        return self.add_user_images(user_name, image_paths)

    def recognize_face(self, image: np.ndarray) -> Optional[Tuple[str, float]]:
        """Recognize a face in the image against the database.

        Args:
            image: Input image as numpy array

        Returns:
            Tuple of (user_name, similarity_score) or None if no match
        """
        query_embedding = self.get_face_embedding(image)
        if query_embedding is None:
            return None

        best_match = None
        best_score = 0

        for user_name, stored_embedding in self.face_database.items():
            # Calculate cosine similarity (InsightFace default)
            similarity = np.dot(query_embedding, stored_embedding)

            if similarity > best_score and similarity > SIMILARITY_THRESHOLD:
                best_score = similarity
                best_match = user_name
                # Save recognized face to user's folder
                if best_match:
                    self._save_recognized_face(image, best_match)

        return (best_match, float(best_score)) if best_match else None

    def load_all_database_users(self, max_images_per_user: int = DEFAULT_MAX_IMAGES_PER_USER) -> int:
        """Load all users from the database folder structure.

        Args:
            max_images_per_user: Maximum number of images to process per user (default: 5 for fast loading)

        Returns:
            Number of users successfully loaded
        """
        import sys
        if not os.path.exists(DATABASE_DIR):
            print(f"⚠️  Database directory does not exist: {DATABASE_DIR}", file=sys.stderr)
            return 0

        print(f"📂 Scanning database directory: {DATABASE_DIR}", file=sys.stderr)
        user_folders = [f for f in os.listdir(DATABASE_DIR) if os.path.isdir(os.path.join(DATABASE_DIR, f))]
        print(f"📊 Found {len(user_folders)} user folders", file=sys.stderr)
        print(f"⚡ Fast mode: Processing max {max_images_per_user} images per user", file=sys.stderr)
        
        loaded_count = 0
        for idx, user_folder in enumerate(user_folders, 1):
            user_path = os.path.join(DATABASE_DIR, user_folder)
            print(f"   [{idx}/{len(user_folders)}] Loading user: {user_folder}...", file=sys.stderr, end=' ')
            if self.add_user_from_database_folder(user_folder, max_images=max_images_per_user):
                loaded_count += 1
                print("✅", file=sys.stderr)
            else:
                print("❌ (no valid faces found)", file=sys.stderr)

        print(f"✅ Successfully loaded {loaded_count}/{len(user_folders)} users", file=sys.stderr)
        return loaded_count

    def get_user_list(self) -> List[str]:
        """Get list of all users in the database.

        Returns:
            List of user names
        """
        return list(self.face_database.keys())

    def delete_user(self, user_name: str) -> bool:
        """Delete a user from the face database and optionally their image folder.

        Args:
            user_name: Name of the user to delete

        Returns:
            True if user was deleted successfully, False otherwise
        """
        if user_name not in self.face_database:
            return False
        
        # Remove from in-memory database
        del self.face_database[user_name]
        
        # Save updated database
        self._save_face_database()
        
        # Optionally delete user's image folder from database directory
        user_dir = os.path.join(DATABASE_DIR, user_name)
        if os.path.exists(user_dir):
            try:
                shutil.rmtree(user_dir)
            except Exception as e:
                print(f"Warning: Could not delete user directory {user_dir}: {e}")
        
        return True

    def clear_embeddings(self) -> None:
        """Clear all face embeddings from memory and storage."""
        self.face_database.clear()
        if os.path.exists(EMBEDDINGS_FILE):
            os.remove(EMBEDDINGS_FILE)

    def _load_face_database(self) -> Dict[str, np.ndarray]:
        """Load face database from pickle file."""
        if os.path.exists(EMBEDDINGS_FILE):
            try:
                with open(EMBEDDINGS_FILE, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading face database: {e}")
        return {}

    def _save_face_database(self) -> None:
        """Save face database to pickle file."""
        try:
            with open(EMBEDDINGS_FILE, "wb") as f:
                pickle.dump(self.face_database, f)
        except Exception as e:
            print(f"Error saving face database: {e}")

    def _save_recognized_face(self, image: np.ndarray, user_name: str) -> None:
        """Save recognized face to user's database folder.

        Args:
            image: Input image as numpy array
            user_name: Name of the recognized user
        """
        user_dir = os.path.join(DATABASE_DIR, user_name)
        os.makedirs(user_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{user_name}.jpg"
        file_path = os.path.join(user_dir, filename)
        
        cv2.imwrite(file_path, image)

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better face detection.
        InsightFace handles most preprocessing internally.

        Args:
            image: Input image

        Returns:
            Preprocessed image
        """
        # InsightFace handles preprocessing internally, just ensure RGB format
        if len(image.shape) == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
