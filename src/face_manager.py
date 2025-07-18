"""
Face detection and recognition using InsightFace.
Simple implementation with default settings.
"""

import os
import cv2
import numpy as np
import pickle
from typing import List, Tuple, Optional, Dict, Any
import insightface
from insightface.app import FaceAnalysis

# Handle both relative and absolute imports
try:
    from .config import (
        FACE_MODEL_NAME, DETECTION_SIZE, DETECTION_THRESHOLD, 
        SIMILARITY_THRESHOLD, EMBEDDINGS_FILE, DATABASE_DIR
    )
except ImportError:
    from config import (
        FACE_MODEL_NAME, DETECTION_SIZE, DETECTION_THRESHOLD, 
        SIMILARITY_THRESHOLD, EMBEDDINGS_FILE, DATABASE_DIR
    )


class FaceManager:
    """
    Simple face manager using InsightFace defaults.
    Handles face detection, recognition, and embedding storage.
    """
    
    def __init__(self):
        """Initialize face analysis app with default settings."""
        self.app = FaceAnalysis(name=FACE_MODEL_NAME)
        self.app.prepare(ctx_id=0, det_size=DETECTION_SIZE, det_thresh=DETECTION_THRESHOLD)
        self.face_database = self._load_face_database()
    
    def detect_faces(self, image: np.ndarray) -> List[Any]:
        """
        Detect faces in an image using InsightFace defaults.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of face objects with embeddings
        """
        return self.app.get(image)
    
    def get_face_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Get face embedding from image using InsightFace.
        
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
        """
        Add a new user to the database with multiple images.
        
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
    
    def add_user_from_database_folder(self, user_name: str) -> bool:
        """
        Add user from existing database folder structure.
        
        Args:
            user_name: Name of the user (folder name in database)
            
        Returns:
            True if successfully added, False otherwise
        """
        user_folder = os.path.join(DATABASE_DIR, user_name)
        if not os.path.exists(user_folder):
            return False
        
        image_paths = []
        for filename in os.listdir(user_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_paths.append(os.path.join(user_folder, filename))
        
        return self.add_user_images(user_name, image_paths)
    
    def recognize_face(self, image: np.ndarray) -> Optional[Tuple[str, float]]:
        """
        Recognize a face in the image against the database.
        
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
        
        return (best_match, float(best_score)) if best_match else None
    
    def load_all_database_users(self) -> int:
        """
        Load all users from the database folder structure.
        
        Returns:
            Number of users successfully loaded
        """
        if not os.path.exists(DATABASE_DIR):
            return 0
        
        loaded_count = 0
        for user_folder in os.listdir(DATABASE_DIR):
            user_path = os.path.join(DATABASE_DIR, user_folder)
            if os.path.isdir(user_path):
                if self.add_user_from_database_folder(user_folder):
                    loaded_count += 1
        
        return loaded_count
    
    def get_user_list(self) -> List[str]:
        """
        Get list of all users in the database.
        
        Returns:
            List of user names
        """
        return list(self.face_database.keys())
    
    def _load_face_database(self) -> Dict[str, np.ndarray]:
        """Load face database from pickle file."""
        if os.path.exists(EMBEDDINGS_FILE):
            try:
                with open(EMBEDDINGS_FILE, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading face database: {e}")
        return {}
    
    def _save_face_database(self) -> None:
        """Save face database to pickle file."""
        try:
            with open(EMBEDDINGS_FILE, 'wb') as f:
                pickle.dump(self.face_database, f)
        except Exception as e:
            print(f"Error saving face database: {e}")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better face detection.
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
