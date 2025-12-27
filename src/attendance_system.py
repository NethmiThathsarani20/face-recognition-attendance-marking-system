"""Main attendance system logic.
Handles attendance marking for both camera and upload inputs.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

# Suppress TensorFlow logging BEFORE importing models
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

import cv2
import numpy as np

# Handle both relative and absolute imports
try:
    from .config import (
        ATTENDANCE_DATE_FORMAT,
        ATTENDANCE_DIR,
        ATTENDANCE_TIME_FORMAT,
        CNN_CONFIDENCE_THRESHOLD,
        BASE_DIR,
        CNN_MODELS_DIR,
        DATABASE_DIR,
        SIMILARITY_THRESHOLD,
        USE_CNN_MODEL,
    )
    from .face_manager import FaceManager
except ImportError:
    from config import (
        ATTENDANCE_DATE_FORMAT,
        ATTENDANCE_DIR,
        ATTENDANCE_TIME_FORMAT,
        CNN_CONFIDENCE_THRESHOLD,
        BASE_DIR,
        CNN_MODELS_DIR,
        DATABASE_DIR,
        SIMILARITY_THRESHOLD,
        USE_CNN_MODEL,
    )
    from face_manager import FaceManager

# Trainers will be imported lazily
CNNTrainer = None
EmbeddingTrainer = None
CustomEmbeddingTrainer = None


def _import_trainers():
    """Lazy import for trainers to avoid TensorFlow initialization."""
    global CNNTrainer, EmbeddingTrainer, CustomEmbeddingTrainer
    if CNNTrainer is None:
        try:
            from .cnn_trainer import CNNTrainer as CNN
            from .embedding_trainer import EmbeddingTrainer as Embedding
            from .custom_embedding_trainer import CustomEmbeddingTrainer as CustomEmbedding
            CNNTrainer = CNN
            EmbeddingTrainer = Embedding
            CustomEmbeddingTrainer = CustomEmbedding
        except ImportError:
            from cnn_trainer import CNNTrainer as CNN
            from embedding_trainer import EmbeddingTrainer as Embedding
            from custom_embedding_trainer import CustomEmbeddingTrainer as CustomEmbedding
            CNNTrainer = CNN
            EmbeddingTrainer = Embedding
            CustomEmbeddingTrainer = CustomEmbedding


class AttendanceSystem:
    """Simple attendance system with unified processing for camera and upload.
    """

    def __init__(self):
        """Initialize attendance system with face manager and model backends."""
        self._face_manager: Optional[FaceManager] = None  # Lazy initialization
        # Model selection flags (default to InsightFace)
        self.use_cnn_model: bool = USE_CNN_MODEL
        self.use_embedding_model: bool = False
        self.use_custom_embedding_model: bool = False

        # Trainers (lazy init on switch)
        self.cnn_trainer: Optional[object] = None
        self.embedding_trainer: Optional[object] = None
        self.custom_embedding_trainer: Optional[object] = None

        # Availability flags (set when models are loaded)
        self.embedding_model_available: bool = False
        self.custom_embedding_model_available: bool = False
        
        self._last_captured_image = None
        self._users_loaded: bool = False
        # Models will be loaded on first use (lazy loading)

    @property
    def face_manager(self) -> FaceManager:
        """Lazy load face manager on first access."""
        if self._face_manager is None:
            import sys
            print("🔄 Initializing face recognition models... (this may take 10-15 seconds on first load)", file=sys.stderr)
            self._face_manager = FaceManager()
            print("✅ Face recognition models loaded successfully", file=sys.stderr)
        return self._face_manager

    def _load_existing_users(self) -> None:
        """Load all existing users from database folder."""
        import sys
        if self._users_loaded:
            return
        
        # Check if we already have embeddings loaded from the pickle file
        existing_users = len(self.face_manager.face_database)
        print(f"📊 Found {existing_users} users already in memory from embeddings file", file=sys.stderr)
        
        # Only reload from database if no embeddings are loaded
        if existing_users == 0:
            print("📂 No cached embeddings found, loading users from database folder...", file=sys.stderr)
            loaded_count = self.face_manager.load_all_database_users()
            print(f"✅ Loaded {loaded_count} users from database", file=sys.stderr)
        else:
            print(f"✅ Using {existing_users} cached user embeddings (fast load!)", file=sys.stderr)
        
        self._users_loaded = True

    def mark_attendance(
        self, input_source: Union[np.ndarray, str, int],
        save_captured: bool = True
    ) -> Dict[str, Any]:
        """Unified attendance marking function for camera, upload, or image array.

        Args:
            input_source: Can be:
                - np.ndarray: Image array
                - str: Path to image file
                - int: Camera index
            save_captured: Whether to save the captured image for recognized faces

        Returns:
            Dictionary with attendance result and metadata
        """
        # Lazy load users on first request
        self._load_existing_users()
        
        print(f"📋 Starting attendance marking process with input type: {type(input_source)}")
        
        # Get image from input source
        image = self._get_image_from_source(input_source)
        if image is None:
            print("❌ Failed to get valid image from input source")
            return self._create_result(False, "Failed to get image from source")
        
        print(f"✅ Successfully obtained image of shape: {image.shape}")

        # Optionally store last captured image if requested
        self._last_captured_image = image if save_captured else None
        
        recognition_result = None

        # Use selected model (priority: CNN > CustomEmbedding > Embedding > InsightFace similarity)
        if self.use_cnn_model and self.cnn_trainer is not None and self.cnn_trainer.model is not None:
            print("🧠 Using CNN model for face recognition")
            recognition_result = self.cnn_trainer.predict_face(image, confidence_threshold=CNN_CONFIDENCE_THRESHOLD)
        elif self.use_custom_embedding_model and self.custom_embedding_model_available and self.custom_embedding_trainer._embedding_model is not None:
            print("🧠 Using Custom Embedding model for face recognition")
            recognition_result = self.custom_embedding_trainer.predict(image)
        elif self.use_embedding_model and self.embedding_model_available and self.embedding_trainer.model is not None:
            print("🧠 Using Embedding classifier for face recognition")
            recognition_result = self.embedding_trainer.predict(image, threshold=0.0)
        else:
            print("👤 Using InsightFace model for face recognition")
            recognition_result = self.face_manager.recognize_face(image)
        
        if recognition_result is None:
            print("❌ No face detected or recognition failed")
            return self._create_result(False, "No face detected or recognition failed")
        
        user_name, confidence = recognition_result
        
        if user_name is None:
            print("❓ Unknown person detected")
            return self._create_result(False, "Unknown person detected", 
                                      {"confidence": float(confidence) if confidence else 0.0})
        
        print(f"✓ Recognized user: {user_name} with confidence: {confidence:.4f}")

        # Record attendance
        try:
            attendance_record = self._record_attendance(user_name, confidence)
            print(f"📝 Attendance recorded for {user_name}")
        except Exception as e:
            print(f"❌ Error recording attendance: {e}")
            return self._create_result(False, f"Error recording attendance: {str(e)}")

        return self._create_result(
            True,
            "Attendance marked successfully",
            {
                "user_name": user_name,
                "confidence": float(confidence),
                "attendance_record": attendance_record,
            },
        )

    def _get_image_from_source(
        self, source: Union[np.ndarray, str, int],
    ) -> Optional[np.ndarray]:
        """Get image from various input sources.

        Args:
            source: Input source (image array, file path, camera index, or IP camera URL)

        Returns:
            Image as numpy array or None if failed
        """
        if isinstance(source, np.ndarray):
            return source

        if isinstance(source, str):
            # Check if it's an IP camera URL
            if source.startswith(("http://", "https://", "rtsp://", "rtmp://")):
                # IP camera URL
                return self.capture_from_camera(source)
            if os.path.exists(source):
                # File path
                return cv2.imread(source)
            return None

        if isinstance(source, int):
            # Camera index
            return self.capture_from_camera(source)

        return None

    def _record_attendance(self, user_name: str, confidence: float) -> Dict[str, Any]:
        """Record attendance to JSON file.

        Args:
            user_name: Name of the user
            confidence: Recognition confidence score

        Returns:
            Attendance record dictionary
        """
        now = datetime.now()
        date_str = now.strftime(ATTENDANCE_DATE_FORMAT)
        time_str = now.strftime(ATTENDANCE_TIME_FORMAT)

        attendance_record = {
            "user_name": user_name,
            "date": date_str,
            "time": time_str,
            "confidence": float(confidence),
            "timestamp": now.isoformat(),
        }

        # Save to daily attendance file
        attendance_file = os.path.join(ATTENDANCE_DIR, f"attendance_{date_str}.json")
        self._save_attendance_record(attendance_file, attendance_record)

        # Save captured image to user's database folder if recognition is successful
        if hasattr(self, '_last_captured_image') and self._last_captured_image is not None:
            user_folder = os.path.join(DATABASE_DIR, user_name)
            os.makedirs(user_folder, exist_ok=True)
            timestamp = now.strftime("%Y%m%d_%H%M%S")
            image_path = os.path.join(user_folder, f"{timestamp}_{user_name}.jpg")
            cv2.imwrite(image_path, self._last_captured_image)

        return attendance_record

    def _save_attendance_record(self, file_path: str, record: Dict[str, Any]) -> None:
        """Save attendance record to JSON file.

        Args:
            file_path: Path to attendance file
            record: Attendance record dictionary
        """
        try:
            # Ensure attendance directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # Create file with empty list if it doesn't exist yet
            if not os.path.exists(file_path):
                with open(file_path, "w") as f:
                    json.dump([], f)

            # Load existing records
            with open(file_path) as f:
                try:
                    records = json.load(f)
                    if not isinstance(records, list):  # Prevents corruption
                        records = []
                except json.JSONDecodeError:
                    records = []

            # Add new record
            records.append(record)

            # Save updated records
            with open(file_path, "w") as f:
                json.dump(records, f, indent=2)

        except Exception as e:
            print(f"Error saving attendance record: {e}")

    def _create_result(
        self, success: bool, message: str, data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create standardized result dictionary.

        Args:
            success: Whether operation was successful
            message: Result message
            data: Additional data

        Returns:
            Result dictionary
        """
        result = {
            "success": success,
            "message": message,
            "timestamp": datetime.now().isoformat(),
        }

        if data:
            result.update(data)

        return result

    def get_today_attendance(self) -> List[Dict[str, Any]]:
        """Get today's attendance records.

        Returns:
            List of attendance records for today
        """
        today = datetime.now().strftime(ATTENDANCE_DATE_FORMAT)
        attendance_file = os.path.join(ATTENDANCE_DIR, f"attendance_{today}.json")

        if os.path.exists(attendance_file):
            try:
                with open(attendance_file) as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading attendance records: {e}")

        return []

    def get_user_list(self) -> List[str]:
        """Get list of all registered users.

        Returns:
            List of user names
        """
        # Lazy load users on first request
        self._load_existing_users()
        return self.face_manager.get_user_list()

    def add_new_user(self, user_name: str, image_files: List[str]) -> Dict[str, Any]:
        """Add a new user to the system.

        Args:
            user_name: Name of the user
            image_files: List of image file paths

        Returns:
            Result dictionary
        """
        if self.face_manager.add_user_images(user_name, image_files):
            return self._create_result(True, f"User '{user_name}' added successfully")
        return self._create_result(
            False, f"Failed to add user '{user_name}'. No valid faces found.",
        )

    def capture_from_camera(
        self, camera_source: Union[int, str] = 0,
    ) -> Optional[np.ndarray]:
        """Capture single frame from camera (local or IP camera).

        Args:
            camera_source: Camera index (int) for local cameras or URL (str) for IP cameras

        Returns:
            Captured image or None if failed
        """
        cap = None
        try:
            print(f"📷 Attempting to connect to camera: {camera_source}")

            # Create VideoCapture object
            cap = cv2.VideoCapture(camera_source)

            # Set buffer size for better performance
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if cap and cap.isOpened():
                print("✅ Camera connection established")

                # Try to read a frame
                ret, frame = cap.read()

                if ret and frame is not None:
                    print(f"✅ Frame captured successfully: {frame.shape}")
                    return frame
                print("❌ Failed to read frame from camera")
                return None
            print("❌ Failed to open camera connection")
            if isinstance(camera_source, str):
                self._print_ip_camera_troubleshooting(camera_source)
            return None

        except Exception as e:
            print(f"❌ Camera capture error: {e}")
            if isinstance(camera_source, str):
                self._print_ip_camera_troubleshooting(camera_source)
            return None
        finally:
            if cap is not None:
                cap.release()

    def _print_ip_camera_troubleshooting(self, camera_url: str):
        """Print troubleshooting information for IP cameras."""
        print(f"   🔍 IP Camera Troubleshooting for: {camera_url}")
        print("   Common issues and solutions:")
        print("   1. Authentication Required (401 Error):")
        print("      - Add credentials to URL: http://username:password@IP:PORT/video")
        print("      - Example: http://admin:password@192.168.1.4:8554/video")
        print("   2. Wrong URL Format:")
        print("      - Try different endpoints:")
        print("        • /video (common for IP webcams)")
        print("        • /stream (common for ESP32-CAM)")
        print("        • /mjpeg (MJPEG streams)")
        print("        • /shot.jpg (single frame)")
        print("   3. Network Issues:")
        print(
            f"      - Check if camera is accessible: ping {camera_url.split('://')[1].split(':')[0]}",
        )
        print("      - Verify firewall settings")
        print("      - Check if camera is on the same network")
        print("   4. Camera Settings:")
        print("      - Set camera to MJPEG mode (not H.264)")
        print("      - Check camera resolution settings")
        print("      - Verify camera is streaming (not just taking photos)")
        print("   5. Test in Browser:")
        print(f"      - Try opening {camera_url} in a web browser")
        print("      - Should show video stream or ask for credentials")

    def switch_to_cnn_model(self):
        """Switch to using CNN model for recognition."""
        self.use_cnn_model = True
        self.use_embedding_model = False
        # Lazy-initialize CNN trainer on first switch
        if self.cnn_trainer is None:
            _import_trainers()
            self.cnn_trainer = CNNTrainer()
        print("🔄 Switched to CNN model for face recognition")

    def switch_to_insightface_model(self):
        """Switch to using InsightFace model for recognition."""
        self.use_cnn_model = False
        self.use_embedding_model = False
        self.use_custom_embedding_model = False
        print("🔄 Switched to InsightFace model for face recognition")

    def switch_to_embedding_model(self):
        """Switch to using the embedding classifier (if available)."""
        self.use_cnn_model = False
        self.use_custom_embedding_model = False
        self.use_embedding_model = True
        # Ensure embedding model is loaded
        if self.embedding_trainer is None:
            _import_trainers()
            self.embedding_trainer = EmbeddingTrainer()
        self.embedding_model_available = self.embedding_trainer.load_if_available()
        if self.embedding_model_available:
            print("🔄 Switched to Embedding classifier for face recognition")
        else:
            print("⚠️ Embedding classifier not available; staying on InsightFace if needed")
            self.use_embedding_model = False

    def switch_to_custom_embedding_model(self):
        """Switch to using the custom embedding model (independent from InsightFace embeddings)."""
        self.use_cnn_model = False
        self.use_embedding_model = False
        self.use_custom_embedding_model = True
        if self.custom_embedding_trainer is None:
            _import_trainers()
            self.custom_embedding_trainer = CustomEmbeddingTrainer()
        self.custom_embedding_model_available = self.custom_embedding_trainer.load_if_available()
        if self.custom_embedding_model_available:
            print("🔄 Switched to Custom Embedding model for face recognition")
        else:
            print("⚠️ Custom Embedding model not available; staying on InsightFace if needed")
            self.use_custom_embedding_model = False

    def get_current_model_info(self) -> Dict[str, Any]:
        """Get information about the currently active model.

        Returns:
            Dictionary with model information
        """
        if self.use_cnn_model:
            current = "CNN"
        elif self.use_custom_embedding_model and self.custom_embedding_trainer is not None and self.custom_embedding_trainer._embedding_model is not None:
            current = "CustomEmbedding"
        elif self.use_embedding_model and self.embedding_trainer is not None and self.embedding_trainer.model is not None:
            current = "Embedding"
        else:
            current = "InsightFace"

        # Compute on-disk availability of models to enable toggles without pre-loading heavy trainers
        try:
            # CNN artifacts
            cnn_model_path = os.path.join(CNN_MODELS_DIR, "custom_face_model.keras")
            cnn_encoder_path = os.path.join(CNN_MODELS_DIR, "label_encoder.pkl")
            cnn_files_exist = os.path.exists(cnn_model_path) and os.path.exists(cnn_encoder_path)
        except Exception:
            cnn_files_exist = False

        try:
            # Embedding classifier artifacts
            embed_dir = os.path.join(BASE_DIR, "embedding_models")
            embed_model_path = os.path.join(embed_dir, "embedding_classifier.joblib")
            embed_encoder_path = os.path.join(embed_dir, "label_encoder.pkl")
            emb_files_exist = os.path.exists(embed_model_path) and os.path.exists(embed_encoder_path)
        except Exception:
            emb_files_exist = False

        try:
            # Custom embedding artifacts
            cust_dir = os.path.join(BASE_DIR, "custom_embedding_models")
            cust_model_path = os.path.join(cust_dir, "custom_embedding_model.keras")
            cust_encoder_path = os.path.join(cust_dir, "label_encoder.pkl")
            cust_centroids_path = os.path.join(cust_dir, "class_centroids.npy")
            cust_files_exist = (
                os.path.exists(cust_model_path)
                and os.path.exists(cust_encoder_path)
                and os.path.exists(cust_centroids_path)
            )
        except Exception:
            cust_files_exist = False

        info = {
            "current_model": current,
            # Consider a model available if it's already loaded OR required files exist on disk
            "cnn_model_available": bool((self.cnn_trainer and self.cnn_trainer.model is not None) or cnn_files_exist),
            "custom_embedding_model_available": bool((self.custom_embedding_trainer and self.custom_embedding_trainer._embedding_model is not None) or cust_files_exist),
            "embedding_model_available": bool((self.embedding_trainer and self.embedding_trainer.model is not None) or emb_files_exist),
            "insightface_available": True,  # Always available
        }

        if self.cnn_trainer is not None and self.cnn_trainer.model is not None:
            info["cnn_training_status"] = self.cnn_trainer.get_training_status()
        if self.embedding_trainer is not None and self.embedding_trainer.model is not None:
            try:
                info["embedding_training_status"] = {
                    "num_classes": len(self.embedding_trainer.label_encoder.classes_),
                    "model_path": getattr(self.embedding_trainer, "model_path", None),
                }
            except Exception:
                pass
        if self.custom_embedding_trainer is not None and self.custom_embedding_trainer._embedding_model is not None:
            try:
                info["custom_embedding_training_status"] = {
                    "num_classes": len(self.custom_embedding_trainer.label_encoder.classes_),
                    "model_path": getattr(self.custom_embedding_trainer, "model_path", None),
                }
            except Exception:
                pass

        return info

    def get_cnn_trainer(self) -> "object":
        """Get the CNN trainer instance for training operations."""
        if self.cnn_trainer is None:
            _import_trainers()
            self.cnn_trainer = CNNTrainer()
        return self.cnn_trainer

    def start_automatic_recognition(
        self, camera_source: Union[int, str] = 0, delay: float = 2.0
    ) -> None:
        """Start automatic face recognition with periodic capture.

        Args:
            camera_source: Camera index (int) for local cameras or URL (str) for IP cameras
            delay: Delay between captures in seconds
        """
        print(f"📷 Starting automatic recognition with camera: {camera_source}")
        cap = cv2.VideoCapture(camera_source)

        if not cap.isOpened():
            print("❌ Failed to open camera connection")
            if isinstance(camera_source, str):
                self._print_ip_camera_troubleshooting(camera_source)
            return

        try:
            while True:
                # Capture frame
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("❌ Failed to capture frame")
                    break

                # Force InsightFace recognition
                self.switch_to_insightface_model()
                recognition_result = self.face_manager.recognize_face(frame)

                if recognition_result is not None:
                    user_name, confidence = recognition_result
                    if confidence >= SIMILARITY_THRESHOLD:  # Only save if confidence meets threshold
                        # Save recognized face to user's database folder
                        user_folder = os.path.join(DATABASE_DIR, user_name)
                        os.makedirs(user_folder, exist_ok=True)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        image_path = os.path.join(user_folder, f"{timestamp}_{user_name}.jpg")
                        cv2.imwrite(image_path, frame)
                        print(f"✅ Recognized and saved {user_name} (confidence: {confidence:.2f})")
                        # Record attendance
                        self._record_attendance(user_name, confidence)
                    else:
                        print(f"⚠️ Low confidence recognition for {user_name} ({confidence:.2f})")
                else:
                    print("👤 No known user recognized in frame")

                # Wait for specified delay
                cv2.waitKey(int(delay * 1000))  # Convert to milliseconds

        except KeyboardInterrupt:
            print("\n⏹️ Automatic recognition stopped by user")
        finally:
            cap.release()
            print("📷 Camera released")

    def draw_faces_with_names(self, image: np.ndarray) -> np.ndarray:
        """Draw bounding boxes and names on detected faces.

        Args:
            image: Input image as numpy array

        Returns:
            Image with bounding boxes and labels drawn
        """
        # Make a copy to avoid modifying the original
        output_image = image.copy()
        
        try:
            # Ensure users are loaded
            self._load_existing_users()
            
            # Detect faces using InsightFace
            faces = self.face_manager.detect_faces(image)
            
            if not faces:
                return output_image
            
            # Process each detected face
            for face in faces:
                # Get bounding box coordinates
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox
                
                # Try to recognize the face
                face_embedding = face.normed_embedding
                
                # Find best match in database
                best_match = None
                best_score = 0
                
                for user_name, stored_embedding in self.face_manager.face_database.items():
                    # Calculate cosine similarity
                    similarity = np.dot(face_embedding, stored_embedding)
                    
                    if similarity > best_score and similarity > SIMILARITY_THRESHOLD:
                        best_score = similarity
                        best_match = user_name
                
                # Determine label and color
                if best_match:
                    label = f"{best_match} ({best_score:.2f})"
                    box_color = (0, 255, 0)  # Green for recognized
                    text_bg_color = (0, 200, 0)
                else:
                    label = "Unknown"
                    box_color = (0, 0, 255)  # Red for unknown
                    text_bg_color = (0, 0, 200)
                
                # Draw bounding box
                cv2.rectangle(output_image, (x1, y1), (x2, y2), box_color, 2)
                
                # Draw label background
                label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                y1_label = max(y1, label_size[1] + 10)
                cv2.rectangle(
                    output_image,
                    (x1, y1_label - label_size[1] - 10),
                    (x1 + label_size[0], y1_label + baseline - 10),
                    text_bg_color,
                    cv2.FILLED
                )
                
                # Draw label text
                cv2.putText(
                    output_image,
                    label,
                    (x1, y1_label - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )
        
        except Exception as e:
            print(f"Error drawing faces: {e}")
        
        return output_image
