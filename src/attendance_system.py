"""Main attendance system logic.
Handles attendance marking for both camera and upload inputs.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np

# Handle both relative and absolute imports
try:
    from .cnn_trainer import CNNTrainer
    from .config import (
        ATTENDANCE_DATE_FORMAT,
        ATTENDANCE_DIR,
        ATTENDANCE_TIME_FORMAT,
        CNN_CONFIDENCE_THRESHOLD,
        USE_CNN_MODEL,
    )
    from .face_manager import FaceManager
except ImportError:
    from cnn_trainer import CNNTrainer
    from config import (
        ATTENDANCE_DATE_FORMAT,
        ATTENDANCE_DIR,
        ATTENDANCE_TIME_FORMAT,
        CNN_CONFIDENCE_THRESHOLD,
        USE_CNN_MODEL,
    )
    from face_manager import FaceManager


class AttendanceSystem:
    """Simple attendance system with unified processing for camera and upload.
    """

    def __init__(self):
        """Initialize attendance system with face manager and CNN trainer."""
        self.face_manager = FaceManager()
        self.cnn_trainer = CNNTrainer()
        self.use_cnn_model = USE_CNN_MODEL
        self._load_existing_users()

    def _load_existing_users(self) -> None:
        """Load all existing users from database folder."""
        loaded_count = self.face_manager.load_all_database_users()
        print(f"Loaded {loaded_count} users from database")

    def mark_attendance(
        self, input_source: Union[np.ndarray, str, int],
    ) -> Dict[str, Any]:
        """Unified attendance marking function for camera, upload, or image array.

        Args:
            input_source: Can be:
                - np.ndarray: Image array
                - str: Path to image file
                - int: Camera index

        Returns:
            Dictionary with attendance result and metadata
        """
        # Get image from input source
        image = self._get_image_from_source(input_source)
        if image is None:
            return self._create_result(False, "Failed to get image from source")

        # Recognize face using either CNN or InsightFace
        recognition_result = None

        if self.use_cnn_model and self.cnn_trainer.model is not None:
            # Try CNN model first
            recognition_result = self.cnn_trainer.predict_face(
                image, CNN_CONFIDENCE_THRESHOLD,
            )

        if recognition_result is None:
            # Fall back to InsightFace or use as primary method
            recognition_result = self.face_manager.recognize_face(image)

        if recognition_result is None:
            # If InsightFace doesn't recognize and we have CNN with auto-training enabled
            if self.cnn_trainer.auto_training_enabled:
                # Add as unknown user for training
                unknown_name = self.cnn_trainer.add_unknown_user(image)
                return self._create_result(
                    False, f"Unknown face saved as {unknown_name} for training",
                )
            return self._create_result(False, "No face recognized")

        user_name, confidence = recognition_result

        # Record attendance
        attendance_record = self._record_attendance(user_name, confidence)

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

        return attendance_record

    def _save_attendance_record(self, file_path: str, record: Dict[str, Any]) -> None:
        """Save attendance record to JSON file.

        Args:
            file_path: Path to attendance file
            record: Attendance record dictionary
        """
        try:
            # Load existing records
            records = []
            if os.path.exists(file_path):
                with open(file_path) as f:
                    records = json.load(f)

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
        print("🔄 Switched to CNN model for face recognition")

    def switch_to_insightface_model(self):
        """Switch to using InsightFace model for recognition."""
        self.use_cnn_model = False
        print("🔄 Switched to InsightFace model for face recognition")

    def get_current_model_info(self) -> Dict[str, Any]:
        """Get information about the currently active model.

        Returns:
            Dictionary with model information
        """
        info = {
            "current_model": "CNN" if self.use_cnn_model else "InsightFace",
            "cnn_model_available": self.cnn_trainer.model is not None,
            "insightface_available": True,  # Always available
            "auto_training_enabled": self.cnn_trainer.auto_training_enabled,
        }

        if self.cnn_trainer.model is not None:
            info["cnn_training_status"] = self.cnn_trainer.get_training_status()

        return info

    def get_cnn_trainer(self) -> CNNTrainer:
        """Get the CNN trainer instance for training operations."""
        return self.cnn_trainer
