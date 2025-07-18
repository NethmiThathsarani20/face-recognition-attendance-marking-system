"""
Main attendance system logic.
Handles attendance marking for both camera and upload inputs.
"""

import os
import cv2
import json
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any, Union, List

# Handle both relative and absolute imports
try:
    from .face_manager import FaceManager
    from .config import ATTENDANCE_DIR, ATTENDANCE_DATE_FORMAT, ATTENDANCE_TIME_FORMAT
except ImportError:
    from face_manager import FaceManager
    from config import ATTENDANCE_DIR, ATTENDANCE_DATE_FORMAT, ATTENDANCE_TIME_FORMAT


class AttendanceSystem:
    """
    Simple attendance system with unified processing for camera and upload.
    """
    
    def __init__(self):
        """Initialize attendance system with face manager."""
        self.face_manager = FaceManager()
        self._load_existing_users()
    
    def _load_existing_users(self) -> None:
        """Load all existing users from database folder."""
        loaded_count = self.face_manager.load_all_database_users()
        print(f"Loaded {loaded_count} users from database")
    
    def mark_attendance(self, input_source: Union[np.ndarray, str, int]) -> Dict[str, Any]:
        """
        Unified attendance marking function for camera, upload, or image array.
        
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
        
        # Recognize face
        recognition_result = self.face_manager.recognize_face(image)
        if recognition_result is None:
            return self._create_result(False, "No face recognized")
        
        user_name, confidence = recognition_result
        
        # Record attendance
        attendance_record = self._record_attendance(user_name, confidence)
        
        return self._create_result(True, "Attendance marked successfully", {
            'user_name': user_name,
            'confidence': float(confidence),
            'attendance_record': attendance_record
        })
    
    def _get_image_from_source(self, source: Union[np.ndarray, str, int]) -> Optional[np.ndarray]:
        """
        Get image from various input sources.
        
        Args:
            source: Input source (image array, file path, camera index, or IP camera URL)
            
        Returns:
            Image as numpy array or None if failed
        """
        if isinstance(source, np.ndarray):
            return source
        
        elif isinstance(source, str):
            # Check if it's an IP camera URL
            if source.startswith(('http://', 'https://', 'rtsp://', 'rtmp://')):
                # IP camera URL
                return self.capture_from_camera(source)
            elif os.path.exists(source):
                # File path
                return cv2.imread(source)
            return None
        
        elif isinstance(source, int):
            # Camera index
            return self.capture_from_camera(source)
        
        return None
    
    def _record_attendance(self, user_name: str, confidence: float) -> Dict[str, Any]:
        """
        Record attendance to JSON file.
        
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
            'user_name': user_name,
            'date': date_str,
            'time': time_str,
            'confidence': float(confidence),
            'timestamp': now.isoformat()
        }
        
        # Save to daily attendance file
        attendance_file = os.path.join(ATTENDANCE_DIR, f"attendance_{date_str}.json")
        self._save_attendance_record(attendance_file, attendance_record)
        
        return attendance_record
    
    def _save_attendance_record(self, file_path: str, record: Dict[str, Any]) -> None:
        """
        Save attendance record to JSON file.
        
        Args:
            file_path: Path to attendance file
            record: Attendance record dictionary
        """
        try:
            # Load existing records
            records = []
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    records = json.load(f)
            
            # Add new record
            records.append(record)
            
            # Save updated records
            with open(file_path, 'w') as f:
                json.dump(records, f, indent=2)
                
        except Exception as e:
            print(f"Error saving attendance record: {e}")
    
    def _create_result(self, success: bool, message: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create standardized result dictionary.
        
        Args:
            success: Whether operation was successful
            message: Result message
            data: Additional data
            
        Returns:
            Result dictionary
        """
        result = {
            'success': success,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
        
        if data:
            result.update(data)
            
        return result
    
    def get_today_attendance(self) -> List[Dict[str, Any]]:
        """
        Get today's attendance records.
        
        Returns:
            List of attendance records for today
        """
        today = datetime.now().strftime(ATTENDANCE_DATE_FORMAT)
        attendance_file = os.path.join(ATTENDANCE_DIR, f"attendance_{today}.json")
        
        if os.path.exists(attendance_file):
            try:
                with open(attendance_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading attendance records: {e}")
        
        return []
    
    def get_user_list(self) -> List[str]:
        """
        Get list of all registered users.
        
        Returns:
            List of user names
        """
        return self.face_manager.get_user_list()
    
    def add_new_user(self, user_name: str, image_files: List[str]) -> Dict[str, Any]:
        """
        Add a new user to the system.
        
        Args:
            user_name: Name of the user
            image_files: List of image file paths
            
        Returns:
            Result dictionary
        """
        if self.face_manager.add_user_images(user_name, image_files):
            return self._create_result(True, f"User '{user_name}' added successfully")
        else:
            return self._create_result(False, f"Failed to add user '{user_name}'. No valid faces found.")
    
    def capture_from_camera(self, camera_source: Union[int, str] = 0) -> Optional[np.ndarray]:
        """
        Capture single frame from camera (local or IP camera).
        
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
                else:
                    print("❌ Failed to read frame from camera")
                    return None
            else:
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
        print(f"      - Check if camera is accessible: ping {camera_url.split('://')[1].split(':')[0]}")
        print("      - Verify firewall settings")
        print("      - Check if camera is on the same network")
        print("   4. Camera Settings:")
        print("      - Set camera to MJPEG mode (not H.264)")
        print("      - Check camera resolution settings")
        print("      - Verify camera is streaming (not just taking photos)")
        print("   5. Test in Browser:")
        print(f"      - Try opening {camera_url} in a web browser")
        print("      - Should show video stream or ask for credentials")
