"""
Unit tests for attendance_system module.
"""

import unittest
import os
import sys
import json
import tempfile
import shutil
from datetime import datetime
import numpy as np
import cv2
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from src.attendance_system import AttendanceSystem
from test_config import create_test_image, TEST_CONFIG


class TestAttendanceSystem(unittest.TestCase):
    """
    Test cases for AttendanceSystem class.
    """
    
    def setUp(self):
        """Set up test case."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_attendance_dir = None
    
    def tearDown(self):
        """Clean up test case."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @patch('src.attendance_system.FaceManager')
    def test_attendance_system_initialization(self, mock_face_manager):
        """Test AttendanceSystem initialization."""
        mock_fm = MagicMock()
        mock_face_manager.return_value = mock_fm
        mock_fm.load_all_database_users.return_value = 5
        
        attendance_system = AttendanceSystem()
        
        self.assertIsNotNone(attendance_system)
        mock_face_manager.assert_called_once()
        mock_fm.load_all_database_users.assert_called_once()
    
    @patch('src.attendance_system.FaceManager')
    def test_mark_attendance_with_image_array(self, mock_face_manager):
        """Test marking attendance with image array."""
        mock_fm = MagicMock()
        mock_face_manager.return_value = mock_fm
        mock_fm.load_all_database_users.return_value = 1
        mock_fm.recognize_face.return_value = ('test_user', 0.85)
        
        attendance_system = AttendanceSystem()
        test_image = create_test_image()
        
        with patch.object(attendance_system, '_record_attendance') as mock_record:
            mock_record.return_value = {'user_name': 'test_user', 'time': '10:00:00'}
            
            result = attendance_system.mark_attendance(test_image)
            
            self.assertTrue(result['success'])
            self.assertEqual(result['user_name'], 'test_user')
            self.assertEqual(result['confidence'], 0.85)
            mock_fm.recognize_face.assert_called_once_with(test_image)
    
    @patch('src.attendance_system.FaceManager')
    @patch('src.attendance_system.cv2.imread')
    @patch('src.attendance_system.os.path.exists')
    def test_mark_attendance_with_file_path(self, mock_exists, mock_imread, mock_face_manager):
        """Test marking attendance with file path."""
        mock_fm = MagicMock()
        mock_face_manager.return_value = mock_fm
        mock_fm.load_all_database_users.return_value = 1
        mock_fm.recognize_face.return_value = ('test_user', 0.85)
        
        mock_exists.return_value = True
        mock_imread.return_value = create_test_image()
        
        attendance_system = AttendanceSystem()
        
        with patch.object(attendance_system, '_record_attendance') as mock_record:
            mock_record.return_value = {'user_name': 'test_user', 'time': '10:00:00'}
            
            result = attendance_system.mark_attendance('test_image.jpg')
            
            self.assertTrue(result['success'])
            mock_exists.assert_called_once_with('test_image.jpg')
            mock_imread.assert_called_once_with('test_image.jpg')
    
    @patch('src.attendance_system.FaceManager')
    @patch('src.attendance_system.cv2.VideoCapture')
    def test_mark_attendance_with_camera(self, mock_video_capture, mock_face_manager):
        """Test marking attendance with camera."""
        mock_fm = MagicMock()
        mock_face_manager.return_value = mock_fm
        mock_fm.load_all_database_users.return_value = 1
        mock_fm.recognize_face.return_value = ('test_user', 0.85)
        
        # Mock camera capture
        mock_cap = MagicMock()
        mock_video_capture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, create_test_image())
        
        attendance_system = AttendanceSystem()
        
        with patch.object(attendance_system, '_record_attendance') as mock_record:
            mock_record.return_value = {'user_name': 'test_user', 'time': '10:00:00'}
            
            result = attendance_system.mark_attendance(0)
            
            self.assertTrue(result['success'])
            mock_video_capture.assert_called_once_with(0)
            mock_cap.read.assert_called_once()
            mock_cap.release.assert_called_once()
    
    @patch('src.attendance_system.FaceManager')
    @patch('src.attendance_system.cv2.VideoCapture')
    def test_mark_attendance_with_ip_camera(self, mock_video_capture, mock_face_manager):
        """Test marking attendance with IP camera."""
        mock_fm = MagicMock()
        mock_face_manager.return_value = mock_fm
        mock_fm.load_all_database_users.return_value = 1
        mock_fm.recognize_face.return_value = ('test_user', 0.85)
        
        # Mock IP camera capture
        mock_cap = MagicMock()
        mock_video_capture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, create_test_image())
        
        attendance_system = AttendanceSystem()
        
        with patch.object(attendance_system, '_record_attendance') as mock_record:
            mock_record.return_value = {'user_name': 'test_user', 'time': '10:00:00'}
            
            ip_camera_url = "http://192.168.1.100:8080/video"
            result = attendance_system.mark_attendance(ip_camera_url)
            
            self.assertTrue(result['success'])
            mock_video_capture.assert_called_once_with(ip_camera_url)
            mock_cap.set.assert_called_once_with(cv2.CAP_PROP_BUFFERSIZE, 1)
            mock_cap.read.assert_called_once()
            mock_cap.release.assert_called_once()
    
    @patch('src.attendance_system.FaceManager')
    def test_mark_attendance_no_face_recognized(self, mock_face_manager):
        """Test marking attendance when no face is recognized."""
        mock_fm = MagicMock()
        mock_face_manager.return_value = mock_fm
        mock_fm.load_all_database_users.return_value = 1
        mock_fm.recognize_face.return_value = None  # No face recognized
        
        attendance_system = AttendanceSystem()
        test_image = create_test_image()
        
        result = attendance_system.mark_attendance(test_image)
        
        self.assertFalse(result['success'])
        self.assertEqual(result['message'], 'No face recognized')
    
    @patch('src.attendance_system.FaceManager')
    @patch('src.attendance_system.ATTENDANCE_DIR', new_callable=lambda: tempfile.mkdtemp())
    def test_record_attendance(self, mock_attendance_dir, mock_face_manager):
        """Test recording attendance."""
        mock_fm = MagicMock()
        mock_face_manager.return_value = mock_fm
        mock_fm.load_all_database_users.return_value = 1
        
        attendance_system = AttendanceSystem()
        
        record = attendance_system._record_attendance('test_user', 0.85)
        
        self.assertEqual(record['user_name'], 'test_user')
        self.assertEqual(record['confidence'], 0.85)
        self.assertIn('date', record)
        self.assertIn('time', record)
        self.assertIn('timestamp', record)
        
        # Check if file was created
        today = datetime.now().strftime('%Y-%m-%d')
        attendance_file = os.path.join(mock_attendance_dir, f'attendance_{today}.json')
        self.assertTrue(os.path.exists(attendance_file))
        
        # Check file content
        with open(attendance_file, 'r') as f:
            records = json.load(f)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]['user_name'], 'test_user')
    
    @patch('src.attendance_system.FaceManager')
    def test_get_today_attendance(self, mock_face_manager):
        """Test getting today's attendance."""
        mock_fm = MagicMock()
        mock_face_manager.return_value = mock_fm
        mock_fm.load_all_database_users.return_value = 1
        
        attendance_system = AttendanceSystem()
        
        with patch('src.attendance_system.ATTENDANCE_DIR', self.temp_dir):
            # Create test attendance file
            today = datetime.now().strftime('%Y-%m-%d')
            attendance_file = os.path.join(self.temp_dir, f'attendance_{today}.json')
            test_records = [
                {'user_name': 'user1', 'time': '09:00:00'},
                {'user_name': 'user2', 'time': '09:15:00'}
            ]
            
            with open(attendance_file, 'w') as f:
                json.dump(test_records, f)
            
            records = attendance_system.get_today_attendance()
            
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]['user_name'], 'user1')
            self.assertEqual(records[1]['user_name'], 'user2')
    
    @patch('src.attendance_system.FaceManager')
    def test_get_user_list(self, mock_face_manager):
        """Test getting user list."""
        mock_fm = MagicMock()
        mock_face_manager.return_value = mock_fm
        mock_fm.load_all_database_users.return_value = 2
        mock_fm.get_user_list.return_value = ['user1', 'user2']
        
        attendance_system = AttendanceSystem()
        
        users = attendance_system.get_user_list()
        
        self.assertEqual(len(users), 2)
        self.assertIn('user1', users)
        self.assertIn('user2', users)
    
    @patch('src.attendance_system.FaceManager')
    def test_add_new_user(self, mock_face_manager):
        """Test adding new user."""
        mock_fm = MagicMock()
        mock_face_manager.return_value = mock_fm
        mock_fm.load_all_database_users.return_value = 1
        mock_fm.add_user_images.return_value = True
        
        attendance_system = AttendanceSystem()
        
        result = attendance_system.add_new_user('new_user', ['img1.jpg', 'img2.jpg'])
        
        self.assertTrue(result['success'])
        self.assertIn('new_user', result['message'])
        mock_fm.add_user_images.assert_called_once_with('new_user', ['img1.jpg', 'img2.jpg'])
    
    @patch('src.attendance_system.FaceManager')
    def test_add_new_user_failure(self, mock_face_manager):
        """Test adding new user failure."""
        mock_fm = MagicMock()
        mock_face_manager.return_value = mock_fm
        mock_fm.load_all_database_users.return_value = 1
        mock_fm.add_user_images.return_value = False
        
        attendance_system = AttendanceSystem()
        
        result = attendance_system.add_new_user('new_user', ['img1.jpg', 'img2.jpg'])
        
        self.assertFalse(result['success'])
        self.assertIn('Failed to add user', result['message'])
    
    @patch('src.attendance_system.FaceManager')
    @patch('src.attendance_system.cv2.VideoCapture')
    def test_capture_from_camera(self, mock_video_capture, mock_face_manager):
        """Test capturing from camera."""
        mock_fm = MagicMock()
        mock_face_manager.return_value = mock_fm
        mock_fm.load_all_database_users.return_value = 1
        
        # Mock camera capture
        mock_cap = MagicMock()
        mock_video_capture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        test_image = create_test_image()
        mock_cap.read.return_value = (True, test_image)
        
        attendance_system = AttendanceSystem()
        
        captured_image = attendance_system.capture_from_camera(0)
        
        self.assertIsNotNone(captured_image)
        np.testing.assert_array_equal(captured_image, test_image)
        mock_video_capture.assert_called_once_with(0)
        mock_cap.read.assert_called_once()
        mock_cap.release.assert_called_once()
    
    @patch('src.attendance_system.FaceManager')
    @patch('src.attendance_system.cv2.VideoCapture')
    def test_capture_from_camera_failure(self, mock_video_capture, mock_face_manager):
        """Test capturing from camera failure."""
        mock_fm = MagicMock()
        mock_face_manager.return_value = mock_fm
        mock_fm.load_all_database_users.return_value = 1
        
        # Mock camera failure
        mock_cap = MagicMock()
        mock_video_capture.return_value = mock_cap
        mock_cap.isOpened.return_value = False
        
        attendance_system = AttendanceSystem()
        
        captured_image = attendance_system.capture_from_camera(0)
        
        self.assertIsNone(captured_image)


if __name__ == '__main__':
    unittest.main()
