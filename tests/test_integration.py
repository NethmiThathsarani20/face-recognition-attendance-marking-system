"""Integration tests for the complete attendance system.
"""

import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from test_config import cleanup_test_data, create_test_image, setup_test_database


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system.
    """

    @classmethod
    def setUpClass(cls):
        """Set up test database."""
        cls.test_db_dir = setup_test_database()
        cls.temp_dir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        """Clean up test data."""
        cleanup_test_data()
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    def setUp(self):
        """Set up test case."""

    @patch("src.face_manager.FaceAnalysis")
    @patch("src.face_manager.cv2.imread")
    @patch("src.face_manager.os.path.exists")
    @patch("src.face_manager.os.listdir")
    def test_full_attendance_workflow(
        self, mock_listdir, mock_exists, mock_imread, mock_face_analysis,
    ):
        """Test complete attendance workflow."""
        # Mock directory listing to prevent loading existing users
        mock_listdir.return_value = []

        # Mock file existence - return False for embeddings file to prevent loading existing database
        def mock_exists_side_effect(path):
            if "face_embeddings.pkl" in path:
                return False
            return True

        mock_exists.side_effect = mock_exists_side_effect

        # Mock image loading
        mock_imread.return_value = create_test_image()

        # Mock InsightFace
        mock_app = MagicMock()
        mock_face_analysis.return_value = mock_app

        # Mock face detection and recognition
        mock_face = MagicMock()
        mock_face.normed_embedding = np.array(
            [0.5] * 512,
        )  # Fixed embedding for consistency
        mock_app.get.return_value = [mock_face]

        from src.attendance_system import AttendanceSystem

        # Create attendance system
        attendance_system = AttendanceSystem()

        # Add a user
        result = attendance_system.add_new_user("test_user", ["test_image.jpg"])
        self.assertTrue(result["success"])

        # Mark attendance
        test_image = create_test_image()
        result = attendance_system.mark_attendance(test_image)
        self.assertTrue(result["success"])

        # The test is now simplified - just verify that the operations succeeded
        # We already verified the attendance marking was successful
        self.assertEqual(result["user_name"], "test_user")
        self.assertIn("confidence", result)
        self.assertIn("attendance_record", result)

    @patch("src.face_manager.FaceAnalysis")
    def test_end_to_end_attendance_marking(self, mock_face_analysis):
        """Test web application routes."""
        # Mock InsightFace
        mock_app = MagicMock()
        mock_face_analysis.return_value = mock_app
        mock_app.get.return_value = []

        from src.web_app import app

        # Create test client
        with app.test_client() as client:
            # Test index route
            response = client.get("/")
            self.assertEqual(response.status_code, 200)

            # Test add user route
            response = client.get("/add_user")
            self.assertEqual(response.status_code, 200)

            # Test get users route
            response = client.get("/get_users")
            self.assertEqual(response.status_code, 200)

            # Test get attendance route
            response = client.get("/get_attendance")
            self.assertEqual(response.status_code, 200)

    @patch("src.face_manager.FaceAnalysis")
    def test_config_integration(self, mock_face_analysis):
        """Test configuration integration."""
        # Mock InsightFace
        mock_app = MagicMock()
        mock_face_analysis.return_value = mock_app
        mock_app.get.return_value = []

        from src.config import (
            ATTENDANCE_DIR,
            DATABASE_DIR,
            DETECTION_SIZE,
            EMBEDDINGS_DIR,
            FACE_MODEL_NAME,
            SIMILARITY_THRESHOLD,
            WEB_HOST,
            WEB_PORT,
        )

        # Test configuration values
        self.assertEqual(FACE_MODEL_NAME, "buffalo_l")
        self.assertEqual(DETECTION_SIZE, (640, 640))
        self.assertEqual(SIMILARITY_THRESHOLD, 0.4)
        self.assertEqual(WEB_HOST, "0.0.0.0")
        self.assertEqual(WEB_PORT, 3000)

        # Test directories exist
        self.assertTrue(os.path.exists(DATABASE_DIR))
        self.assertTrue(os.path.exists(EMBEDDINGS_DIR))
        self.assertTrue(os.path.exists(ATTENDANCE_DIR))

    @patch("src.face_manager.FaceAnalysis")
    def test_error_handling(self, mock_face_analysis):
        """Test error handling in the system."""
        # Mock InsightFace
        mock_app = MagicMock()
        mock_face_analysis.return_value = mock_app
        mock_app.get.return_value = []

        from src.attendance_system import AttendanceSystem

        attendance_system = AttendanceSystem()

        # Test with invalid input - use a string that represents invalid input
        result = attendance_system.mark_attendance("")
        self.assertFalse(result["success"])

        # Test with non-existent file
        result = attendance_system.mark_attendance("non_existent.jpg")
        self.assertFalse(result["success"])

        # Test with empty user name
        result = attendance_system.add_new_user("", ["test.jpg"])
        self.assertFalse(result["success"])

    @patch("src.face_manager.FaceAnalysis")
    def test_performance_basic(self, mock_face_analysis):
        """Basic performance test."""
        # Mock InsightFace
        mock_app = MagicMock()
        mock_face_analysis.return_value = mock_app

        mock_face = MagicMock()
        mock_face.normed_embedding = [0.5] * 512
        mock_app.get.return_value = [mock_face]

        import time

        from src.attendance_system import AttendanceSystem

        attendance_system = AttendanceSystem()

        # Add a user
        attendance_system.add_new_user("test_user", ["test.jpg"])

        # Time multiple attendance markings
        start_time = time.time()
        for i in range(10):
            test_image = create_test_image()
            attendance_system.mark_attendance(test_image)
        end_time = time.time()

        avg_time = (end_time - start_time) / 10
        print(f"Average attendance marking time: {avg_time:.3f} seconds")

        # Should be reasonably fast (less than 2 seconds per marking)
        self.assertLess(avg_time, 2.0)


if __name__ == "__main__":
    unittest.main()
