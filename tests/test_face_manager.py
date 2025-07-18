"""
Unit tests for face_manager module.
"""

import unittest
import os
import sys
import cv2
import numpy as np
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from src.face_manager import FaceManager
from test_config import setup_test_database, cleanup_test_data, create_test_image, get_test_image_path


class TestFaceManager(unittest.TestCase):
    """
    Test cases for FaceManager class.
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up test database."""
        cls.test_db_dir = setup_test_database()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test data."""
        cleanup_test_data()
    
    def setUp(self):
        """Set up test case."""
        # Mock InsightFace to avoid model loading during tests
        self.face_manager = None
    
    @patch('src.face_manager.FaceAnalysis')
    def test_face_manager_initialization(self, mock_face_analysis):
        """Test FaceManager initialization."""
        mock_app = MagicMock()
        mock_face_analysis.return_value = mock_app
        
        face_manager = FaceManager()
        
        self.assertIsNotNone(face_manager)
        mock_face_analysis.assert_called_once()
        mock_app.prepare.assert_called_once()
    
    @patch('src.face_manager.FaceAnalysis')
    def test_detect_faces(self, mock_face_analysis):
        """Test face detection."""
        mock_app = MagicMock()
        mock_face_analysis.return_value = mock_app
        
        # Mock face detection result
        mock_face = MagicMock()
        mock_face.normed_embedding = np.random.rand(512)
        mock_app.get.return_value = [mock_face]
        
        face_manager = FaceManager()
        test_image = create_test_image()
        
        faces = face_manager.detect_faces(test_image)
        
        self.assertEqual(len(faces), 1)
        mock_app.get.assert_called_once()
    
    @patch('src.face_manager.FaceAnalysis')
    def test_get_face_embedding(self, mock_face_analysis):
        """Test face embedding extraction."""
        mock_app = MagicMock()
        mock_face_analysis.return_value = mock_app
        
        # Mock face with embedding
        mock_face = MagicMock()
        mock_embedding = np.random.rand(512)
        mock_face.normed_embedding = mock_embedding
        mock_app.get.return_value = [mock_face]
        
        face_manager = FaceManager()
        test_image = create_test_image()
        
        embedding = face_manager.get_face_embedding(test_image)
        
        self.assertIsNotNone(embedding)
        if embedding is not None:
            self.assertEqual(len(embedding), 512)
            np.testing.assert_array_equal(embedding, mock_embedding)
    
    @patch('src.face_manager.FaceAnalysis')
    def test_get_face_embedding_no_face(self, mock_face_analysis):
        """Test face embedding extraction when no face is detected."""
        mock_app = MagicMock()
        mock_face_analysis.return_value = mock_app
        mock_app.get.return_value = []  # No faces detected
        
        face_manager = FaceManager()
        test_image = create_test_image()
        
        embedding = face_manager.get_face_embedding(test_image)
        
        self.assertIsNone(embedding)
    
    @patch('src.face_manager.FaceAnalysis')
    @patch('src.face_manager.cv2.imread')
    @patch('src.face_manager.os.path.exists')
    def test_add_user_images(self, mock_exists, mock_imread, mock_face_analysis):
        """Test adding user images."""
        mock_app = MagicMock()
        mock_face_analysis.return_value = mock_app
        
        # Mock file existence
        mock_exists.return_value = True
        
        # Mock image loading and face detection
        mock_imread.return_value = create_test_image()
        mock_face = MagicMock()
        mock_face.normed_embedding = np.random.rand(512)
        mock_app.get.return_value = [mock_face]
        
        face_manager = FaceManager()
        
        # Test adding user with valid images
        result = face_manager.add_user_images('test_user', ['img1.jpg', 'img2.jpg'])
        
        self.assertTrue(result)
        self.assertIn('test_user', face_manager.face_database)
        self.assertEqual(mock_imread.call_count, 2)
        self.assertGreaterEqual(mock_exists.call_count, 2)
    
    @patch('src.face_manager.FaceAnalysis')
    def test_recognize_face(self, mock_face_analysis):
        """Test face recognition."""
        mock_app = MagicMock()
        mock_face_analysis.return_value = mock_app
        
        face_manager = FaceManager()
        
        # Add a user to database
        test_embedding = np.random.rand(512)
        face_manager.face_database['test_user'] = test_embedding
        
        # Mock face detection for query
        mock_face = MagicMock()
        mock_face.normed_embedding = test_embedding + 0.1 * np.random.rand(512)  # Similar embedding
        mock_app.get.return_value = [mock_face]
        
        test_image = create_test_image()
        result = face_manager.recognize_face(test_image)
        
        self.assertIsNotNone(result)
        if result is not None:
            self.assertEqual(result[0], 'test_user')
            self.assertGreater(result[1], 0.0)
    
    @patch('src.face_manager.FaceAnalysis')
    def test_recognize_face_no_match(self, mock_face_analysis):
        """Test face recognition with no match."""
        mock_app = MagicMock()
        mock_face_analysis.return_value = mock_app
        
        face_manager = FaceManager()
        
        # Add a user to database
        face_manager.face_database['test_user'] = np.random.rand(512)
        
        # Mock face detection for query with very different embedding
        mock_face = MagicMock()
        mock_face.normed_embedding = np.random.rand(512) * -1  # Very different embedding
        mock_app.get.return_value = [mock_face]
        
        test_image = create_test_image()
        result = face_manager.recognize_face(test_image)
        
        self.assertIsNone(result)
    
    @patch('src.face_manager.FaceAnalysis')
    def test_get_user_list(self, mock_face_analysis):
        """Test getting user list."""
        mock_app = MagicMock()
        mock_face_analysis.return_value = mock_app
        
        face_manager = FaceManager()
        
        # Clear existing database and add specific users
        face_manager.face_database.clear()
        face_manager.face_database['user1'] = np.random.rand(512)
        face_manager.face_database['user2'] = np.random.rand(512)
        
        users = face_manager.get_user_list()
        
        self.assertEqual(len(users), 2)
        self.assertIn('user1', users)
        self.assertIn('user2', users)
    
    @patch('src.face_manager.FaceAnalysis')
    def test_preprocess_image(self, mock_face_analysis):
        """Test image preprocessing."""
        mock_app = MagicMock()
        mock_face_analysis.return_value = mock_app
        
        face_manager = FaceManager()
        
        # Test with BGR image - create a test image with different color channels
        bgr_image = np.zeros((100, 100, 3), dtype=np.uint8)
        bgr_image[:, :, 0] = 255  # Blue channel
        bgr_image[:, :, 1] = 0    # Green channel
        bgr_image[:, :, 2] = 0    # Red channel
        
        processed = face_manager.preprocess_image(bgr_image)
        
        self.assertEqual(processed.shape, bgr_image.shape)
        # Should convert BGR to RGB, so red channel should now be 255
        self.assertEqual(processed[0, 0, 0], 0)    # Red channel in RGB (was blue in BGR)
        self.assertEqual(processed[0, 0, 2], 255)  # Blue channel in RGB (was red in BGR)


if __name__ == '__main__':
    unittest.main()
