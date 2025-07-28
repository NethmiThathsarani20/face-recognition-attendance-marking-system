"""Unit tests for CNN trainer module.
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

from src.cnn_trainer import CNNTrainer


class TestCNNTrainer(unittest.TestCase):
    """Test cases for CNNTrainer class.
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
        # Create temporary directory for models
        self.temp_dir = tempfile.mkdtemp()

        # Mock the model paths to use temp directory
        with patch("src.cnn_trainer.os.path.dirname") as mock_dirname:
            mock_dirname.return_value = self.temp_dir
            with patch("src.face_manager.FaceAnalysis"):
                self.cnn_trainer = CNNTrainer()
                # Override paths to use temp directory
                self.cnn_trainer.models_dir = os.path.join(self.temp_dir, "cnn_models")
                os.makedirs(self.cnn_trainer.models_dir, exist_ok=True)
                self.cnn_trainer.model_path = os.path.join(
                    self.cnn_trainer.models_dir, "test_model.h5",
                )
                self.cnn_trainer.encoder_path = os.path.join(
                    self.cnn_trainer.models_dir, "test_encoder.pkl",
                )
                self.cnn_trainer.training_log_path = os.path.join(
                    self.cnn_trainer.models_dir, "test_log.json",
                )

    def tearDown(self):
        """Clean up test case."""
        # Clean up temporary directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch("src.face_manager.FaceAnalysis")
    def test_cnn_trainer_initialization(self, mock_face_analysis):
        """Test CNNTrainer initialization."""
        mock_app = MagicMock()
        mock_face_analysis.return_value = mock_app

        with patch("src.cnn_trainer.os.path.dirname") as mock_dirname:
            mock_dirname.return_value = self.temp_dir
            trainer = CNNTrainer()

            self.assertIsNotNone(trainer)
            self.assertIsNotNone(trainer.face_manager)
            self.assertEqual(trainer.target_size, (112, 112))
            self.assertTrue(trainer.auto_training_enabled)

    def test_create_model(self):
        """Test CNN model creation."""
        with patch("src.cnn_trainer.models.Sequential") as mock_sequential:
            mock_model = MagicMock()
            mock_sequential.return_value = mock_model

            model = self.cnn_trainer.create_model(5)

            self.assertIsNotNone(model)
            mock_sequential.assert_called_once()
            mock_model.compile.assert_called_once()

    @patch("src.face_manager.FaceAnalysis")
    @patch("src.cnn_trainer.os.listdir")
    @patch("src.cnn_trainer.os.path.exists")
    def test_prepare_training_data(self, mock_exists, mock_listdir, mock_face_analysis):
        """Test training data preparation."""
        # Mock face manager
        mock_app = MagicMock()
        mock_face_analysis.return_value = mock_app

        # Mock file system
        mock_exists.return_value = True
        mock_listdir.return_value = ["image1.jpg", "image2.jpg"]

        # Mock face manager methods
        self.cnn_trainer.face_manager.get_user_list = MagicMock(
            return_value=["user1", "user2"],
        )

        # Mock face extraction
        with patch.object(self.cnn_trainer, "_extract_and_align_face") as mock_extract:
            mock_extract.return_value = np.random.rand(112, 112, 3)

            result = self.cnn_trainer.prepare_training_data()

            self.assertTrue(result)
            self.assertGreater(len(self.cnn_trainer.training_data), 0)
            self.assertGreater(len(self.cnn_trainer.training_labels), 0)

    @patch("src.face_manager.FaceAnalysis")
    def test_extract_and_align_face(self, mock_face_analysis):
        """Test face extraction and alignment."""
        # Create test image file
        test_image = create_test_image()
        temp_image_path = os.path.join(self.temp_dir, "test_image.jpg")

        # Mock cv2.imread
        with patch("src.cnn_trainer.cv2.imread") as mock_imread:
            mock_imread.return_value = test_image

            # Mock face detection
            mock_face = MagicMock()
            mock_face.bbox = np.array([50, 50, 150, 150])
            self.cnn_trainer.face_manager.detect_faces = MagicMock(
                return_value=[mock_face],
            )

            # Mock cv2.resize
            with patch("src.cnn_trainer.cv2.resize") as mock_resize:
                mock_resize.return_value = np.random.rand(112, 112, 3)

                result = self.cnn_trainer._extract_and_align_face(temp_image_path)

                self.assertIsNotNone(result)
                if result is not None:
                    self.assertEqual(result.shape, (112, 112, 3))
                mock_imread.assert_called_once_with(temp_image_path)

    @patch("src.face_manager.FaceAnalysis")
    def test_extract_and_align_face_no_face(self, mock_face_analysis):
        """Test face extraction when no face is detected."""
        # Create test image file
        test_image = create_test_image()
        temp_image_path = os.path.join(self.temp_dir, "test_image.jpg")

        with patch("src.cnn_trainer.cv2.imread") as mock_imread:
            mock_imread.return_value = test_image

            # Mock no face detection
            self.cnn_trainer.face_manager.detect_faces = MagicMock(return_value=[])

            result = self.cnn_trainer._extract_and_align_face(temp_image_path)

            self.assertIsNone(result)

    def test_predict_face_no_model(self):
        """Test face prediction when no model is loaded."""
        test_image = create_test_image()

        result = self.cnn_trainer.predict_face(test_image)

        self.assertIsNone(result)

    @patch("src.face_manager.FaceAnalysis")
    def test_predict_face_with_model(self, mock_face_analysis):
        """Test face prediction with loaded model."""
        # Mock model
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array(
            [[0.1, 0.8, 0.1]],
        )  # High confidence for class 1
        self.cnn_trainer.model = mock_model

        # Mock label encoder
        mock_encoder = MagicMock()
        mock_encoder.classes_ = ["user1", "user2", "user3"]
        self.cnn_trainer.label_encoder = mock_encoder

        # Mock face detection
        mock_face = MagicMock()
        mock_face.bbox = np.array([50, 50, 150, 150])
        self.cnn_trainer.face_manager.detect_faces = MagicMock(return_value=[mock_face])

        test_image = create_test_image()

        with patch("src.cnn_trainer.cv2.resize") as mock_resize:
            mock_resize.return_value = np.random.rand(112, 112, 3)

            result = self.cnn_trainer.predict_face(test_image, confidence_threshold=0.7)

            self.assertIsNotNone(result)
            if result is not None:
                self.assertEqual(result[0], "user2")
                self.assertAlmostEqual(result[1], 0.8, places=1)

    @patch("src.face_manager.FaceAnalysis")
    def test_predict_face_low_confidence(self, mock_face_analysis):
        """Test face prediction with low confidence."""
        # Mock model with low confidence
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([[0.3, 0.4, 0.3]])  # Low confidence
        self.cnn_trainer.model = mock_model

        # Mock label encoder
        mock_encoder = MagicMock()
        mock_encoder.classes_ = ["user1", "user2", "user3"]
        self.cnn_trainer.label_encoder = mock_encoder

        # Mock face detection
        mock_face = MagicMock()
        mock_face.bbox = np.array([50, 50, 150, 150])
        self.cnn_trainer.face_manager.detect_faces = MagicMock(return_value=[mock_face])

        test_image = create_test_image()

        with patch("src.cnn_trainer.cv2.resize") as mock_resize:
            mock_resize.return_value = np.random.rand(112, 112, 3)

            result = self.cnn_trainer.predict_face(test_image, confidence_threshold=0.7)

            self.assertIsNone(result)

    @patch("src.face_manager.FaceAnalysis")
    @patch("src.cnn_trainer.cv2.imwrite")
    def test_add_unknown_user(self, mock_imwrite, mock_face_analysis):
        """Test adding unknown user."""
        test_image = create_test_image()

        with patch("src.cnn_trainer.os.makedirs"):
            result = self.cnn_trainer.add_unknown_user(test_image)

            self.assertIsNotNone(result)
            self.assertTrue(result.startswith("unknown_user_"))
            mock_imwrite.assert_called_once()

    def test_toggle_auto_training(self):
        """Test toggling auto-training mode."""
        initial_state = self.cnn_trainer.auto_training_enabled

        self.cnn_trainer.toggle_auto_training(not initial_state)

        self.assertEqual(self.cnn_trainer.auto_training_enabled, not initial_state)

    def test_get_training_status(self):
        """Test getting training status."""
        status = self.cnn_trainer.get_training_status()

        self.assertIsInstance(status, dict)
        self.assertIn("model_exists", status)
        self.assertIn("auto_training_enabled", status)
        self.assertIn("training_data_count", status)
        self.assertIn("unknown_counter", status)

    @patch("src.face_manager.FaceAnalysis")
    @patch("src.cnn_trainer.cv2.VideoCapture")
    def test_add_training_data_from_video(self, mock_video_capture, mock_face_analysis):
        """Test extracting training data from video."""
        # Mock video capture
        mock_cap = MagicMock()
        mock_video_capture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        mock_cap.read.side_effect = [
            (True, create_test_image()),
            (False, None),
        ]  # One frame then end

        # Mock face extraction
        with patch.object(
            self.cnn_trainer, "_extract_and_align_face_from_frame",
        ) as mock_extract:
            mock_extract.return_value = np.random.rand(112, 112, 3)

            with patch("src.cnn_trainer.os.path.exists") as mock_exists:
                mock_exists.return_value = True

                result = self.cnn_trainer.add_training_data_from_video(
                    "test_video.mp4", "test_user",
                )

                self.assertTrue(result["success"])
                self.assertGreater(result["faces_extracted"], 0)

    @patch("src.face_manager.FaceAnalysis")
    def test_add_training_data_from_video_file_not_found(self, mock_face_analysis):
        """Test video processing with non-existent file."""
        result = self.cnn_trainer.add_training_data_from_video(
            "nonexistent.mp4", "test_user",
        )

        self.assertFalse(result["success"])
        self.assertIn("not found", result["message"])


if __name__ == "__main__":
    unittest.main()
