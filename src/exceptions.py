"""Custom exceptions for the face recognition attendance system."""


class FaceRecognitionError(Exception):
    """Base exception for face recognition errors."""


class ModelNotFoundError(FaceRecognitionError):
    """Raised when a required model is not found."""


class InsufficientDataError(FaceRecognitionError):
    """Raised when there's insufficient training data."""


class FaceDetectionError(FaceRecognitionError):
    """Raised when face detection fails."""


class ModelTrainingError(FaceRecognitionError):
    """Raised when model training fails."""


class UserNotFoundError(FaceRecognitionError):
    """Raised when a user is not found in the database."""


class DatabaseError(FaceRecognitionError):
    """Raised when database operations fail."""


class VideoProcessingError(FaceRecognitionError):
    """Raised when video processing fails."""


class ConfigurationError(FaceRecognitionError):
    """Raised when configuration is invalid."""
