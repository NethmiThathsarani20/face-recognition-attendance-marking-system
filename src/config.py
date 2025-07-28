"""Simple configuration file for attendance system.
Uses InsightFace defaults as much as possible.
"""

import os

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATABASE_DIR = os.path.join(BASE_DIR, "database")
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")
ATTENDANCE_DIR = os.path.join(BASE_DIR, "attendance_records")

# CNN Models directory
CNN_MODELS_DIR = os.path.join(BASE_DIR, "cnn_models")

# Create directories if they don't exist
for directory in [DATABASE_DIR, EMBEDDINGS_DIR, ATTENDANCE_DIR, CNN_MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)

# InsightFace settings (using defaults)
FACE_MODEL_NAME = "buffalo_l"  # Default InsightFace model
DETECTION_SIZE = (640, 640)  # Default detection size
DETECTION_THRESHOLD = 0.5  # Default detection threshold
SIMILARITY_THRESHOLD = 0.4  # Default similarity threshold for matching

# Web application settings
WEB_HOST = "0.0.0.0"
WEB_PORT = 5000
WEB_DEBUG = True

# File settings
EMBEDDINGS_FILE = os.path.join(EMBEDDINGS_DIR, "face_embeddings.pkl")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp"}

# Camera settings
DEFAULT_CAMERA_INDEX = 0
CAMERA_RESOLUTION = (640, 480)
IP_CAMERA_TIMEOUT = 10  # seconds
IP_CAMERA_BUFFER_SIZE = 1  # reduce latency

# Common IP camera URL formats (examples)
IP_CAMERA_FORMATS = {
    "generic_mjpeg": "http://IP:PORT/video",
    "generic_rtsp": "rtsp://IP:PORT/stream",
    "android_ip_webcam": "http://IP:8080/video",
    "esp32_cam": "http://IP:81/stream",
}

# Attendance settings
ATTENDANCE_DATE_FORMAT = "%Y-%m-%d"
ATTENDANCE_TIME_FORMAT = "%H:%M:%S"

# Model selection settings
USE_CNN_MODEL = False  # False: use InsightFace, True: use custom CNN
CNN_CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence for CNN predictions
