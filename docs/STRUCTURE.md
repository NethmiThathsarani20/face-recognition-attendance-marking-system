# Project Structure

```
face-recognition-attendance-marking-system/
├── src/
│   ├── config.py              # System configuration with InsightFace defaults
│   ├── face_manager.py        # InsightFace detection and recognition (primary engine)
│   ├── cnn_trainer.py         # Custom CNN training capabilities (optional)
│   ├── attendance_system.py   # Main attendance logic using InsightFace
│   ├── web_app.py            # Flask web interface with full functionality
│   └── exceptions.py         # Custom exception framework
├── templates/
│   ├── index.html            # Main attendance dashboard
│   ├── add_user.html         # User registration with multi-image upload
│   └── cnn_training.html     # CNN training management interface
├── static/
│   └── style.css             # Professional styling for web interface
├── database/                 # User images organized by folders (auto-created)
├── embeddings/              # InsightFace embeddings (pickle files)
├── cnn_models/              # CNN model storage directory (when trained)
│   ├── custom_face_model.keras  # Trained CNN model
│   ├── label_encoder.pkl     # Class label encoder
│   └── training_log.json     # Training history and metrics
├── attendance_records/      # Daily JSON attendance files
├── temp/                    # Temporary file storage for uploads
├── tests/                   # Comprehensive test suite
│   ├── test_face_manager.py   # InsightFace module tests
│   ├── test_cnn_trainer.py   # CNN training tests
│   ├── test_attendance_system.py # Main system tests
│   ├── test_integration.py   # Integration tests
│   └── run_tests.py         # Test runner
├── requirements.txt         # Core dependencies (InsightFace, OpenCV, Flask, TensorFlow)
├── pyproject.toml          # Ruff configuration and project metadata
├── Makefile                # Development automation
├── README.md               # Complete documentation
└── run.py                  # Main entry point
```

## Key Components

### 1. Face Manager (`face_manager.py`)
- Uses InsightFace FaceAnalysis with buffalo_l model for state-of-the-art recognition
- Handles face detection, alignment, and embedding generation
- Stores/loads embeddings using pickle files for persistence
- Multi-image support for users with average embedding calculation
- Automatic face saving for recognized users to database folders

### 2. Attendance System (`attendance_system.py`)
- Primary attendance marking logic using InsightFace recognition
- Unified processing for camera capture, image upload, and file paths
- Comprehensive IP camera support with troubleshooting guidance
- JSON-based attendance logging with daily files
- Automatic image saving for recognized faces with timestamps

### 3. Web Application (`web_app.py`)
- Professional Flask web interface with multiple pages
- Camera integration (local USB and IP cameras)
- User management with multi-image upload support
- CNN training interface for optional custom model training
- Real-time status updates and progress monitoring

### 4. CNN Trainer (`cnn_trainer.py`)
- Optional custom CNN training for specialized scenarios
- Lightweight 6-layer CNN architecture optimized for face recognition
- Video processing for training data extraction
- Model persistence with automatic saving and loading
- Integration with InsightFace for face alignment and preprocessing

### 5. Configuration (`config.py`)
- Comprehensive settings using InsightFace defaults for optimal performance
- Directory management with automatic creation
- Camera configuration for both local and IP cameras
- Model selection and threshold configuration

### 6. Exception Framework (`exceptions.py`)
- Custom exception hierarchy for better error handling
- Specific exceptions for different failure modes
- Enhanced debugging and error reporting capabilities

## Design Principles

1. **InsightFace-First Architecture** - Primary focus on production-ready InsightFace recognition with optional CNN training
2. **Modular Design** - Clear separation of concerns with dedicated modules for each functionality
3. **Professional Web Interface** - Multi-page web application with camera integration and real-time updates
4. **Comprehensive Error Handling** - Custom exception framework with graceful error recovery
5. **Flexible Input Sources** - Support for camera capture, image upload, file paths, and IP cameras
6. **Quality Assurance** - Comprehensive testing, linting with Ruff, and modern Python practices
7. **Persistence** - Automatic data management with embeddings, attendance records, and user images
8. **Performance Optimization** - Efficient face detection and recognition with minimal configuration

## Technical Architecture

### Recognition Pipeline
1. **Image Input** - Camera capture, file upload, or direct array input via web interface
2. **Face Detection** - InsightFace detection with automatic quality assessment and alignment
3. **Recognition** - Primary InsightFace embedding matching with optional CNN fallback
4. **Recording** - JSON attendance logging with automatic image saving to user folders
5. **Web Interface** - Real-time updates and comprehensive user management
