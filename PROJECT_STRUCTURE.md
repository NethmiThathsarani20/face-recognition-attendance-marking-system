# Face Recognition Attendance System - Project Structure

This document describes the clean, professional structure of the project.

## Root Directory Files

### Main Entry Points
- `run.py` - Main application entry point (starts Flask web server)
- `demo.py` - Demo script for testing face recognition
- `train.py` - Unified training script for all models (CNN, Embedding, Custom Embedding)

### Alternative Training Scripts
- `train_cnn.py` - Standalone CNN training (provides granular control)
- `train_embedding.py` - Standalone embedding classifier training (extra options: max_iter, C, solver, penalty)
- `train_custom_embedding.py` - Standalone custom embedding training

### Utility Scripts
- `setup.py` - Project setup and installation
- `verify_requirements.py` - Dependency verification
- `ip.py` - Network utility to find device IP by MAC address

### Configuration Files
- `requirements.txt` - Python dependencies
- `pyproject.toml` - Ruff linting and build configuration
- `Makefile` - Build, test, and development commands
- `.gitignore` - Git ignore patterns

### Documentation
- `README.md` - Comprehensive project documentation with setup, usage, and troubleshooting
- `INSTRUCTIONS.md` - Detailed installation and setup instructions

## Source Code (`src/`)

Core Python modules:
- `config.py` - System configuration and settings
- `face_manager.py` - InsightFace integration for face detection and recognition
- `attendance_system.py` - Main attendance tracking logic
- `web_app.py` - Flask web application and routes
- `exceptions.py` - Custom exception framework
- `cnn_trainer.py` - Lightweight CNN model trainer (uses Haar Cascade)
- `embedding_trainer.py` - Embedding classifier trainer (InsightFace + Logistic Regression)
- `custom_embedding_trainer.py` - Custom embedding model trainer (experimental)

## Templates (`templates/`)

HTML templates for web interface:
- `index.html` - Main dashboard with attendance marking
- `add_user.html` - User registration page
- `cnn_training.html` - CNN training page (currently disabled but available for re-enabling)

## Static Files (`static/`)

- `style.css` - Optimized CSS for Raspberry Pi (solid colors, minimal animations)

## Tests (`tests/`)

Comprehensive test suite:
- `run_tests.py` - Test runner
- `test_face_manager.py` - Face detection and recognition tests
- `test_attendance_system.py` - Attendance logic tests
- `test_cnn_trainer.py` - CNN training tests
- `test_config.py` - Configuration tests
- `test_auth.py` - Authentication tests
- `test_integration.py` - Integration tests
- `test_ip_camera.py` - IP camera integration tests

## Scripts (`scripts/`)

Helper utilities:
- `edge_sync.sh` - Raspberry Pi edge sync script (commits and pushes database images to GitHub)
- `generate_model_comparison.py` - Generate model comparison charts and tables after training

## Documentation (`docs/`)

Detailed documentation:
- `MODEL_TRAINING.md` - Comprehensive model training guide
- `TECHNICAL_OVERVIEW.md` - Technical architecture overview
- `STRUCTURE.md` - Project structure documentation

## ESP32-CAM (`esp32-camera/`)

Arduino firmware for ESP32-CAM:
- `esp32-camera.ino` - Main Arduino sketch
- `app_httpd.cpp` - HTTP server implementation
- `camera_pins.h` - Pin configuration
- `camera_index.h` - Web interface HTML
- `board_config.h` - Board configuration
- `partitions.csv` - Flash memory partitions

## Data Directories (Auto-created)

- `database/` - User face images (67 users, 9,648 samples, 240×240 pixels)
- `embeddings/` - Cached face embeddings (pickle files)
- `attendance_records/` - Daily attendance JSON logs
- `cnn_models/` - Trained CNN model artifacts and evaluation charts
- `embedding_models/` - Trained embedding classifier artifacts and charts
- `custom_embedding_models/` - Trained custom embedding artifacts and charts

## CI/CD (`.github/workflows/`)

- `train.yml` - GitHub Actions workflow for cloud training on database updates

## Key Features

### Clean Structure
- No duplicate or temporary files
- All cache files properly ignored
- Clear separation of concerns
- Well-documented code

### Professional Organization
- Unified training interface (`train.py`)
- Modular architecture
- Comprehensive tests
- Automated CI/CD

### Development Tools
- Makefile commands for common tasks
- Pre-commit hooks support
- Ruff linting configuration
- Type checking with mypy

## Common Commands

```bash
# Run application
python run.py

# Train all models
python train.py --epochs 30 --validation-split 0.2

# Train specific model
python train.py --only embedding
python train.py --only cnn
python train.py --only custom-embedding

# Run tests
python tests/run_tests.py
make test

# Code quality
make lint          # Lint and fix
make format        # Format code
make type-check    # Type checking
make all-checks    # All checks

# Cleanup
make clean         # Remove cache and temporary files
```

## Notes

- The project uses InsightFace for production face recognition (best accuracy)
- CNN and custom embedding models are available for research and comparison
- The web UI is optimized for Raspberry Pi performance
- All training can be done in the cloud via GitHub Actions
- Edge devices (Raspberry Pi) can sync data using `scripts/edge_sync.sh`
