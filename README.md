# Face Recognition Attendance System

A comprehensive face recognition attendance system powered by InsightFace for production-grade recognition, with optional custom CNN training capabilities. Features professional web interface, comprehensive camera support, and robust user management.

## 📄 Abstract (Edge + Cloud Training)

This project delivers an IoT-enabled, edge-assisted attendance system that recognizes known users using InsightFace for detection, alignment, and embeddings, paired with a lightweight embedding-based classifier for production. ESP32‑CAM devices stream faces to a Raspberry Pi that acts as the edge host for the web UI, storage, and device/network orchestration; all resource‑intensive training is automated in the cloud via GitHub Actions instead of running on the Pi. On the current dataset (65 users; 1,545 embeddings), the InsightFace + LogisticRegression pipeline achieves 0.994 train accuracy and 0.984 validation accuracy (top‑3: 0.984). Experimental baselines trained on the same dataset perform poorly: the lightweight CNN yields ~3.9% validation accuracy (top‑3: ~6.1%) and the custom‑embedding model ~1.0% validation accuracy. These experimental paths remain available for research, while the production‑grade path is InsightFace embeddings + a simple classifier. The end‑to‑end pipeline covers acquisition, detection, alignment, embedding, and matching against a persistent database, with daily JSON attendance logs, and emphasizes modularity, reliability, and maintainability for scalable deployment.

Keywords
face recognition; attendance system; insightface; convolutional neural network; computer vision; esp32-cam

## 🚀 Features

### Core Recognition System
- **InsightFace Integration**: Production-ready face recognition using buffalo_l model
- **Multi-Input Support**: Camera capture, image upload, and file path processing
- **Advanced Face Detection**: High-quality face detection and alignment
- **Embedding Storage**: Efficient pickle-based persistence system

### Professional Web Interface
- **Multi-Page Application**: Clean dashboard with navigation between features
- **Camera Integration**: Support for local USB cameras and IP cameras (MJPEG/RTSP)
- **User Management**: Add users with multiple image uploads for better accuracy
- **Real-Time Updates**: Live attendance tracking with immediate feedback

### Advanced Capabilities
- **Training Modes (optional)**: Lightweight CNN, embedding-based classifier (InsightFace + Logistic Regression), and an experimental custom-embedding model
- **Video Processing**: Extract training data from video uploads
- **Automatic Image Management**: Recognized faces saved to user folders with timestamps
- **Comprehensive Error Handling**: Custom exception framework with graceful recovery
- **Quality Assurance**: Modern Python practices with Ruff linting and type safety
- **Model Switching**: Runtime toggle between InsightFace, CNN, embedding classifier, and custom-embedding backends

## 📦 Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Optional: Set up development environment
make setup-dev
# (Optional) Developer tools
# pip install ruff mypy bandit
```

## 🎯 Quick Start

### 1. Run the Application
```bash
python run.py
```

### 2. Access Web Interface
Open your browser: `http://localhost:3000`

### 3. Add Users
1. Navigate to "Add User" page
2. Enter user name
3. Upload multiple images for better accuracy
4. System automatically processes and stores face embeddings

### 4. Mark Attendance
Choose your preferred method:
- **Local Camera**: Select camera index → Click "Mark Attendance"
- **IP Camera**: Enter camera URL → Click "Mark Attendance"  
- **Image Upload**: Select image file → Click "Mark Attendance"

## 💡 Usage Guide

### Camera Support
- **Local Cameras**: USB cameras, built-in webcams (camera index: 0, 1, 2, etc.)
- **IP Cameras**: Network cameras with MJPEG or RTSP streams
  - Android IP Webcam: `http://192.168.1.100:8080/video`
  - ESP32-CAM: `http://192.168.1.100:81/stream`
  - With authentication: `http://user:pass@IP:PORT/video`
  - Generic MJPEG: `http://IP:PORT/video`
  - Generic RTSP: `rtsp://IP:PORT/stream`

### User Management
1. **Add Users**: Upload multiple images per user for improved recognition accuracy
2. **Database Structure**: Images automatically organized in `database/username/` folders
3. **Automatic Processing**: Face detection and embedding generation happens automatically
4. **Multi-Image Support**: System averages multiple embeddings for better accuracy

### Attendance Workflow
1. **Face Detection**: InsightFace detects and aligns faces automatically
2. **Recognition**: Matches against stored user embeddings
3. **Recording**: Saves attendance to JSON files with timestamps
4. **Image Saving**: Automatically saves recognized faces to user folders

### Optional CNN Training
1. **Access Training**: Navigate to "CNN Training" page
2. **Prepare Data**: System extracts training data from user images
3. **Train Model**: Configure and train the lightweight CNN
4. **Video Training**: Upload videos to extract multiple training frames
5. **Switch Models**: Toggle between InsightFace, CNN, embedding classifier, and custom-embedding
   - API endpoints: `/switch/insightface`, `/switch/cnn`, `/switch/embedding`, `/switch/custom_embedding`, status at `/model_status`

## ⚙️ Configuration

### Main Settings (`src/config.py`)
```python
# InsightFace Settings
FACE_MODEL_NAME = "buffalo_l"        # Production-grade model
SIMILARITY_THRESHOLD = 0.4           # Recognition threshold
DETECTION_THRESHOLD = 0.5            # Face detection threshold

# Web Application
WEB_HOST = "0.0.0.0"
WEB_PORT = 3000
WEB_DEBUG = True

# Camera Configuration
DEFAULT_CAMERA_INDEX = 0
IP_CAMERA_TIMEOUT = 10
```

## 🏗️ Project Structure

```
face-recognition-based-attendance-marking-system/
├── .github/workflows/           # CI workflows (cloud training)
│   └── train.yml                # GitHub Actions: trains CNN + embedding (and custom-embedding)
├── train.py                     # Unified trainer (cnn, embedding, custom-embedding)
├── train_cnn.py                 # CNN-only CI entrypoint (optional)
├── train_embedding.py           # Standalone embedding-classifier trainer
├── train_custom_embedding.py    # Standalone custom-embedding trainer (experimental)
├── src/                         # Source code modules
│   ├── config.py                # System configuration
│   ├── face_manager.py          # InsightFace integration
│   ├── cnn_trainer.py           # Optional CNN training
│   ├── attendance_system.py     # Main attendance logic
│   ├── web_app.py               # Flask web interface
│   └── exceptions.py            # Custom exception framework
├── templates/                   # HTML templates
├── static/                      # CSS and JavaScript
├── database/                    # User images (auto-created)
├── embeddings/                  # Face embeddings storage
├── cnn_models/                  # Trained CNN artifacts
├── embedding_models/            # Trained embedding classifier artifacts
├── custom_embedding_models/     # Trained custom-embedding artifacts (experimental)
├── attendance_records/          # Daily attendance JSON files
├── tests/                       # Comprehensive test suite
├── scripts/                     # Helper scripts (edge sync)
│   └── edge_sync.sh             # Commit & push database images from Raspberry Pi
└── requirements.txt             # Dependencies
```

## 🧪 Testing

```bash
# Run all tests
python tests/run_tests.py

# Or via Makefile
make test

# Run a specific test class (example)
python -m pytest tests.test_face_manager.TestFaceManager
```

## 🚀 Development

```bash
# Development setup
make setup-dev

# Code quality checks
make lint           # Fix linting issues
make format         # Format code
make type-check     # Run type checking
make test           # Run test suite
```

## 📄 Dependencies

Core requirements:
- **insightface**: Production-grade face recognition
- **opencv-python**: Image processing and camera handling
- **flask**: Web framework for user interface
- **numpy**: Numerical operations and array handling
- **onnxruntime**: InsightFace ONNX runtime providers
- **Pillow**, **werkzeug**: Image and web utilities
- **tensorflow**: Required for CNN/custom-embedding training (optional in production)
- **scikit-learn**: Embedding-classifier (Logistic Regression)

## 🔧 Troubleshooting

### Common Issues

1. **Camera Not Working**
   - Check camera permissions
   - Try different camera indices (0, 1, 2)
   - For IP cameras, verify URL format and network connectivity

2. **Face Not Recognized**
   - Ensure good lighting conditions
   - Add more training images for the user
   - Check if face is clearly visible and not obscured

3. **IP Camera Connection Issues**
   - Verify camera URL format
   - Check authentication credentials
   - Ensure camera is on the same network
   - Try accessing camera URL in a web browser first

4. **Installation Issues**
   - Install Visual Studio Build Tools (Windows)
   - Use Python 3.8+ for best compatibility
   - Consider using conda environment for complex dependencies

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with proper testing
4. Ensure code quality with `make lint`
5. Submit a pull request

## 📚 Technical Details

### Recognition Pipeline
1. **Image Input** → **Face Detection** → **Alignment** → **Embedding Generation** → **Matching** → **Attendance Recording**

### Storage Structure
- **Embeddings**: Pickle files for fast loading
- **Attendance**: Daily JSON files with timestamps
- **Images**: Organized by user in database folders

### Performance Optimization
- Efficient embedding storage and retrieval
- Optimized camera capture with buffer management
- Minimal configuration approach using InsightFace defaults

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⭐ Acknowledgments

- **InsightFace**: For providing excellent face recognition models
- **OpenCV**: For comprehensive computer vision capabilities
- **Flask**: For the lightweight web framework

---

## ☁️ Cloud Training with GitHub Actions

Minimal setup so the Raspberry Pi stays light and the cloud does the heavy lifting.

What happens:
- Edge (Pi) captures images to `database/<user>/...` (or `database1/`, `database2/` if you use alternative folders)
- Pi commits and pushes those images to this repo
- GitHub Actions trains on every push that touches `database/**`, `database1/**`, or `database2/**`
- Trained artifacts are written to `cnn_models/`, `embedding_models/`, and `custom_embedding_models/` and pushed back to the repo

Included files:
- `.github/workflows/train.yml` – CI workflow (trains CNN + embedding + custom-embedding)
- `train.py` – unified training entrypoint used by CI
- `scripts/edge_sync.sh` – helper to push only database updates from the Pi

### Raspberry Pi (Edge) steps
1) Put new images per user under `database/<User_Name>/...`
2) Push changes:
   - First time only: make it executable: `chmod +x scripts/edge_sync.sh`
   - Run: `./scripts/edge_sync.sh "Add images for <User_Name>"`
   - Optional: add a cron job to auto-push hourly if changes exist.

### Cloud (CI) steps
- Trigger: any push modifying `database/**`, `database1/**`, or `database2/**` (you can also run the workflow manually)
- Action runner does:
   - Install deps via `requirements.txt`
   - Run `python train.py --epochs 30 --validation-split 0.2`
   - Commit updated artifacts in `cnn_models/`, `embedding_models/`, and `custom_embedding_models/`

Notes:
- The workflow only triggers on `database/**` changes, so model pushes won’t loop CI.
- `train.py` re-prepares data and retrains selected models, overwriting artifacts as needed.

### Switching model in the app
InsightFace is the default. You can switch at runtime:
- `/switch/insightface`
- `/switch/cnn`
- `/switch/embedding`
- `/switch/custom_embedding`
Check `/model_status` for availability and details.
