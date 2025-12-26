# Face Recognition Attendance System

A comprehensive face recognition attendance system powered by InsightFace for production-grade recognition, with optional custom CNN training capabilities. Features professional web interface, comprehensive camera support, and robust user management.

## 📄 Abstract (Edge + Cloud Training)

This project delivers an IoT-enabled, edge-assisted attendance system that recognizes known users using InsightFace for detection, alignment, and embeddings, paired with a lightweight embedding-based classifier for production. ESP32‑CAM devices stream faces to a Raspberry Pi that acts as the edge host for the web UI, storage, and device/network orchestration; all resource‑intensive training is automated in the cloud via GitHub Actions instead of running on the Pi. On the current dataset (67 users; 9,648 samples), the InsightFace + LogisticRegression pipeline achieves **99.94% train accuracy** and **99.74% validation accuracy** (top‑3: 99.90%). Experimental baselines trained on the same dataset show varying performance: the lightweight CNN yields 64.04% validation accuracy (top‑3: 82.80%) and the custom‑embedding model 98.86% validation accuracy. These experimental paths remain available for research, while the production‑grade path is InsightFace embeddings + Logistic Regression classifier delivering exceptional accuracy. The end‑to‑end pipeline covers acquisition, detection, alignment, embedding, and matching against a persistent database, with daily JSON attendance logs, and emphasizes modularity, reliability, and maintainability for scalable deployment.

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

## 📋 Prerequisites

Before installing, ensure you have:
- **Python 3.8+** (Python 3.12 recommended)
- **pip** package manager
- **Virtual environment** (recommended)
- **System libraries**: `libgl1`, `libglib2.0-0` (Linux/Raspberry Pi)

### System Requirements
- **OS**: Windows 10/11, Linux (Ubuntu 18.04+), macOS 10.15+, or Raspberry Pi OS
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space
- **Camera**: USB webcam, IP camera, or ESP32-CAM

## 📦 Installation

### Quick Install (Recommended)

1. **Clone the repository**
```bash
git clone https://github.com/NethmiThathsarani20/face-recognition-attendance-marking-system.git
cd face-recognition-attendance-marking-system
```

2. **Create and activate virtual environment**

**Linux/macOS/Raspberry Pi:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

3. **Upgrade pip**
```bash
python -m pip install --upgrade pip
```

4. **Install dependencies**
```bash
pip install -r requirements.txt
```

This installs all required packages including:
- `insightface` - Face recognition engine
- `opencv-python` - Computer vision
- `flask` - Web framework
- `numpy`, `tensorflow`, `scikit-learn` - ML libraries
- And more...

5. **Run setup (optional)**
```bash
python setup.py
```

### System-Specific Setup

**Ubuntu/Debian/Raspberry Pi:**
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv libgl1 libglib2.0-0 python3-dev build-essential
```

**macOS:**
```bash
brew install python@3.12 opencv
```

**Windows:**
- Install [Python](https://www.python.org/downloads/) (check "Add to PATH")
- Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/)

### Development Environment (Optional)
```bash
make setup-dev
# Or manually:
pip install ruff mypy bandit pre-commit
pre-commit install
```

## 🎯 Quick Start

### 1. Activate Virtual Environment (if created)
```bash
# Linux/macOS/Raspberry Pi
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 2. Run the Application
```bash
python run.py
```

Output:
```
Starting Simple Attendance System...
Open your browser and go to: http://localhost:3000
Press Ctrl+C to stop the server
```

### 3. Access Web Interface
Open your browser: `http://localhost:3000`

**Remote Access (Raspberry Pi):**
- From same network: `http://<raspberry-pi-ip>:3000`
- Example: `http://10.74.63.231:3000`

### 4. Add Users
1. Navigate to "Add User" page
2. Enter user name
3. Upload multiple images for better accuracy (3-5 images recommended)
4. System automatically processes and stores face embeddings

### 5. Mark Attendance
Choose your preferred method:
- **Local Camera**: Select camera index (0, 1, 2, etc.) → Click "Mark Attendance"
- **IP Camera**: Enter camera URL → Click "Mark Attendance"  
- **Image Upload**: Select image file → Click "Mark Attendance"

### Quick Demo
```bash
# Test the system with existing database
python demo.py
```

## 💡 Usage Guide

### Camera Support
- **Local Cameras**: USB cameras, built-in webcams (camera index: 0, 1, 2, etc.)
- **IP Cameras**: Network cameras with MJPEG or RTSP streams
  - Android IP Webcam: `http://192.168.1.100:8080/video`
  - ESP32-CAM: `http://192.168.1.100:81/stream` or `http://10.74.63.131:81/stream`
  - With authentication: `http://user:pass@IP:PORT/video`
  - Generic MJPEG: `http://IP:PORT/video`
  - Generic RTSP: `rtsp://IP:PORT/stream`

**Testing Camera:**
```bash
# Test in browser first
# Open: http://10.74.63.131:81/stream (for ESP32-CAM)

# Or test with curl
curl http://10.74.63.131:81/stream
```

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

### Environment Setup

**Create Virtual Environment (Recommended):**
```bash
# Linux/macOS/Raspberry Pi
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate

# Verify activation (should show (venv) prefix)
which python  # Linux/macOS
where python  # Windows
```

**Deactivate Virtual Environment:**
```bash
deactivate
```

### Main Settings (`src/config.py`)
```python
# InsightFace Settings
FACE_MODEL_NAME = "buffalo_l"        # Production-grade model
SIMILARITY_THRESHOLD = 0.4           # Recognition threshold (lower = more lenient)
DETECTION_THRESHOLD = 0.5            # Face detection threshold

# Web Application
WEB_HOST = "0.0.0.0"                 # Listen on all interfaces
WEB_PORT = 3000                       # Server port
WEB_DEBUG = False                     # Debug mode (set True for development)

# Camera Configuration
DEFAULT_CAMERA_INDEX = 0              # Default USB camera
IP_CAMERA_TIMEOUT = 10                # Connection timeout in seconds

# Directories (auto-created)
DATABASE_DIR = "database"             # User images
EMBEDDINGS_DIR = "embeddings"         # Face embeddings
ATTENDANCE_DIR = "attendance_records" # Attendance logs
CNN_MODELS_DIR = "cnn_models"         # Trained models
```

### Customization Examples

**Change Server Port:**
```python
# Edit src/config.py
WEB_PORT = 5000  # Change from 3000 to 5000
```

**Adjust Recognition Sensitivity:**
```python
# More strict (fewer false positives)
SIMILARITY_THRESHOLD = 0.5

# More lenient (more false positives, fewer misses)
SIMILARITY_THRESHOLD = 0.3
```

**Change Default Camera:**
```python
DEFAULT_CAMERA_INDEX = 1  # Use second camera
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

### Run Tests

```bash
# Activate virtual environment first
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Run all tests
python tests/run_tests.py

# Or via Makefile
make test

# Run specific test file
python -m pytest tests/test_face_manager.py

# Run specific test class
python -m pytest tests/test_face_manager.py::TestFaceManager

# Run specific test method
python -m pytest tests/test_face_manager.py::TestFaceManager::test_detect_face

# Run with verbose output
python tests/run_tests.py -v

# Run with coverage
make test-coverage
# Or
coverage run tests/run_tests.py
coverage report
coverage html  # Generate HTML report in htmlcov/
```

### Test Structure

```
tests/
├── run_tests.py              # Test runner
├── test_face_manager.py      # Face detection tests
├── test_attendance_system.py # Attendance logic tests
├── test_cnn_trainer.py       # CNN training tests
└── test_web_app.py           # Web interface tests
```

### Manual Testing

```bash
# Test basic functionality
python demo.py

# Test camera capture
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera failed')"

# Test IP camera
python -c "import cv2; cap = cv2.VideoCapture('http://10.74.63.131:81/stream'); print('IP Camera OK' if cap.isOpened() else 'Failed')"

# Test face detection
python -c "from src.face_manager import FaceManager; fm = FaceManager(); print('FaceManager OK')"

# Test web server
python run.py
# Then visit http://localhost:3000
```

## 🚀 Development

### Setup Development Environment

```bash
# Install development dependencies
make setup-dev

# Or manually
pip install ruff mypy bandit pre-commit pytest coverage
pre-commit install
```

### Code Quality Tools

```bash
# Linting (auto-fix)
make lint
# Or
ruff check src/ tests/ --fix

# Check only (no fixes)
make lint-check
ruff check src/ tests/

# Format code
make format
ruff format src/ tests/

# Type checking
make type-check
mypy src/ --ignore-missing-imports --no-strict-optional

# Security scanning
make security
bandit -r src/ -f json

# Run all checks
make all-checks
```

### Pre-commit Hooks

```bash
# Install hooks
pre-commit install

# Run on all files
make pre-commit
# Or
pre-commit run --all-files

# Run on staged files only
git add .
pre-commit run
```

### Git Workflow

```bash
# Create feature branch
git checkout -b feature/new-feature

# Make changes and test
make lint
make test

# Commit
git add .
git commit -m "feat: add new feature"

# Push
git push origin feature/new-feature
```

### Project Commands Reference

```bash
make help           # Show all available commands
make install        # Install dependencies
make setup-dev      # Setup development environment
make format         # Format code with Ruff
make lint           # Lint and fix code
make lint-check     # Check without fixing
make test           # Run test suite
make test-coverage  # Run tests with coverage
make clean          # Clean temporary files
make run            # Run application
make demo           # Run demo script
make security       # Run security checks
make type-check     # Run type checking
make all-checks     # Run all quality checks
make pre-commit     # Run pre-commit hooks
```

## 📄 Dependencies

### Core Requirements

```bash
# Install all at once
pip install -r requirements.txt
```

**Required Packages:**

1. **insightface** - Production-grade face recognition engine
   - Provides buffalo_l model for face detection and embedding
   - High accuracy and performance

2. **opencv-python** - Computer vision library
   - Image processing and camera handling
   - Video capture from USB and IP cameras

3. **flask** - Lightweight web framework
   - Powers the web interface
   - RESTful API endpoints

4. **numpy** - Numerical computing
   - Array operations and mathematical functions
   - Required by InsightFace and OpenCV

5. **onnxruntime** - ONNX model runtime
   - InsightFace model execution
   - CPU and GPU support

6. **Pillow** - Image processing library
   - Image manipulation and format conversion
   - File uploads and storage

7. **werkzeug** - WSGI utility library
   - Secure file handling
   - HTTP utilities for Flask

8. **tensorflow>=2.10.0** - Deep learning framework
   - Used for optional CNN and custom embedding training
   - **Note**: Not required for production deployment (InsightFace-only mode)
   - Can be excluded if only using InsightFace for recognition

9. **scikit-learn>=1.0.0** - Machine learning library
   - Embedding classifier (Logistic Regression)
   - Model evaluation metrics

10. **matplotlib** - Plotting library
    - Training visualization and charts
    - Model performance graphs

### Optional Dependencies

```bash
# Development tools
pip install ruff mypy bandit pre-commit pytest coverage

# Alternative installation methods
# Using conda
conda install -c conda-forge insightface opencv flask numpy

# Using Docker (if Dockerfile exists)
docker build -t attendance-system .
docker run -p 3000:3000 attendance-system
```

### Dependency Details

```text
insightface         # Face recognition (buffalo_l model)
opencv-python       # Computer vision and camera
flask              # Web framework
numpy              # Numerical operations
onnxruntime        # ONNX runtime for InsightFace
Pillow             # Image processing
werkzeug           # HTTP utilities
tensorflow>=2.10.0 # Deep learning (optional training)
scikit-learn>=1.0.0# ML utilities and embedding classifier
matplotlib         # Visualization
```

### Version Requirements

- **Python**: 3.8, 3.9, 3.10, 3.11, or 3.12 (3.12 recommended)
- **pip**: Latest version (upgrade with `pip install --upgrade pip`)
- **setuptools**: Latest version
- **wheel**: Latest version

## 🔧 Troubleshooting

### ESP32-CAM Setup

1. **Hardware Requirements**
   - ESP32-CAM module
   - FTDI programmer or USB-to-TTL adapter
   - Jumper wires
   - 5V power supply

2. **Upload Firmware**
   - Open Arduino IDE
   - Install ESP32 board support:
     - Go to File → Preferences
     - Add this URL to "Additional Board Manager URLs":
       `https://dl.espressif.com/dl/package_esp32_index.json`
     - Go to Tools → Board → Boards Manager
     - Search "esp32" and install
   - Navigate to `esp32-camera/` directory
   - Open `esp32-camera.ino`
   - Update WiFi credentials in code:
     ```cpp
     const char* ssid = "YOUR_WIFI_SSID";
     const char* password = "YOUR_WIFI_PASSWORD";
     ```
   - Select board: "AI Thinker ESP32-CAM"
   - Connect GPIO0 to GND for programming mode
   - Upload sketch
   - Disconnect GPIO0 from GND and reset

3. **Get Camera IP Address**
   - Open Serial Monitor (115200 baud)
   - Reset ESP32-CAM
   - Note the IP address (e.g., `http://10.74.63.131:81/stream`)
   - Test in browser

### Raspberry Pi Setup

1. **Initial Setup**
   ```bash
   # Update system
   sudo apt update && sudo apt upgrade -y
   
   # Install dependencies
   sudo apt install python3 python3-pip python3-venv git libgl1 libglib2.0-0
   ```

2. **Clone and Install**
   ```bash
   git clone https://github.com/NethmiThathsarani20/face-recognition-attendance-marking-system.git
   cd face-recognition-attendance-marking-system
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **SSH Access**
   ```bash
   # From your computer
   ssh pi@10.74.63.231
   # Default password: raspberry
   # ⚠️ CRITICAL: Change this password immediately after first login!
   # Run: passwd
   
   # Navigate to project
   cd face-recognition-attendance-marking-system
   source venv/bin/activate
   python run.py
   ```

4. **Auto-start Service**
   ```bash
   sudo nano /etc/systemd/system/attendance.service
   ```
   
   Add:
   ```ini
   [Unit]
   Description=Face Recognition Attendance System
   After=network.target
   
   [Service]
   Type=simple
   User=pi
   WorkingDirectory=/home/pi/face-recognition-attendance-marking-system
   ExecStart=/home/pi/face-recognition-attendance-marking-system/venv/bin/python run.py
   Restart=always
   
   [Install]
   WantedBy=multi-user.target
   ```
   
   Enable:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable attendance.service
   sudo systemctl start attendance.service
   ```

### Find Device IP by MAC Address

Use the included utility:
```bash
python ip.py 80:f3:da:62:14:c0
```

---

## 🛠️ All Functions and Features

### Core Functions

#### 1. Face Recognition
```python
# Automatic face detection and recognition
# Uses InsightFace buffalo_l model
# Real-time processing
```

#### 2. User Management
```bash
# Add user via web interface or command line
mkdir -p database/John_Doe
cp images/*.jpg database/John_Doe/

# View users
python -c "from src.attendance_system import AttendanceSystem; print(AttendanceSystem().get_user_list())"
```

#### 3. Attendance Marking
```bash
# Via web interface (recommended)
# Or programmatically
python demo.py
```

#### 4. Model Training (Optional)
```bash
# Train all models
python train.py --epochs 30 --validation-split 0.2

# Train CNN only
python train_cnn.py --epochs 30

# Train embedding classifier
python train_embedding.py --epochs 30

# Train custom embedding
python train_custom_embedding.py --epochs 30
```

#### 5. Model Switching
```bash
# Access via web interface or API
curl http://localhost:3000/switch/insightface
curl http://localhost:3000/switch/cnn
curl http://localhost:3000/switch/embedding
curl http://localhost:3000/switch/custom_embedding

# Check status
curl http://localhost:3000/model_status
```

### Development Functions

```bash
# Run all tests
python tests/run_tests.py
# Or
make test

# Code quality
make lint          # Fix linting issues
make format        # Format code
make type-check    # Run type checking
make security      # Security checks

# Clean up
make clean         # Remove temporary files

# Development setup
make setup-dev     # Install dev tools
```

### Edge/Cloud Integration

```bash
# Sync database to GitHub (Raspberry Pi)
./scripts/edge_sync.sh "Add new user images"

# Automatic cloud training
# - Triggered on database/** changes
# - Runs in GitHub Actions
# - Pushes trained models back

# Pull trained models
git pull
```

### API Endpoints

The web application exposes these endpoints:

- `GET /` - Dashboard
- `GET /add_user` - Add user page
- `POST /add_user` - Process new user
- `GET /mark_attendance` - Mark attendance page
- `POST /mark_attendance` - Process attendance
- `GET /view_attendance` - View records
- `GET /train` - Training page
- `POST /train` - Start training
- `GET /model_status` - Check model status
- `POST /switch/insightface` - Switch to InsightFace
- `POST /switch/cnn` - Switch to CNN
- `POST /switch/embedding` - Switch to embedding model
- `POST /switch/custom_embedding` - Switch to custom embedding

---

## 🔧 Troubleshooting

### Common Issues

1. **Camera Not Working**
   - Check camera permissions
   - Try different camera indices (0, 1, 2)
     ```bash
     # Test camera availability
     python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"
     ```
   - For IP cameras, verify URL format and network connectivity
   - Linux: Add user to video group
     ```bash
     sudo usermod -a -G video $USER
     # Logout and login again
     ```

2. **Face Not Recognized**
   - Ensure good lighting conditions
   - Add more training images for the user (5-10 images)
   - Check if face is clearly visible and not obscured
   - Adjust similarity threshold in `src/config.py`:
     ```python
     SIMILARITY_THRESHOLD = 0.3  # Lower = more lenient (default: 0.4)
     ```

3. **IP Camera Connection Issues**
   - Verify camera URL format
   - Check authentication credentials
   - Ensure camera is on the same network
   - Try accessing camera URL in a web browser first
   - Check firewall settings
   - Test connectivity:
     ```bash
     ping 10.74.63.131  # Example ESP32-CAM IP
     curl http://10.74.63.131:81/stream
     ```

4. **Installation Issues**
   - Install Visual Studio Build Tools (Windows)
   - Use Python 3.8+ for best compatibility
   - Consider using conda environment for complex dependencies
   - On Raspberry Pi, ensure sufficient memory (swap file)
   - Upgrade pip before installing:
     ```bash
     python -m pip install --upgrade pip setuptools wheel
     ```

5. **Module Not Found Errors**
   - Ensure virtual environment is activated
     ```bash
     source venv/bin/activate  # Linux/macOS
     venv\Scripts\activate     # Windows
     ```
   - Reinstall requirements:
     ```bash
     pip install -r requirements.txt --force-reinstall
     ```
   - Add project to Python path:
     ```bash
     export PYTHONPATH="${PYTHONPATH}:$(pwd)"
     ```

6. **Out of Memory Errors**
   - Use cloud training via GitHub Actions
   - Reduce batch size in training scripts
   - Close other applications
   - On Raspberry Pi, increase swap:
     ```bash
     sudo dphys-swapfile swapoff
     sudo nano /etc/dphys-swapfile  # Set CONF_SWAPSIZE=2048
     sudo dphys-swapfile setup
     sudo dphys-swapfile swapon
     ```

7. **Permission Denied Errors**
   - Fix directory permissions:
     ```bash
     chmod -R 755 .
     chmod +x scripts/edge_sync.sh
     ```
   - Create missing directories:
     ```bash
     mkdir -p database embeddings attendance_records
     ```

8. **Virtual Environment Issues**
   - Recreate virtual environment:
     ```bash
     rm -rf venv
     python3 -m venv venv
     source venv/bin/activate
     pip install -r requirements.txt
     ```

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

## 📋 Quick Reference Guide

### Installation Quick Start
```bash
git clone https://github.com/NethmiThathsarani20/face-recognition-attendance-marking-system.git
cd face-recognition-attendance-marking-system
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python run.py
# Visit http://localhost:3000
```

### Common Commands
```bash
# Start application
python run.py

# Run demo
python demo.py

# Run tests
python tests/run_tests.py

# Train models
python train.py --epochs 30

# Lint code
make lint

# Find device IP
python ip.py <MAC_ADDRESS>
```

### Network Examples
```bash
# SSH to Raspberry Pi (replace IP with your device's IP)
ssh pi@10.74.63.231

# ESP32-CAM stream URL (example IP, yours will be different)
http://10.74.63.131:81/stream

# Raspberry Pi web app (example IP, yours will be different)
http://10.74.63.231:3000

# Test camera connection (replace with your camera IP)
curl http://10.74.63.131:81/stream
```

**Note**: Replace example IP addresses (10.74.63.x) with your actual device IP addresses.

### File Locations
```
database/               # User face images
embeddings/             # Face embeddings (pickle files)
attendance_records/     # Daily attendance JSON logs
cnn_models/             # Trained CNN models
embedding_models/       # Embedding classifier models
templates/              # HTML templates
static/                 # CSS, JavaScript files
src/                    # Python source code
tests/                  # Test suite
```

### Important URLs
- **Local**: http://localhost:3000
- **Remote**: http://<raspberry-pi-ip>:3000
- **Model Status**: http://localhost:3000/model_status
- **Switch to InsightFace**: POST http://localhost:3000/switch/insightface
- **Switch to CNN**: POST http://localhost:3000/switch/cnn

### Troubleshooting Quick Fixes
```bash
# Camera not working
sudo usermod -a -G video $USER

# Module not found
source venv/bin/activate
pip install -r requirements.txt

# Permission denied
chmod +x scripts/edge_sync.sh

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Test camera
python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"
```

### Configuration Files
- `src/config.py` - Main configuration
- `requirements.txt` - Python dependencies
- `pyproject.toml` - Ruff linting config
- `Makefile` - Build and dev commands
- `.github/workflows/train.yml` - CI/CD pipeline

### Support and Documentation
- **Full Setup Guide**: See `INSTRUCTIONS.md`
- **Technical Details**: See `docs/TECHNICAL_OVERVIEW.md`
- **Model Training**: See `docs/MODEL_TRAINING.md`
- **Project Structure**: See `docs/STRUCTURE.md`
- **GitHub Issues**: https://github.com/NethmiThathsarani20/face-recognition-attendance-marking-system/issues

---

**For detailed setup instructions, see [INSTRUCTIONS.md](INSTRUCTIONS.md)**

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
