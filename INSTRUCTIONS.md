# Face Recognition Attendance System - Setup Instructions

This document provides comprehensive setup instructions for the Face Recognition Attendance Marking System, including prerequisites, environment setup, installation, and usage.

---

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Prerequisites](#prerequisites)
3. [Environment Setup](#environment-setup)
4. [Installation Steps](#installation-steps)
5. [Running the Application](#running-the-application)
6. [Hardware Setup](#hardware-setup)
7. [Usage Guide](#usage-guide)
8. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Minimum Requirements
- **Operating System**: Windows 10/11, Linux (Ubuntu 18.04+), macOS 10.15+, or Raspberry Pi OS (Bullseye or later)
- **Python**: Version 3.8 or higher (Python 3.12 recommended)
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space
- **Camera**: USB webcam, IP camera, or ESP32-CAM module

### For Raspberry Pi (Edge Device)
- **Model**: Raspberry Pi 3B+ or newer (Raspberry Pi 4 recommended)
- **RAM**: 2GB minimum, 4GB+ recommended
- **Storage**: 16GB+ microSD card
- **Network**: WiFi or Ethernet connection

---

## Prerequisites

### 1. Python Installation
Ensure Python 3.8 or higher is installed on your system.

**Check Python version:**
```bash
python --version
# or
python3 --version
```

**Install Python (if needed):**
- **Ubuntu/Debian:**
  ```bash
  sudo apt update
  sudo apt install python3 python3-pip python3-venv
  ```
- **macOS:**
  ```bash
  brew install python@3.12
  ```
- **Windows:**
  Download from [python.org](https://www.python.org/downloads/) and install

- **Raspberry Pi:**
  ```bash
  sudo apt update
  sudo apt install python3 python3-pip python3-venv
  ```

### 2. System Libraries

**Ubuntu/Debian/Raspberry Pi:**
```bash
sudo apt update
sudo apt install libgl1 libglib2.0-0 python3-dev build-essential
```

**macOS:**
```bash
brew install opencv
```

**Windows:**
- Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/) (required for some Python packages)

---

## Environment Setup

### 1. Clone the Repository
```bash
git clone https://github.com/NethmiThathsarani20/face-recognition-attendance-marking-system.git
cd face-recognition-attendance-marking-system
```

### 2. Create Virtual Environment
It's highly recommended to use a virtual environment to avoid dependency conflicts.

**On Linux/macOS/Raspberry Pi:**
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

**On Windows:**
```cmd
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate
```

After activation, your terminal prompt should show `(venv)` prefix.

### 3. Upgrade pip
```bash
python -m pip install --upgrade pip
```

---

## Installation Steps

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

This will install:
- `insightface` - Production-grade face recognition
- `opencv-python` - Computer vision and camera handling
- `flask` - Web framework
- `numpy` - Numerical operations
- `onnxruntime` - InsightFace runtime
- `Pillow` - Image processing
- `tensorflow` - For optional CNN training
- `scikit-learn` - Embedding classifier
- `matplotlib` - For training visualizations

**Note:** Installation may take 5-15 minutes depending on your internet speed and system.

### 2. Run Setup Script (Optional)
The setup script creates necessary directories and validates the installation:
```bash
python setup.py
```

### 3. Verify Installation
```bash
python verify_requirements.py
```

This script checks if all required dependencies are correctly installed.

---

## Running the Application

### 1. Start the Web Application
```bash
python run.py
```

The application will start on `http://localhost:3000` by default.

### 2. Access the Web Interface
Open your web browser and navigate to:
```
http://localhost:3000
```

**For remote access (Raspberry Pi):**
```
http://<raspberry-pi-ip>:3000
```
Example: `http://10.74.63.231:3000`

### 3. Stop the Application
Press `Ctrl+C` in the terminal to stop the server.

---

## Hardware Setup

### ESP32-CAM Setup

1. **Flash ESP32-CAM Firmware**
   - Open Arduino IDE
   - Navigate to `esp32-camera/` directory
   - Open `esp32-camera.ino`
   - Configure WiFi credentials in the code
   - Flash to ESP32-CAM board

2. **Get ESP32-CAM IP Address**
   - After flashing, open Serial Monitor
   - Note the IP address displayed
   - Camera stream URL: `http://<esp32-ip>:81/stream`
   - Example: `http://10.74.63.131:81/stream`

3. **Test Camera Stream**
   - Open camera URL in web browser to verify stream

### Raspberry Pi Setup (Edge Device)

1. **Install Raspberry Pi OS**
   - Use Raspberry Pi Imager
   - Install Raspberry Pi OS (64-bit recommended)
   - Enable SSH during setup

2. **SSH into Raspberry Pi**
   ```bash
   ssh pi@<raspberry-pi-ip>
   # Default password: raspberry (change after first login)
   ```

3. **Update System**
   ```bash
   sudo apt update && sudo apt upgrade -y
   ```

4. **Install Required Packages**
   ```bash
   sudo apt install python3 python3-pip python3-venv git libgl1 libglib2.0-0
   ```

5. **Clone and Setup Project**
   ```bash
   git clone https://github.com/NethmiThathsarani20/face-recognition-attendance-marking-system.git
   cd face-recognition-attendance-marking-system
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

6. **Run Application on Raspberry Pi**
   ```bash
   python run.py
   ```

7. **Auto-start on Boot (Optional)**
   Create systemd service:
   ```bash
   sudo nano /etc/systemd/system/attendance-system.service
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
   
   Enable and start:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable attendance-system.service
   sudo systemctl start attendance-system.service
   ```

### Finding Device IP Address

Use the included script to find device IP by MAC address:
```bash
python ip.py 80:f3:da:62:14:c0
```

---

## Usage Guide

### 1. Add Users

**Via Web Interface:**
1. Navigate to "Add User" page
2. Enter user name
3. Upload multiple images (3-5 images recommended for better accuracy)
4. System automatically processes and stores face embeddings

**Via Command Line:**
```bash
# Organize images in database folder
mkdir -p database/John_Doe
# Copy user images to the folder
cp /path/to/images/*.jpg database/John_Doe/
```

### 2. Mark Attendance

**Using Local USB Camera:**
1. Go to "Mark Attendance" page
2. Select camera index (usually 0 for built-in, 1 for external)
3. Click "Capture" or "Mark Attendance"

**Using IP Camera:**
1. Enter camera URL in the IP camera field
   - ESP32-CAM: `http://10.74.63.131:81/stream`
   - Android IP Webcam: `http://192.168.1.100:8080/video`
   - Generic MJPEG: `http://IP:PORT/video`
   - Generic RTSP: `rtsp://IP:PORT/stream`
2. Click "Mark Attendance"

**Using Image Upload:**
1. Click "Upload Image" button
2. Select image file
3. Click "Mark Attendance"

### 3. View Attendance Records

**Via Web Interface:**
- Navigate to "View Attendance" page
- Records are organized by date

**Via File System:**
- JSON files in `attendance_records/` directory
- Format: `attendance_YYYY-MM-DD.json`

### 4. Optional Model Training

**Train CNN Model (Optional):**
```bash
python train.py --epochs 30 --validation-split 0.2
```

**Switch Between Models:**
- InsightFace (default): `http://localhost:3000/switch/insightface`
- CNN: `http://localhost:3000/switch/cnn`
- Embedding Classifier: `http://localhost:3000/switch/embedding`
- Custom Embedding: `http://localhost:3000/switch/custom_embedding`

**Check Model Status:**
```
http://localhost:3000/model_status
```

### 5. Cloud Training with GitHub Actions

**Automatic Training on Raspberry Pi:**

1. **Add images to database on Raspberry Pi:**
   ```bash
   # Navigate to project directory
   cd face-recognition-attendance-marking-system
   
   # Add user images to database folder
   mkdir -p database/New_User
   cp /path/to/images/*.jpg database/New_User/
   ```

2. **Sync to GitHub (triggers cloud training):**
   ```bash
   # Make script executable (first time only)
   chmod +x scripts/edge_sync.sh
   
   # Run sync script
   ./scripts/edge_sync.sh "Add images for New_User"
   ```

3. **GitHub Actions automatically:**
   - Detects database changes
   - Trains models in the cloud
   - Pushes trained models back to repository

4. **Pull trained models on Raspberry Pi:**
   ```bash
   git pull
   ```

**Manual Training Trigger:**
- Go to repository's "Actions" tab on GitHub
- Click "Train models on dataset updates"
- Click "Run workflow"

---

## Troubleshooting

### Common Issues

#### 1. Camera Not Working
**Problem:** Camera index not found or IP camera not connecting

**Solutions:**
- Try different camera indices (0, 1, 2)
- Check camera permissions:
  ```bash
  # Linux
  sudo usermod -a -G video $USER
  ```
- For IP cameras, verify:
  - URL format is correct
  - Camera is on same network
  - No firewall blocking
  - Try accessing URL in browser first

#### 2. Installation Errors

**Problem:** `pip install` fails for certain packages

**Solutions:**
```bash
# Upgrade pip first
pip install --upgrade pip setuptools wheel

# Install build tools (Ubuntu/Debian)
sudo apt install python3-dev build-essential

# Windows: Install Visual Studio Build Tools
# macOS: Install Xcode Command Line Tools
xcode-select --install
```

#### 3. Face Not Recognized

**Problem:** System doesn't recognize known faces

**Solutions:**
- Ensure good lighting conditions
- Add more training images (5-10 images per person)
- Check if face is clearly visible (no masks, good angle)
- Adjust similarity threshold in `src/config.py`:
  ```python
  SIMILARITY_THRESHOLD = 0.3  # Lower = more lenient
  ```

#### 4. Out of Memory Errors

**Problem:** System crashes or freezes during training

**Solutions:**
- Reduce batch size in training scripts
- Close other applications
- Use cloud training via GitHub Actions
- Consider using lighter model

#### 5. IP Camera Connection Issues

**Problem:** Cannot connect to ESP32-CAM or IP camera

**Solutions:**
```bash
# Test camera URL directly
curl http://10.74.63.131:81/stream

# Check network connectivity
ping 10.74.63.131

# Verify camera is streaming (open in browser)
# Check firewall settings
```

#### 6. Permission Denied Errors

**Problem:** Cannot access files or directories

**Solutions:**
```bash
# Linux/macOS: Fix permissions
chmod -R 755 face-recognition-attendance-marking-system
chmod +x scripts/edge_sync.sh

# Create directories manually
mkdir -p database embeddings attendance_records
```

#### 7. Module Not Found Errors

**Problem:** Import errors even after installation

**Solutions:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Reinstall requirements
pip install -r requirements.txt --force-reinstall

# Add project to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

---

## All Functions and Features

### Core Functions

1. **Face Detection and Recognition**
   - Automatic face detection using InsightFace
   - High-accuracy face recognition
   - Support for multiple faces in single image

2. **User Management**
   - Add new users with multiple images
   - View all registered users
   - Delete users
   - Update user images

3. **Attendance Tracking**
   - Mark attendance with timestamp
   - Daily attendance logs (JSON format)
   - View attendance history
   - Export attendance records

4. **Camera Support**
   - Local USB cameras
   - IP cameras (MJPEG/RTSP)
   - ESP32-CAM modules
   - Image file upload
   - Video file processing

5. **Model Training (Optional)**
   - CNN model training
   - Embedding classifier training
   - Custom embedding model
   - Cloud training via GitHub Actions
   - Model switching at runtime

6. **Web Interface**
   - Dashboard with navigation
   - Real-time camera preview
   - User-friendly forms
   - Responsive design
   - Live attendance updates

### Command-Line Tools

```bash
# Run main application
python run.py

# Run demo (test system)
python demo.py

# Train models
python train.py --epochs 30 --validation-split 0.2

# Run tests
python tests/run_tests.py

# Setup system
python setup.py

# Verify installation
python verify_requirements.py

# Find device by MAC address
python ip.py <MAC_ADDRESS>

# Sync database to GitHub (Raspberry Pi)
./scripts/edge_sync.sh "Commit message"
```

### Development Commands

```bash
# Activate virtual environment
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Run linting
make lint

# Format code
make format

# Run type checking
make type-check

# Run all tests
make test

# Run security checks
make security

# Clean temporary files
make clean
```

---

## Quick Reference

### Network Configuration Examples

```bash
# SSH to Raspberry Pi
ssh pi@10.74.63.231
# Password: raspberry (or your custom password)

# Navigate to project
cd face-recognition-attendance-marking-system

# Activate environment
source venv/bin/activate

# Start application
python run.py
```

### Camera URLs
- **ESP32-CAM**: `http://10.74.63.131:81/stream`
- **Raspberry Pi Web App**: `http://10.74.63.231:3000`
- **Android IP Webcam**: `http://192.168.1.100:8080/video`
- **Generic MJPEG**: `http://<IP>:<PORT>/video`
- **RTSP Stream**: `rtsp://<IP>:<PORT>/stream`

### Important Directories
- `database/` - User face images
- `embeddings/` - Face embeddings storage
- `attendance_records/` - Daily attendance JSON files
- `cnn_models/` - Trained CNN models
- `embedding_models/` - Embedding classifier models
- `custom_embedding_models/` - Custom embedding models
- `templates/` - HTML templates
- `static/` - CSS, JavaScript files

---

## Additional Resources

- **Project Repository**: https://github.com/NethmiThathsarani20/face-recognition-attendance-marking-system
- **InsightFace Documentation**: https://insightface.ai/
- **Flask Documentation**: https://flask.palletsprojects.com/
- **OpenCV Documentation**: https://docs.opencv.org/

For issues or questions, please open an issue on the GitHub repository.