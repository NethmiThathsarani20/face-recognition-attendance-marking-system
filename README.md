# Simple Attendance Marking System

A super simple face recognition attendance system using InsightFace with minimal configuration and maximum use of defaults.

## Features

- **Face Recognition**: Uses InsightFace with default buffalo_l model
- **Simple Web UI**: Flask-based interface for camera/upload
- **Camera Support**: Local cameras (USB/built-in) and IP cameras (MJPEG/RTSP)
- **User Management**: Add new users with automatic face alignment
- **Attendance Marking**: Unified function for camera and upload inputs
- **JSON Output**: Simple attendance records in JSON format
- **Minimal Configuration**: Everything configurable in one place

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Run the Application
```bash
python run.py
```

### Access Web Interface
Open your browser and go to: `http://localhost:5000`

### Add New Users
1. Go to "Add User" page
2. Enter user name
3. Upload user images (system will auto-align faces)
4. Images are processed and embeddings stored

### Mark Attendance
1. **Local Camera**: Select camera index → Click "Mark Attendance"
2. **IP Camera**: Select IP camera → Enter URL → Click "Mark Attendance"
3. **Upload**: Select image → Click "Mark Attendance"

### Camera Support
- **Local Cameras**: USB cameras, built-in webcams (use camera index: 0, 1, 2, etc.)
- **IP Cameras**: Network cameras with MJPEG or RTSP streams
  - Android IP Webcam: `http://192.168.1.100:8080/video`
  - ESP32-CAM: `http://192.168.1.100:81/stream`
  - Generic MJPEG: `http://IP:PORT/video`
  - Generic RTSP: `rtsp://IP:PORT/stream`
  - With authentication: `http://user:pass@IP:PORT/video`

## Configuration

All settings are in `src/config.py`:
- Model settings (uses InsightFace defaults)
- Similarity threshold (default: 0.4)
- Paths for database and output
- Camera settings

## Project Structure

```
├── src/                    # Source code
├── templates/              # HTML templates  
├── static/                # CSS and JS files
├── database/              # User images (organized by name)
├── embeddings/            # Face embeddings (pickle files)
├── attendance_records/    # JSON attendance files
└── tests/                 # Test files
```

## Testing

### Unit Tests
```bash
python -m pytest tests/
```

### Demo Script
```bash
python demo.py
```

### IP Camera Testing
```bash
python test_ip_camera.py
```

Test your IP camera connection and functionality before using it in the main application.

## Dependencies

- insightface: Face recognition
- opencv-python: Image processing
- flask: Web framework
- numpy: Array operations
- pickle: Embedding storage

## Notes

- Uses InsightFace defaults for maximum simplicity
- Minimal code approach (< 500 lines total)
- JSON output for easy integration
- Auto face alignment for optimal recognition
- Single configuration point for easy customization
