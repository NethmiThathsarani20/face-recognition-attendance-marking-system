# Project Structure

```
computer-based-attendance-marking/
├── src/
│   ├── config.py              # Simple configuration settings
│   ├── face_manager.py        # Face detection and recognition using InsightFace
│   ├── attendance_system.py   # Main attendance logic
│   └── web_app.py            # Simple web interface using Flask
├── templates/
│   ├── index.html            # Main page with camera/upload options
│   └── add_user.html         # Add new user page
├── static/
│   └── style.css             # Basic styling
├── database/                 # Existing user images (organized by folders)
├── embeddings/              # Face embeddings stored as pickle files
├── attendance_records/      # JSON attendance output files
├── tests/                   # Test files
├── requirements.txt
├── STRUCTURE.md
├── README.md
├── PROGRESS.md
└── run.py                   # Main entry point
```

## Key Components

### 1. Face Manager (`face_manager.py`)
- Uses InsightFace FaceAnalysis with default buffalo_l model
- Handles face detection, alignment, and embedding generation
- Stores/loads embeddings using pickle files
- Simple face matching with default threshold

### 2. Attendance System (`attendance_system.py`)
- Core attendance marking logic
- Unified function for both camera and upload inputs
- JSON output for attendance records
- Minimal configuration approach

### 3. Web App (`web_app.py`)
- Simple Flask web interface
- Camera selection and image upload
- User management (add new users)
- Attendance marking interface

### 4. Configuration (`config.py`)
- Minimal settings using InsightFace defaults
- Simple paths and thresholds
- Easy to modify single configuration point

## Design Principles

1. **Use InsightFace defaults** - Minimal custom configuration
2. **Keep it simple** - Less than 500 lines total
3. **Single responsibility** - Each module has one clear purpose
4. **Easy testing** - Simple functions that can be tested independently
5. **Minimal dependencies** - Only essential packages
