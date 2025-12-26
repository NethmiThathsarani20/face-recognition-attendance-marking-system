# Project Folder Structure - Face Recognition Attendance System

This document explains the complete folder structure and the purpose of each directory and key file in the project.

---

## Root Directory Structure

```
face-recognition-attendance-marking-system/
├── .github/                  # GitHub-specific files
├── .vscode/                  # VS Code configuration (NEW)
├── src/                      # Python source code
├── templates/                # HTML templates for web UI
├── static/                   # CSS, JavaScript files
├── database/                 # User face images (auto-created)
├── embeddings/               # Face embeddings storage (auto-created)
├── attendance_records/       # Daily attendance logs (auto-created)
├── cnn_models/              # Trained CNN models (auto-created)
├── embedding_models/         # Embedding classifier models (auto-created)
├── custom_embedding_models/  # Custom embedding models (auto-created)
├── tests/                    # Test suite
├── docs/                     # Documentation
├── scripts/                  # Helper scripts
├── esp32-camera/            # ESP32-CAM firmware
├── run.py                   # Main application entry point
├── requirements.txt         # Python dependencies
└── ... (configuration files)
```

---

## Detailed Folder Descriptions

### 📁 `.github/` - GitHub Configuration

Contains GitHub Actions workflows and configuration.

```
.github/
└── workflows/
    └── train.yml            # CI/CD workflow for automatic model training
```

**Purpose:**
- Automates model training in the cloud when database changes are pushed
- Runs on GitHub Actions runners to avoid resource constraints on Raspberry Pi
- Triggered on push to `database/**`, `database1/**`, or `database2/**`

**Key Features:**
- Installs dependencies
- Runs `train.py` with specified parameters
- Commits and pushes trained models back to repository

---

### 📁 `.vscode/` - Visual Studio Code Configuration (NEW)

Complete VS Code workspace configuration for optimal development experience.

```
.vscode/
├── settings.json            # Editor settings, Python config, formatters
├── launch.json              # Debug configurations
├── tasks.json               # Build and run tasks
└── extensions.json          # Recommended VS Code extensions
```

**Files Explained:**

#### `settings.json`
- Python interpreter path pointing to virtual environment
- Auto-formatting on save (Ruff)
- Linting configuration
- File associations
- Editor rulers at 88 and 120 characters
- Excluded file patterns (__pycache__, etc.)

#### `launch.json`
Pre-configured debug modes:
- **Python: Run Application** - Debug main app
- **Python: Run Demo** - Debug demo script
- **Python: Train Models** - Debug training with parameters
- **Python: Run Tests** - Debug test suite
- **Python: Current File** - Debug any file
- **Python: Flask Debug** - Debug Flask with Jinja templates

#### `tasks.json`
Quick tasks accessible via `Ctrl+Shift+P`:
- Run Application
- Install Dependencies
- Run Tests
- Run Linter (Ruff)
- Format Code (Ruff)
- Train Models
- Setup Development Environment
- Clean Project

#### `extensions.json`
Recommended extensions:
- Python (Microsoft)
- Pylance
- Ruff
- GitLens
- GitHub Copilot
- Code Spell Checker
- And more...

**Benefits:**
- ✅ One-click debugging
- ✅ Integrated tasks
- ✅ Auto-formatting on save
- ✅ IntelliSense for Python
- ✅ Consistent development environment

---

### 📁 `src/` - Source Code

Core Python modules for the application.

```
src/
├── config.py                # Configuration settings
├── face_manager.py          # InsightFace integration
├── attendance_system.py     # Attendance logic
├── web_app.py              # Flask web application
├── exceptions.py            # Custom exception framework
├── cnn_trainer.py          # CNN training logic (optional)
├── embedding_trainer.py     # Embedding classifier trainer
└── custom_embedding_trainer.py  # Custom embedding trainer
```

**Key Files:**

#### `config.py`
Global configuration:
```python
FACE_MODEL_NAME = "buffalo_l"        # InsightFace model
SIMILARITY_THRESHOLD = 0.4           # Recognition threshold
WEB_PORT = 3000                      # Server port
DATABASE_DIR = "database"            # User images directory
```

#### `face_manager.py`
- InsightFace integration
- Face detection and alignment
- Embedding generation
- Face recognition matching

#### `attendance_system.py`
- User management
- Attendance marking
- Attendance record management
- Database operations

#### `web_app.py`
- Flask web server
- Route handlers
- UI rendering
- API endpoints

#### `exceptions.py`
Custom exceptions:
- `FaceDetectionError`
- `FaceRecognitionError`
- `UserNotFoundError`
- `CameraError`
- And more...

---

### 📁 `templates/` - HTML Templates

Web interface HTML files using Jinja2 templating.

```
templates/
├── index.html              # Main dashboard
├── add_user.html           # Add user page
├── cnn_training.html       # Model training page
├── index_old.html          # Old version (backup)
├── add_user_old.html       # Old version (backup)
└── cnn_training_old.html   # Old version (backup)
```

**Active Templates:**
- `index.html` - Dashboard with navigation, modern gradient design
- `add_user.html` - User registration with multi-image upload
- `cnn_training.html` - Optional model training interface

**Features:**
- Responsive design
- Modern gradient theme
- Clean navigation
- Form validation
- Real-time feedback

---

### 📁 `static/` - Static Assets

CSS, JavaScript, and other static files.

```
static/
└── style.css               # Application styles
```

**style.css:**
- Modern gradient theme
- Responsive layout
- Button styles
- Form styling
- Card layouts
- Navigation styles

---

### 📁 `database/` - User Face Images

Stores face images for each registered user. **Auto-created** when first user is added.

```
database/
├── John_Doe/
│   ├── John_Doe_0001.jpg
│   ├── John_Doe_0002.jpg
│   ├── John_Doe_0003.jpg
│   └── ...
├── Jane_Smith/
│   ├── Jane_Smith_0001.jpg
│   └── ...
└── [Other Users]/
    └── ...
```

**Structure:**
- One folder per user (named after the user)
- Multiple images per user for better accuracy
- Images automatically saved when attendance is marked
- Timestamped filenames

**Best Practices:**
- Store 3-10 images per user
- Vary lighting and angles
- Clear, front-facing photos work best

**Note:** This folder is tracked in Git and synced to GitHub for cloud training.

---

### 📁 `embeddings/` - Face Embeddings

Stores computed face embeddings in pickle format. **Auto-created** when first embedding is generated.

```
embeddings/
├── John_Doe.pkl
├── Jane_Smith.pkl
└── ...
```

**Contents:**
- One `.pkl` file per user
- Contains numpy arrays of face embeddings
- Generated from user images using InsightFace
- Used for fast face recognition matching

**Note:** These are binary files generated from images in `database/`.

---

### 📁 `attendance_records/` - Attendance Logs

Daily attendance records in JSON format. **Auto-created** when first attendance is marked.

```
attendance_records/
├── attendance_2025-12-26.json
├── attendance_2025-12-27.json
└── ...
```

**Format Example:**
```json
{
  "2025-12-26": [
    {
      "name": "John_Doe",
      "time": "09:15:32",
      "confidence": 0.85
    },
    {
      "name": "Jane_Smith",
      "time": "09:18:45",
      "confidence": 0.92
    }
  ]
}
```

**Features:**
- One file per day
- Timestamped entries
- Includes confidence scores
- Can export to PDF/Excel via web UI

---

### 📁 `cnn_models/` - CNN Model Artifacts

Stores trained CNN models and related files. **Auto-created** during CNN training.

```
cnn_models/
├── custom_face_model.keras          # Trained CNN model
├── label_encoder.pkl                # Label encoder for classes
├── training_log.json                # Training metrics
├── cnn_confusion_matrix.png         # Confusion matrix
├── cnn_confusion_matrix_normalized.png
├── cnn_confidence_curve.png
├── cnn_precision_recall_curve.png
└── cnn_precision_confidence_curve.png
```

**Files:**
- **Model File:** `.keras` format (TensorFlow/Keras)
- **Label Encoder:** Maps class indices to user names
- **Training Log:** JSON with accuracy, loss, metrics
- **Visualizations:** Performance charts and matrices

**Note:** Optional - only created if CNN training is performed.

---

### 📁 `embedding_models/` - Embedding Classifier Models

Stores embedding-based classifier (InsightFace + Logistic Regression). **Auto-created** during embedding training.

```
embedding_models/
├── embedding_classifier.pkl         # Trained classifier
├── label_encoder.pkl                # Label encoder
├── training_log.json                # Training metrics
└── [visualization files]
```

**Purpose:**
- Alternative to CNN approach
- Uses InsightFace embeddings as features
- Logistic Regression for classification
- Typically better accuracy than lightweight CNN

---

### 📁 `custom_embedding_models/` - Custom Embedding Models

Experimental custom embedding approach. **Auto-created** during custom embedding training.

```
custom_embedding_models/
├── custom_embedding_model.keras     # Custom embedding network
├── class_centroids.npy             # Class centroids
├── label_encoder.pkl               # Label encoder
├── training_log.json               # Training metrics
└── [visualization files]
```

**Note:** Experimental feature for research purposes.

---

### 📁 `tests/` - Test Suite

Comprehensive test suite using pytest.

```
tests/
├── run_tests.py                 # Test runner
├── test_face_manager.py         # Face manager tests
├── test_attendance_system.py    # Attendance system tests
├── test_cnn_trainer.py         # CNN trainer tests
├── test_web_app.py             # Web app tests
├── test_auth.py                # Authentication tests
├── test_config.py              # Configuration tests
├── test_integration.py         # Integration tests
└── test_ip_camera.py           # IP camera tests
```

**Run Tests:**
```bash
python tests/run_tests.py
# Or
make test
# Or
pytest tests/
```

---

### 📁 `docs/` - Documentation

Additional documentation files.

```
docs/
├── MODEL_TRAINING.md            # Model training guide
├── TECHNICAL_OVERVIEW.md        # Technical details
├── STRUCTURE.md                 # Project structure
├── PROGRESS.md                  # Development progress
└── Research_Thesis_Template_*.pdf  # Research paper template
```

---

### 📁 `scripts/` - Helper Scripts

Utility scripts for maintenance and operations.

```
scripts/
├── edge_sync.sh                 # Sync database to GitHub (Raspberry Pi)
└── generate_model_comparison.py # Model comparison visualization
```

**edge_sync.sh:**
- Commits database changes
- Pushes to GitHub
- Triggers cloud training
- Usage: `./scripts/edge_sync.sh "Add new user images"`

---

### 📁 `esp32-camera/` - ESP32-CAM Firmware

Arduino firmware for ESP32-CAM module.

```
esp32-camera/
└── esp32-camera.ino             # Arduino sketch
```

**Setup:**
1. Open in Arduino IDE
2. Configure WiFi credentials
3. Flash to ESP32-CAM
4. Get IP address from Serial Monitor
5. Use stream URL: `http://<esp32-ip>:81/stream`

---

## Root Level Files

### Configuration Files

```
├── requirements.txt         # Python dependencies
├── pyproject.toml          # Ruff linter configuration
├── Makefile                # Build and development commands
├── .gitignore              # Git ignore patterns
└── setup.py               # Setup script
```

### Documentation Files

```
├── README.md                     # Main project README
├── INSTRUCTIONS.md              # Detailed setup instructions
├── VS_CODE_SETUP.md            # VS Code comprehensive guide (NEW)
├── QUICK_START_VS_CODE.md      # VS Code quick start (NEW)
├── FOLDER_STRUCTURE.md         # This file (NEW)
├── IMPLEMENTATION_SUMMARY.md    # Implementation details
└── SUMMARY.md                  # Project summary
```

### Entry Point Scripts

```
├── run.py                      # Main application entry point
├── demo.py                     # Demo script
├── train.py                    # Unified training script
├── train_cnn.py               # CNN training only
├── train_embedding.py         # Embedding classifier training
├── train_custom_embedding.py  # Custom embedding training
├── verify_requirements.py     # Dependency verification
└── ip.py                      # Find device IP by MAC address
```

---

## Directory Creation

### Auto-Created Directories

These directories are automatically created when needed:

- `database/` - Created when first user is added
- `embeddings/` - Created when first embedding is generated
- `attendance_records/` - Created when first attendance is marked
- `cnn_models/` - Created during CNN training
- `embedding_models/` - Created during embedding training
- `custom_embedding_models/` - Created during custom embedding training

### Manual Setup

These are included in the repository:

- `.vscode/` - VS Code configuration (NEW)
- `src/` - Source code
- `templates/` - HTML templates
- `static/` - Static files
- `tests/` - Test suite
- `docs/` - Documentation
- `scripts/` - Helper scripts
- `esp32-camera/` - ESP32 firmware

---

## Virtual Environment (Not Tracked)

The virtual environment directory is created locally but **not tracked** in Git:

```
venv/                          # Python virtual environment (ignored by Git)
├── bin/                       # Executables (Linux/Mac)
├── Scripts/                   # Executables (Windows)
├── lib/                       # Python libraries
└── ...
```

**Creating venv:**
```bash
python -m venv venv
source venv/bin/activate       # Linux/Mac
venv\Scripts\activate          # Windows
```

---

## Ignored Files and Directories

The following are excluded from Git (see `.gitignore`):

```
# Virtual environments
venv/
.venv/
env/

# Python cache
__pycache__/
*.pyc
.pytest_cache/
.mypy_cache/
.ruff_cache/

# IDE user-specific
.vscode/settings.json.user
.idea/

# OS files
.DS_Store
Thumbs.db

# Build artifacts
dist/
build/
*.egg-info/

# Coverage reports
htmlcov/
.coverage

# Alternate database directories
database1/
database2/
```

---

## Folder Permissions

### Linux/Mac

```bash
# Make scripts executable
chmod +x scripts/edge_sync.sh

# Ensure proper permissions
chmod -R 755 .
```

### Raspberry Pi Specific

```bash
# Add user to video group (for camera access)
sudo usermod -a -G video $USER

# Ensure proper ownership
chown -R pi:pi ~/face-recognition-attendance-marking-system
```

---

## Cloud Training Workflow

### Edge (Raspberry Pi)

1. Add user images to `database/<User_Name>/`
2. Run sync script:
   ```bash
   ./scripts/edge_sync.sh "Add images for User_Name"
   ```
3. Changes pushed to GitHub

### Cloud (GitHub Actions)

1. Workflow triggered on `database/**` changes
2. Installs dependencies
3. Runs `train.py`
4. Generates models in `cnn_models/`, `embedding_models/`, `custom_embedding_models/`
5. Commits and pushes models back

### Edge (Pull Models)

```bash
git pull
```

---

## Folder Size Estimates

Typical sizes after initial setup:

- `database/` - 50MB-500MB (depends on number of users)
- `embeddings/` - 1MB-10MB (pickle files)
- `attendance_records/` - <1MB (JSON files)
- `cnn_models/` - 5MB-50MB (if trained)
- `embedding_models/` - 1MB-10MB (if trained)
- `venv/` - 200MB-500MB (not in Git)

---

## Quick Navigation

**For Development:**
- Source code: `src/`
- Tests: `tests/`
- Configuration: `src/config.py`

**For Data:**
- User images: `database/`
- Attendance logs: `attendance_records/`
- Models: `*_models/` directories

**For Documentation:**
- Setup: `README.md`, `INSTRUCTIONS.md`
- VS Code: `VS_CODE_SETUP.md`, `QUICK_START_VS_CODE.md`
- Structure: `FOLDER_STRUCTURE.md` (this file)
- Technical: `docs/TECHNICAL_OVERVIEW.md`

**For Operations:**
- Start app: `run.py`
- Train models: `train.py`
- Sync data: `scripts/edge_sync.sh`

---

## Summary

This project is organized into clear, functional directories:

✅ **Well-structured source code** in `src/`  
✅ **Comprehensive VS Code configuration** in `.vscode/`  
✅ **Auto-created data directories** for runtime data  
✅ **Complete test suite** in `tests/`  
✅ **Extensive documentation** in multiple MD files  
✅ **Cloud training workflow** via GitHub Actions  
✅ **Clean separation** of code, data, and configuration  

All folders serve a specific purpose and are designed for scalability, maintainability, and ease of development.

---

**For more information, see:**
- `README.md` - Project overview
- `INSTRUCTIONS.md` - Setup instructions
- `VS_CODE_SETUP.md` - VS Code setup guide
- `docs/STRUCTURE.md` - Code structure details
