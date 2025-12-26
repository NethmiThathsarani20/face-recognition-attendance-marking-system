# Complete Requirements and Steps to Run in Visual Studio Code

**Everything you need to run the Face Recognition Attendance System UI in VS Code**

---

## 📋 Table of Contents
1. [System Requirements](#system-requirements)
2. [Software Prerequisites](#software-prerequisites)
3. [Installation Steps](#installation-steps)
4. [VS Code Configuration](#vs-code-configuration)
5. [Running the Application](#running-the-application)
6. [Folder Structure](#folder-structure)
7. [Troubleshooting](#troubleshooting)

---

## 🖥️ System Requirements

### Minimum Requirements
- **Operating System**: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 18.04+)
- **Processor**: Intel Core i3 or equivalent
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: 2GB free space
- **Internet**: Required for initial setup and dependency installation
- **Camera**: USB webcam, IP camera, or ESP32-CAM (for attendance marking)

### Recommended Requirements
- **Processor**: Intel Core i5 or better
- **RAM**: 8GB or more
- **Storage**: 5GB+ free space (for database and models)
- **Display**: 1920x1080 or higher resolution

---

## 📦 Software Prerequisites

### 1. Python 3.8 or Higher (3.12 Recommended)

**Windows:**
1. Download from [python.org](https://www.python.org/downloads/)
2. Run installer
3. ✅ **IMPORTANT: Check "Add Python to PATH"**
4. Complete installation
5. Verify: Open Command Prompt and run `python --version`

**macOS:**
```bash
# Using Homebrew (install Homebrew first from brew.sh)
brew install python@3.12

# Verify
python3 --version
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv python3-dev build-essential

# Verify
python3 --version
```

### 2. Visual Studio Code

1. Download from [code.visualstudio.com](https://code.visualstudio.com/)
2. Install for your operating system:
   - **Windows**: Run the `.exe` installer
   - **macOS**: Drag to Applications folder
   - **Linux**: Use `.deb` or `.rpm` package, or snap: `sudo snap install code --classic`
3. Launch VS Code

### 3. Git Version Control

**Windows:**
- Download from [git-scm.com](https://git-scm.com/downloads)
- Run installer with default settings

**macOS:**
```bash
# Using Homebrew
brew install git

# Or install Xcode Command Line Tools
xcode-select --install
```

**Linux:**
```bash
sudo apt install git
```

**Verify:**
```bash
git --version
```

### 4. System Libraries (Linux Only)

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install libgl1 libglib2.0-0 python3-dev build-essential
```

**Raspberry Pi:**
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv libgl1 libglib2.0-0 python3-dev build-essential
```

### 5. Visual Studio Build Tools (Windows Only - for some packages)

Download and install [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022)

Select "Desktop development with C++" workload during installation.

---

## 🚀 Installation Steps

### Step 1: Clone the Repository

Open a terminal (Command Prompt, PowerShell, or Terminal app):

```bash
# Navigate to desired directory
cd ~/Documents  # macOS/Linux
# or
cd C:\Users\YourName\Documents  # Windows

# Clone repository
git clone https://github.com/NethmiThathsarani20/face-recognition-attendance-marking-system.git

# Navigate into project
cd face-recognition-attendance-marking-system
```

### Step 2: Open Project in VS Code

**Option A: From Terminal**
```bash
code .
```

**Option B: From VS Code**
1. Launch VS Code
2. File → Open Folder
3. Navigate to `face-recognition-attendance-marking-system`
4. Click "Select Folder" (Windows) or "Open" (macOS)

### Step 3: Install Required VS Code Extensions

VS Code will automatically suggest recommended extensions. Click "Install All" when prompted.

**Or install manually:**

1. Click Extensions icon (Ctrl+Shift+X / Cmd+Shift+X)
2. Search and install each:
   - **Python** (Microsoft) - ms-python.python
   - **Pylance** (Microsoft) - ms-python.vscode-pylance
   - **Python Debugger** (Microsoft) - ms-python.debugpy
   - **Ruff** (Astral Software) - charliermarsh.ruff

**Optional but Recommended:**
   - **GitHub Copilot** - AI assistance
   - **GitLens** - Enhanced Git features
   - **Code Spell Checker** - Spelling in code

### Step 4: Create Virtual Environment

Open VS Code Terminal (`` Ctrl+` `` or View → Terminal):

**Windows (Command Prompt):**
```cmd
python -m venv venv
```

**Windows (PowerShell):**
```powershell
python -m venv venv
```

**macOS/Linux:**
```bash
python3 -m venv venv
```

**Wait for creation to complete** (shows "created virtual environment" message).

### Step 5: Activate Virtual Environment

**Windows (Command Prompt):**
```cmd
venv\Scripts\activate
```

**Windows (PowerShell):**
```powershell
venv\Scripts\Activate.ps1
```

If PowerShell shows execution policy error:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

**Confirmation:** Terminal prompt should show `(venv)` prefix.

### Step 6: Select Python Interpreter in VS Code

1. Press `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (macOS)
2. Type: **"Python: Select Interpreter"**
3. Select the interpreter from **`./venv/bin/python`** or **`.\venv\Scripts\python.exe`**

**Alternative:** Click on Python version in bottom-right status bar.

### Step 7: Install Python Dependencies

With virtual environment activated in terminal:

```bash
# Upgrade pip first (recommended)
python -m pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt
```

**This will install:**
- insightface - Face recognition engine
- opencv-python - Computer vision
- flask - Web framework  
- numpy - Numerical computing
- tensorflow - Deep learning (for optional training)
- scikit-learn - Machine learning utilities
- matplotlib - Visualization
- reportlab - PDF generation
- openpyxl - Excel export
- pandas - Data manipulation
- And more...

**Installation time:** 5-15 minutes depending on internet speed.

**Note:** You may see some warnings - these are usually safe to ignore if installation completes successfully.

### Step 8: Verify Installation

Run verification script:
```bash
python verify_requirements.py
```

This checks if all dependencies are correctly installed.

**Also run structure verification:**
```bash
python verify_structure.py
```

This verifies all folders and files are in place.

---

## ⚙️ VS Code Configuration

All VS Code configurations are pre-configured in the `.vscode/` folder:

### Available Configurations

**1. Editor Settings (`.vscode/settings.json`):**
- Python interpreter path
- Auto-formatting on save
- Linting with Ruff
- File associations
- Terminal environment

**2. Debug Configurations (`.vscode/launch.json`):**
- **Python: Run Application** - Debug main app (F5)
- **Python: Run Demo** - Debug demo script
- **Python: Train Models** - Debug training
- **Python: Run Tests** - Debug tests
- **Python: Current File** - Debug any file
- **Python: Flask Debug** - Flask with template debugging

**3. Tasks (`.vscode/tasks.json`):**
Access via `Ctrl+Shift+P` → "Tasks: Run Task"
- Run Application
- Install Dependencies
- Run Tests
- Run Linter (Ruff)
- Format Code
- Train Models
- Setup Development Environment
- Clean Project

**4. Recommended Extensions (`.vscode/extensions.json`):**
VS Code automatically recommends these when you open the project.

---

## ▶️ Running the Application

### Method 1: Using Run Button (Easiest)

1. Open `run.py` in VS Code editor
2. Click the **▷ Run Python File** button (top-right corner)
3. Application starts in terminal

### Method 2: Using Debug Mode (Recommended for Development)

1. Press **F5**
2. Select **"Python: Run Application"** from dropdown
3. Application starts with debugger attached
4. Set breakpoints by clicking left of line numbers

### Method 3: Using Terminal

In VS Code Terminal (with venv activated):
```bash
python run.py
```

### Method 4: Using VS Code Task

1. Press `Ctrl+Shift+P` (or `Cmd+Shift+P`)
2. Type **"Tasks: Run Task"**
3. Select **"Run Application"**

### Expected Output

```
Starting Simple Attendance System...
Open your browser and go to: http://localhost:3000
Press Ctrl+C to stop the server
```

### Access the Web UI

Open your browser and navigate to:
```
http://localhost:3000
```

**For remote access (e.g., Raspberry Pi):**
```
http://<device-ip>:3000
```

**You should see the Face Recognition Attendance System dashboard! 🎉**

### Stop the Application

- Press **Ctrl+C** in the terminal
- Or click the trash can icon in terminal panel
- Or click the red square "Stop" button (if debugging)

---

## 📂 Folder Structure

The project is organized as follows:

```
face-recognition-attendance-marking-system/
├── .vscode/                  # VS Code configuration (auto-configured) ⭐
│   ├── settings.json         # Editor settings
│   ├── launch.json           # Debug configurations
│   ├── tasks.json            # Build/run tasks
│   └── extensions.json       # Recommended extensions
│
├── src/                      # Python source code
│   ├── config.py             # Settings (port, thresholds, etc.)
│   ├── face_manager.py       # Face recognition logic
│   ├── attendance_system.py  # Attendance management
│   ├── web_app.py            # Flask web application
│   └── exceptions.py         # Custom exceptions
│
├── templates/                # HTML templates (Jinja2)
│   ├── index.html            # Main dashboard
│   ├── add_user.html         # Add user page
│   └── cnn_training.html     # Training page
│
├── static/                   # CSS, JavaScript
│   └── style.css             # Application styles
│
├── database/                 # User face images (auto-created)
│   └── [User_Name]/          # One folder per user
│       └── *.jpg             # User images
│
├── embeddings/               # Face embeddings (auto-created)
│   └── *.pkl                 # Pickle files with embeddings
│
├── attendance_records/       # Daily attendance logs (auto-created)
│   └── attendance_*.json     # JSON files by date
│
├── cnn_models/              # Trained models (auto-created if trained)
├── embedding_models/         # Embedding classifiers (auto-created)
├── custom_embedding_models/  # Custom models (auto-created)
│
├── tests/                    # Test suite
│   ├── run_tests.py          # Test runner
│   └── test_*.py             # Test files
│
├── docs/                     # Documentation
├── scripts/                  # Helper scripts
├── esp32-camera/            # ESP32-CAM firmware
│
├── run.py                   # Main entry point ⭐
├── requirements.txt         # Dependencies
├── README.md                # Project README
├── INSTRUCTIONS.md          # Setup instructions
├── VS_CODE_SETUP.md        # VS Code guide (detailed)
├── QUICK_START_VS_CODE.md  # VS Code quick start
├── FOLDER_STRUCTURE.md     # This folder structure
└── ...
```

**See `FOLDER_STRUCTURE.md` for complete details.**

---

## 🎯 Using the Application

### 1. Add Users

1. Navigate to **"Add User"** page
2. Enter user name
3. Upload 3-5 images per person (different angles/lighting)
4. System automatically:
   - Detects faces
   - Generates embeddings
   - Stores in database

### 2. Mark Attendance

**Using Local Camera:**
1. Go to "Mark Attendance" page
2. Select camera index (0, 1, 2...)
3. Click "Mark Attendance"
4. Face detected → Attendance marked

**Using IP Camera:**
1. Enter camera URL:
   - ESP32-CAM: `http://192.168.1.100:81/stream`
   - Android IP Webcam: `http://192.168.1.100:8080/video`
   - Generic MJPEG: `http://IP:PORT/video`
2. Click "Mark Attendance"

**Using Image Upload:**
1. Click "Upload Image"
2. Select image file
3. Click "Mark Attendance"

### 3. View Attendance Records

- Navigate to "View Attendance" page
- See daily attendance records
- Export to PDF or Excel

### 4. Optional Model Training

Train custom models (optional):
```bash
python train.py --epochs 30 --validation-split 0.2
```

Or use VS Code task: `Ctrl+Shift+P` → "Tasks: Run Task" → "Train Models"

---

## 🐛 Troubleshooting

### Issue: Virtual Environment Not Activating

**Windows PowerShell:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
venv\Scripts\Activate.ps1
```

**Windows Command Prompt:**
```cmd
venv\Scripts\activate.bat
```

### Issue: Python Not Found

1. `Ctrl+Shift+P` → "Python: Select Interpreter"
2. If venv not listed, click "Enter interpreter path..."
3. Browse to:
   - Windows: `venv\Scripts\python.exe`
   - Mac/Linux: `venv/bin/python`

### Issue: Import Errors

```bash
# Ensure venv is activated (look for (venv) prefix)
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Issue: Port 3000 Already in Use

**Find and kill process:**

**Windows:**
```cmd
netstat -ano | findstr :3000
taskkill /PID <PID> /F
```

**Mac/Linux:**
```bash
lsof -ti:3000 | xargs kill -9
```

**Or change port in `src/config.py`:**
```python
WEB_PORT = 5000  # Change from 3000
```

### Issue: Camera Not Working

**Local Camera:**
- Try different indices: 0, 1, 2
- Check camera permissions
- Test: `python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"`

**Linux camera permissions:**
```bash
sudo usermod -a -G video $USER
# Then logout and login
```

**IP Camera:**
- Verify URL in browser first
- Check network connectivity: `ping <camera-ip>`
- Ensure no firewall blocking

### Issue: Face Not Recognized

**Solutions:**
- Ensure good lighting
- Add more training images (5-10 per person)
- Face should be front-facing and clear
- Adjust threshold in `src/config.py`:
  ```python
  SIMILARITY_THRESHOLD = 0.3  # Lower = more lenient
  ```

### Issue: Installation Errors

**Windows:**
- Install Visual Studio Build Tools
- Ensure "Add Python to PATH" was checked

**All Platforms:**
```bash
# Upgrade pip first
python -m pip install --upgrade pip setuptools wheel

# Then try installing again
pip install -r requirements.txt
```

---

## 📚 Additional Resources

### Documentation Files

- **README.md** - Project overview and features
- **INSTRUCTIONS.md** - Detailed installation guide
- **VS_CODE_SETUP.md** - Comprehensive VS Code guide
- **QUICK_START_VS_CODE.md** - Quick VS Code setup
- **FOLDER_STRUCTURE.md** - Complete folder organization
- **docs/TECHNICAL_OVERVIEW.md** - Technical details
- **docs/MODEL_TRAINING.md** - Model training guide

### Quick Commands

```bash
# Start application
python run.py

# Run demo
python demo.py

# Run tests
python tests/run_tests.py

# Train models
python train.py --epochs 30

# Verify installation
python verify_requirements.py

# Verify structure
python verify_structure.py

# Lint code
make lint

# Format code
make format
```

### VS Code Shortcuts

| Action | Windows/Linux | Mac |
|--------|--------------|-----|
| Command Palette | `Ctrl+Shift+P` | `Cmd+Shift+P` |
| Quick Open | `Ctrl+P` | `Cmd+P` |
| Terminal | `` Ctrl+` `` | `` Cmd+` `` |
| Debug | `F5` | `F5` |
| Run File | Click ▷ | Click ▷ |
| Format Code | `Shift+Alt+F` | `Shift+Option+F` |
| Find | `Ctrl+F` | `Cmd+F` |
| Save All | `Ctrl+K S` | `Cmd+K S` |

---

## ✅ Verification Checklist

Before running, ensure:

- [ ] Python 3.8+ installed and in PATH
- [ ] VS Code installed
- [ ] Git installed
- [ ] Repository cloned
- [ ] Project opened in VS Code
- [ ] VS Code Python extension installed
- [ ] Virtual environment created (`venv` folder exists)
- [ ] Virtual environment activated (`(venv)` shown in terminal)
- [ ] Python interpreter selected in VS Code
- [ ] Dependencies installed (`pip install -r requirements.txt` completed)
- [ ] `verify_requirements.py` passes
- [ ] `verify_structure.py` passes
- [ ] All required folders exist (src, templates, static, etc.)
- [ ] Camera available (if needed for attendance)

---

## 🎉 Success!

If all steps completed successfully:

1. ✅ Application runs: `python run.py`
2. ✅ Browser opens: `http://localhost:3000`
3. ✅ UI loads with dashboard
4. ✅ Can add users
5. ✅ Can mark attendance
6. ✅ Can view records

**You're ready to use the Face Recognition Attendance System!**

---

## 📞 Support

**Need help?**
- Check troubleshooting section above
- See `VS_CODE_SETUP.md` for detailed guide
- See `INSTRUCTIONS.md` for setup help
- Open issue: https://github.com/NethmiThathsarani20/face-recognition-attendance-marking-system/issues

---

**Happy Coding! 🚀**
