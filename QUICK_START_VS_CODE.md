# Quick Start Guide for Visual Studio Code

**Get up and running in 5 minutes!**

---

## Prerequisites (5 minutes)

### 1. Install Python 3.12
- **Windows:** Download from [python.org](https://www.python.org/downloads/) → **CHECK "Add to PATH"**
- **Mac:** `brew install python@3.12`
- **Linux:** `sudo apt install python3 python3-pip python3-venv`

### 2. Install VS Code
- Download from [code.visualstudio.com](https://code.visualstudio.com/)

### 3. Install Git
- **Windows:** [git-scm.com](https://git-scm.com/)
- **Mac:** `brew install git`
- **Linux:** `sudo apt install git`

---

## Setup (3 minutes)

### Step 1: Clone and Open Project

```bash
# Clone repository
git clone https://github.com/NethmiThathsarani20/face-recognition-attendance-marking-system.git
cd face-recognition-attendance-marking-system

# Open in VS Code
code .
```

### Step 2: Install Python Extension

1. In VS Code, press `Ctrl+Shift+X` (or `Cmd+Shift+X` on Mac)
2. Search for **"Python"** by Microsoft
3. Click **Install**
4. Also install **"Ruff"** for linting

### Step 3: Create Virtual Environment

Open Terminal in VS Code (`` Ctrl+` ``):

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 4: Select Python Interpreter

1. Press `Ctrl+Shift+P` (or `Cmd+Shift+P`)
2. Type **"Python: Select Interpreter"**
3. Choose the one from `venv` folder

### Step 5: Install Dependencies

In Terminal (with venv activated):
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

⏱️ This takes 5-15 minutes depending on your internet speed.

---

## Run the Application (30 seconds)

### Method 1: Simple Run (Recommended)

1. Open `run.py` in VS Code
2. Click the **▷ Run** button (top-right corner)

### Method 2: Terminal

```bash
python run.py
```

### Method 3: Debug Mode (Press F5)

1. Press `F5`
2. Select **"Python: Run Application"**

---

## Access the UI

Open your browser and go to:
```
http://localhost:3000
```

**You should see the Face Recognition Attendance System dashboard! 🎉**

---

## Stop the Application

Press `Ctrl+C` in the terminal

---

## What You Can Do Now

### 1. Add Users
- Navigate to **"Add User"** page
- Upload 3-5 images per person
- System automatically processes faces

### 2. Mark Attendance
Choose method:
- **Local Camera:** Camera index 0, 1, 2...
- **IP Camera:** Enter URL like `http://192.168.1.100:8080/video`
- **Upload Image:** Select image file

### 3. View Attendance
- Check daily attendance records
- Export to PDF/Excel

---

## Quick Commands Reference

### VS Code Shortcuts

| Action | Windows/Linux | Mac |
|--------|--------------|-----|
| Command Palette | `Ctrl+Shift+P` | `Cmd+Shift+P` |
| Quick Open File | `Ctrl+P` | `Cmd+P` |
| Toggle Terminal | `` Ctrl+` `` | `` Cmd+` `` |
| Run File | Click ▷ or `Ctrl+F5` | Click ▷ or `Cmd+F5` |
| Debug | `F5` | `F5` |
| Format Code | `Shift+Alt+F` | `Shift+Option+F` |

### Terminal Commands

```bash
# Run application
python run.py

# Run tests
python tests/run_tests.py

# Run demo
python demo.py

# Train models
python train.py --epochs 30

# Lint code
make lint

# Format code
make format
```

---

## Common Tasks in VS Code

### Run Tests
1. Click **Testing** icon (left sidebar)
2. Click **Configure Python Tests** → **pytest** → **tests**
3. Click ▷ to run tests

### Debug with Breakpoints
1. Click left of line number (red dot appears)
2. Press `F5`
3. Code pauses at breakpoint
4. Use toolbar: Continue (F5), Step Over (F10), Step Into (F11)

### Format Code on Save
Already configured! Just save (`Ctrl+S`) and code auto-formats.

---

## Troubleshooting Quick Fixes

### Virtual Environment Not Working?

**Windows PowerShell:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
venv\Scripts\Activate.ps1
```

**Windows CMD:**
```cmd
venv\Scripts\activate.bat
```

### Python Not Found?
1. `Ctrl+Shift+P` → **"Python: Select Interpreter"**
2. Choose `./venv/Scripts/python.exe` or `./venv/bin/python`

### Port 3000 Already in Use?

**Kill the process:**

**Windows:**
```cmd
netstat -ano | findstr :3000
taskkill /PID <PID> /F
```

**Mac/Linux:**
```bash
lsof -ti:3000 | xargs kill -9
```

### Import Errors?
```bash
# Activate venv first
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

# Reinstall
pip install -r requirements.txt --force-reinstall
```

---

## Project Structure Overview

```
face-recognition-attendance-marking-system/
├── .vscode/              ← VS Code configs (auto-configured)
├── src/                  ← Python source code
│   ├── config.py         ← Settings (port, thresholds)
│   ├── face_manager.py   ← Face recognition logic
│   ├── attendance_system.py
│   └── web_app.py        ← Flask web app
├── templates/            ← HTML files
├── static/               ← CSS, JavaScript
├── database/             ← User images (auto-created)
├── embeddings/           ← Face data (auto-created)
├── attendance_records/   ← Attendance logs (auto-created)
├── requirements.txt      ← Dependencies
└── run.py               ← Start here! ← ⭐
```

---

## Next Steps

### Configure Camera

**Local USB Camera:**
- Usually camera index `0`
- Try `1`, `2` if `0` doesn't work

**IP Camera (ESP32-CAM, Android IP Webcam):**
- ESP32-CAM: `http://192.168.1.100:81/stream`
- Android: `http://192.168.1.100:8080/video`
- Generic MJPEG: `http://IP:PORT/video`

### Customize Settings

Edit `src/config.py`:
```python
WEB_PORT = 3000              # Change port
SIMILARITY_THRESHOLD = 0.4   # Recognition sensitivity
DEFAULT_CAMERA_INDEX = 0     # Default camera
```

### Train Custom Models (Optional)

```bash
python train.py --epochs 30 --validation-split 0.2
```

Or use VS Code Task:
- `Ctrl+Shift+P` → **"Tasks: Run Task"** → **"Train Models"**

---

## Need More Help?

📖 **Full Documentation:**
- **VS Code Setup Guide:** `VS_CODE_SETUP.md` (detailed version)
- **Installation Guide:** `INSTRUCTIONS.md`
- **Project README:** `README.md`

💡 **Tips:**
- All VS Code configurations are in `.vscode/` folder
- Extensions will be recommended automatically
- Debug configurations are pre-configured (press F5)
- Tasks are available via `Ctrl+Shift+P` → "Tasks: Run Task"

🐛 **Issues?**
- Check `VS_CODE_SETUP.md` Troubleshooting section
- Open issue on GitHub

---

## VS Code Features You'll Love

✅ **Auto-configured for this project:**
- Python IntelliSense (auto-complete)
- Auto-formatting on save (Ruff)
- Linting (catches errors)
- Debugging with breakpoints
- Test runner integration
- Git integration
- Multiple run configurations

✅ **Pre-configured Tasks:**
- Run Application
- Run Tests
- Train Models
- Lint Code
- Format Code
- Install Dependencies

✅ **Pre-configured Debug Modes:**
- Run Application
- Run Demo
- Train Models
- Run Tests
- Flask Debug Mode

---

**You're all set! Start the app and visit http://localhost:3000** 🚀

For detailed explanations, see **`VS_CODE_SETUP.md`**
