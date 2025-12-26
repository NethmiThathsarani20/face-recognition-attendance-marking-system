# Visual Studio Code Setup Guide - Face Recognition Attendance System

Complete step-by-step guide to set up and run the Face Recognition Attendance System in Visual Studio Code.

---

## Table of Contents
1. [Prerequisites Installation](#prerequisites-installation)
2. [VS Code Setup](#vs-code-setup)
3. [Project Installation](#project-installation)
4. [Running the Application](#running-the-application)
5. [Using VS Code Features](#using-vs-code-features)
6. [Debugging](#debugging)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites Installation

### Step 1: Install Python

**Windows:**
1. Download Python 3.12 from [python.org](https://www.python.org/downloads/)
2. During installation, **check "Add Python to PATH"**
3. Verify installation:
   ```cmd
   python --version
   ```

**macOS:**
```bash
# Using Homebrew
brew install python@3.12

# Verify
python3 --version
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv python3-dev build-essential
python3 --version
```

### Step 2: Install Visual Studio Code

1. Download VS Code from [code.visualstudio.com](https://code.visualstudio.com/)
2. Install for your operating system
3. Launch VS Code

### Step 3: Install Git

**Windows:**
- Download from [git-scm.com](https://git-scm.com/downloads)

**macOS:**
```bash
brew install git
```

**Linux:**
```bash
sudo apt install git
```

Verify:
```bash
git --version
```

---

## VS Code Setup

### Step 1: Install Required Extensions

Open VS Code and install these essential extensions:

1. **Python** (Microsoft) - `ms-python.python`
   - Press `Ctrl+Shift+X` (Windows/Linux) or `Cmd+Shift+X` (macOS)
   - Search "Python" and install the official Microsoft extension

2. **Pylance** (Microsoft) - `ms-python.vscode-pylance`
   - Advanced Python language support

3. **Ruff** (Charlie Marsh) - `charliermarsh.ruff`
   - Fast Python linter and formatter

4. **Python Debugger** (Microsoft) - `ms-python.debugpy`
   - Python debugging support

**Optional but Recommended:**
- **GitHub Copilot** - AI pair programmer
- **GitLens** - Enhanced Git capabilities
- **Code Spell Checker** - Spell checking in code
- **Jinja** - Template support for HTML files

**Quick Install via Command Palette:**
1. Press `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (macOS)
2. Type "Extensions: Show Recommended Extensions"
3. Click "Install All"

---

## Project Installation

### Step 1: Clone the Repository

Open VS Code Terminal (`Ctrl+`\` or Terminal → New Terminal):

```bash
# Navigate to your desired directory
cd ~/Documents  # or C:\Users\YourName\Documents on Windows

# Clone the repository
git clone https://github.com/NethmiThathsarani20/face-recognition-attendance-marking-system.git

# Navigate into the project
cd face-recognition-attendance-marking-system
```

### Step 2: Open Project in VS Code

**Option A: From Terminal**
```bash
code .
```

**Option B: From VS Code**
1. File → Open Folder
2. Select the `face-recognition-attendance-marking-system` folder

### Step 3: Create Virtual Environment

In VS Code Terminal:

**Windows:**
```cmd
python -m venv venv
```

**macOS/Linux:**
```bash
python3 -m venv venv
```

### Step 4: Activate Virtual Environment

**Windows (Command Prompt):**
```cmd
venv\Scripts\activate
```

**Windows (PowerShell):**
```powershell
venv\Scripts\Activate.ps1
```

If you get an error about execution policies in PowerShell:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

### Step 5: Select Python Interpreter

1. Press `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (macOS)
2. Type "Python: Select Interpreter"
3. Choose the interpreter from `./venv/bin/python` or `.\venv\Scripts\python.exe`

**Or click on the Python version in the bottom-right status bar**

### Step 6: Install Dependencies

With virtual environment activated:

```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt
```

This will install (may take 5-15 minutes):
- `insightface` - Face recognition
- `opencv-python` - Computer vision
- `flask` - Web framework
- `numpy`, `tensorflow`, `scikit-learn` - ML libraries
- And more...

### Step 7: Verify Installation

```bash
python verify_requirements.py
```

This checks if all dependencies are correctly installed.

---

## Running the Application

### Method 1: Using VS Code Run Button (Recommended)

1. Open `run.py` in the editor
2. Click the **Run Python File** button (▷) in the top-right corner
3. Or press `Ctrl+Shift+D` and select "Python: Run Application" from the dropdown
4. Click the green play button ▷

### Method 2: Using Terminal

In VS Code Terminal (with venv activated):

```bash
python run.py
```

### Method 3: Using VS Code Tasks

1. Press `Ctrl+Shift+P` (or `Cmd+Shift+P`)
2. Type "Tasks: Run Task"
3. Select "Run Application"

### Access the Web Interface

Once running, you'll see:
```
Starting Simple Attendance System...
Open your browser and go to: http://localhost:3000
Press Ctrl+C to stop the server
```

**Open your browser:**
- Local: `http://localhost:3000`
- Remote (if on Raspberry Pi): `http://<your-pi-ip>:3000`

### Stop the Application

Press `Ctrl+C` in the terminal or click the trash can icon in the terminal panel.

---

## Using VS Code Features

### Running Tests

**Option 1: Test Explorer**
1. Click the Testing icon in the Activity Bar (left sidebar)
2. Click "Configure Python Tests"
3. Select "pytest"
4. Select "tests" directory
5. Click the play button to run tests

**Option 2: Terminal**
```bash
python tests/run_tests.py
```

**Option 3: VS Code Task**
1. `Ctrl+Shift+P` → "Tasks: Run Task" → "Run Tests"

### Linting and Formatting

**Auto-format on Save:**
Already configured in `.vscode/settings.json`

**Manual Formatting:**
- Press `Shift+Alt+F` (Windows/Linux) or `Shift+Option+F` (macOS)

**Run Linter:**
1. `Ctrl+Shift+P` → "Tasks: Run Task" → "Run Linter (Ruff)"

**Or via Terminal:**
```bash
# Check and fix issues
make lint

# Format code
make format
```

### File Navigation

**Quick File Open:**
- `Ctrl+P` (Windows/Linux) or `Cmd+P` (macOS)
- Type filename to open

**Go to Symbol:**
- `Ctrl+Shift+O` (Windows/Linux) or `Cmd+Shift+O` (macOS)
- Shows functions, classes in current file

**Find in Files:**
- `Ctrl+Shift+F` (Windows/Linux) or `Cmd+Shift+F` (macOS)

### Terminal Management

**New Terminal:**
- `` Ctrl+Shift+` `` or Terminal → New Terminal

**Split Terminal:**
- Click the split icon in terminal toolbar

**Multiple Terminals:**
- Useful for running app in one, tests in another

---

## Debugging

### Debug the Main Application

1. Set breakpoints by clicking left of line numbers (red dot appears)
2. Press `F5` or click Run → Start Debugging
3. Select "Python: Run Application"
4. Code will pause at breakpoints
5. Use debug toolbar to:
   - **Continue** (F5)
   - **Step Over** (F10)
   - **Step Into** (F11)
   - **Step Out** (Shift+F11)
   - **Restart** (Ctrl+Shift+F5)
   - **Stop** (Shift+F5)

### Debug Variables

- **Variables Panel:** Shows all variables in current scope
- **Watch Panel:** Add expressions to monitor
- **Call Stack:** See function call hierarchy
- **Debug Console:** Execute code in debug context

### Available Debug Configurations

Open `.vscode/launch.json` to see all configurations:

1. **Python: Run Application** - Main app with debugger
2. **Python: Run Demo** - Demo script
3. **Python: Train Models** - Model training with parameters
4. **Python: Run Tests** - Test suite
5. **Python: Current File** - Debug currently open file
6. **Python: Flask Debug** - Flask app with Jinja template debugging

### Quick Debugging Tips

- `F9` - Toggle breakpoint
- `F5` - Start/Continue debugging
- `F10` - Step over
- `F11` - Step into
- Hover over variables to see values
- Use Debug Console to evaluate expressions

---

## Common VS Code Tasks

### Installing New Packages

```bash
# Activate venv first
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Install package
pip install package-name

# Update requirements.txt
pip freeze > requirements.txt
```

### Training Models

**Via Task:**
1. `Ctrl+Shift+P` → "Tasks: Run Task" → "Train Models"

**Via Terminal:**
```bash
python train.py --epochs 30 --validation-split 0.2
```

**Via Debug:**
1. Press `F5`
2. Select "Python: Train Models"

### Setup Development Environment

Install development tools (linters, formatters, etc.):

```bash
make setup-dev
```

Or via task: `Ctrl+Shift+P` → "Tasks: Run Task" → "Setup Development Environment"

### Clean Project

Remove temporary files:

```bash
make clean
```

Or via task: `Ctrl+Shift+P` → "Tasks: Run Task" → "Clean Project"

---

## Workspace Organization

### Recommended Folder Structure View

In VS Code Explorer:
```
face-recognition-attendance-marking-system/
├── .vscode/              ← VS Code configurations (NEW)
│   ├── settings.json     ← Editor settings
│   ├── launch.json       ← Debug configurations
│   ├── tasks.json        ← Build/run tasks
│   └── extensions.json   ← Recommended extensions
├── src/                  ← Python source code
│   ├── config.py
│   ├── face_manager.py
│   ├── attendance_system.py
│   └── web_app.py
├── templates/            ← HTML templates
├── static/               ← CSS, JavaScript
├── database/             ← User face images (auto-created)
├── embeddings/           ← Face embeddings (auto-created)
├── attendance_records/   ← Daily attendance logs (auto-created)
├── tests/                ← Test suite
├── requirements.txt      ← Python dependencies
└── run.py               ← Main entry point
```

### Useful VS Code Shortcuts

**General:**
- `Ctrl+Shift+P` / `Cmd+Shift+P` - Command Palette
- `Ctrl+P` / `Cmd+P` - Quick Open File
- `Ctrl+,` / `Cmd+,` - Settings
- `Ctrl+Shift+E` / `Cmd+Shift+E` - Explorer
- `Ctrl+Shift+G` / `Cmd+Shift+G` - Source Control
- `Ctrl+Shift+D` / `Cmd+Shift+D` - Debug
- `` Ctrl+` `` / `` Cmd+` `` - Toggle Terminal

**Editing:**
- `Alt+Up/Down` - Move line up/down
- `Shift+Alt+Up/Down` - Copy line up/down
- `Ctrl+/` / `Cmd+/` - Toggle comment
- `Ctrl+D` / `Cmd+D` - Select next occurrence
- `Alt+Click` - Multiple cursors

**Navigation:**
- `Ctrl+Tab` - Switch between files
- `Alt+Left/Right` - Navigate back/forward
- `Ctrl+G` - Go to line
- `F12` - Go to definition
- `Alt+F12` - Peek definition

---

## Troubleshooting

### Issue: Virtual Environment Not Activating

**Solution:**

**Windows PowerShell:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
venv\Scripts\Activate.ps1
```

**Windows Command Prompt:**
```cmd
venv\Scripts\activate.bat
```

### Issue: Python Interpreter Not Found

**Solution:**
1. `Ctrl+Shift+P` → "Python: Select Interpreter"
2. If venv not listed, click "Enter interpreter path..."
3. Browse to `venv/Scripts/python.exe` (Windows) or `venv/bin/python` (Mac/Linux)

### Issue: Import Errors

**Solution:**
```bash
# Ensure venv is activated
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"  # Mac/Linux
set PYTHONPATH=%PYTHONPATH%;%CD%          # Windows
```

### Issue: Linter/Formatter Not Working

**Solution:**
1. Install Ruff in venv:
   ```bash
   pip install ruff
   ```
2. Reload VS Code: `Ctrl+Shift+P` → "Developer: Reload Window"
3. Check extension is installed: Search "Ruff" in Extensions

### Issue: Debugger Not Starting

**Solution:**
1. Install debugpy:
   ```bash
   pip install debugpy
   ```
2. Check `.vscode/launch.json` exists
3. Select correct debug configuration from dropdown
4. Ensure file you're debugging is saved

### Issue: Terminal Shows Wrong Python

**Solution:**
1. Close all terminals
2. Reopen terminal (`` Ctrl+` ``)
3. Activate venv again
4. Verify: `which python` (Mac/Linux) or `where python` (Windows)

### Issue: Port 3000 Already in Use

**Solution:**

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

---

## Advanced Features

### Using Multiple Python Versions

If you need different Python versions:

```bash
# Create venv with specific Python
python3.12 -m venv venv312
source venv312/bin/activate

# Or on Windows
py -3.12 -m venv venv312
venv312\Scripts\activate
```

### Remote Development (Raspberry Pi)

1. Install "Remote - SSH" extension
2. `Ctrl+Shift+P` → "Remote-SSH: Connect to Host"
3. Enter: `pi@<raspberry-pi-ip>`
4. Enter password
5. Open project folder on Pi
6. Continue development remotely

### Integrated Git

**Stage and Commit:**
1. `Ctrl+Shift+G` - Open Source Control
2. Click `+` next to files to stage
3. Enter commit message
4. Click ✓ to commit
5. Click ⋯ → Push

**Or use terminal:**
```bash
git add .
git commit -m "Your message"
git push
```

### Snippets

Create custom snippets for common code:

1. File → Preferences → User Snippets
2. Select "Python"
3. Add custom snippets

---

## Quick Reference

### Starting Fresh

```bash
# 1. Clone repo
git clone https://github.com/NethmiThathsarani20/face-recognition-attendance-marking-system.git
cd face-recognition-attendance-marking-system

# 2. Open in VS Code
code .

# 3. Create and activate venv (in VS Code terminal)
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

# 4. Select interpreter in VS Code
# Ctrl+Shift+P → "Python: Select Interpreter" → Choose venv

# 5. Install dependencies
pip install -r requirements.txt

# 6. Run application
python run.py

# 7. Open browser
# http://localhost:3000
```

### Daily Workflow

```bash
# 1. Open VS Code to project folder
code ~/path/to/face-recognition-attendance-marking-system

# 2. Activate venv (if terminal doesn't auto-activate)
source venv/bin/activate

# 3. Pull latest changes
git pull

# 4. Run application
python run.py

# Or press F5 to debug
```

---

## Additional Resources

- **VS Code Python Tutorial:** https://code.visualstudio.com/docs/python/python-tutorial
- **VS Code Debugging:** https://code.visualstudio.com/docs/editor/debugging
- **Python in VS Code:** https://code.visualstudio.com/docs/languages/python
- **Project README:** See `README.md` in project root
- **Setup Instructions:** See `INSTRUCTIONS.md` for detailed installation

---

## Support

For issues or questions:
- Check [Troubleshooting](#troubleshooting) section above
- See `README.md` for general project information
- Open issue on GitHub: https://github.com/NethmiThathsarani20/face-recognition-attendance-marking-system/issues

---

**Happy Coding! 🚀**
