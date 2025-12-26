# How to Run in Visual Studio Code - Quick Guide

**Simple steps to run the Face Recognition Attendance System UI in VS Code**

---

## Prerequisites Already Installed

Before running, make sure you have:
- ✅ Python 3.8+ installed
- ✅ VS Code installed
- ✅ Project cloned/downloaded
- ✅ Virtual environment created and activated
- ✅ Dependencies installed (`pip install -r requirements.txt`)

If you haven't done the setup yet, see `QUICK_START_VS_CODE.md` or `VS_CODE_SETUP.md`.

---

## 🚀 How to Run in VS Code

### Method 1: Simple Run (Click and Go) ⭐ **EASIEST**

1. Open the project folder in VS Code
2. Open the file `run.py` in the editor
3. Click the **▷ Run** button in the top-right corner of the editor
4. Done! The app starts in the terminal

![Run Button Location: Top-right corner, green play button ▷]

### Method 2: Press F5 to Debug

1. Open VS Code with the project folder
2. Press **F5** key
3. Select **"Python: Run Application"** from the dropdown (first time only)
4. Application starts with debugger attached
5. Done!

### Method 3: Use Terminal in VS Code

1. Open VS Code terminal: Press `` Ctrl+` `` (backtick key)
2. Make sure virtual environment is activated (you should see `(venv)` in prompt)
3. Type:
   ```bash
   python run.py
   ```
4. Press Enter
5. Done!

### Method 4: Use VS Code Tasks

1. Press `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (Mac)
2. Type: **"Tasks: Run Task"**
3. Select: **"Run Application"**
4. Done!

---

## ✅ What Happens When You Run

You'll see this output in the terminal:
```
Starting Simple Attendance System...
Open your browser and go to: http://localhost:3000
Press Ctrl+C to stop the server
```

---

## 🌐 Access the UI

1. Open your web browser (Chrome, Firefox, Edge, Safari)
2. Go to: **http://localhost:3000**
3. You should see the Face Recognition Attendance System dashboard

---

## 🛑 How to Stop

**To stop the application:**
- Press **Ctrl+C** in the terminal, or
- Click the **trash can icon** in the terminal panel, or
- Click the **red square** button (if in debug mode)

---

## 🔧 Quick Troubleshooting

### Virtual Environment Not Activated?

**Windows:**
```cmd
venv\Scripts\activate
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

Look for `(venv)` at the start of your terminal prompt.

### Port Already in Use?

Change port in `src/config.py`:
```python
WEB_PORT = 5000  # or any other port
```

Or kill the process using port 3000:

**Windows:**
```cmd
netstat -ano | findstr :3000
taskkill /PID <PID_NUMBER> /F
```

**Mac/Linux:**
```bash
lsof -ti:3000 | xargs kill -9
```

### Python Not Found?

1. Press `Ctrl+Shift+P`
2. Type: "Python: Select Interpreter"
3. Choose the one from `venv` folder

---

## 📝 Summary

**Simplest way to run:**
1. Open VS Code
2. Open `run.py` file
3. Click ▷ Run button (top-right)
4. Open browser: http://localhost:3000

**That's it!** 🎉

---

## Need More Help?

- **Setup Guide**: See `QUICK_START_VS_CODE.md`
- **Detailed Guide**: See `VS_CODE_SETUP.md`
- **Full Documentation**: See `README.md`
