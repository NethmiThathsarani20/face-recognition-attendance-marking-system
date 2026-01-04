# Physical Photos and Screenshots Guide

This document provides guidance on capturing the physical photos and screenshots referenced in the thesis but not automatically generated.

## Required Physical Hardware Photos

### 1. ESP32-CAM with LED Light Panel

**File:** `esp32_cam_led_setup.jpg`

**What to capture:**
- ESP32-CAM module with LED light panel mounted in ring configuration
- Close-up showing the circular LED arrangement around the camera lens
- Top view and side view
- LEDs should be illuminated to show the ring light effect

**Photo Tips:**
- Use macro mode for close-up details
- Ensure good lighting to show the setup clearly
- Include a ruler or coin for scale reference
- Take from multiple angles (top, side, front, 45-degree)

**Suggested filename:** `esp32_cam_led_panel_setup.jpg`

---

### 2. Raspberry Pi with Cooling Fan

**File:** `raspberry_pi_cooling_fan.jpg`

**What to capture:**
- Raspberry Pi 4 with 30mm cooling fan installed
- Show the fan mounted above the CPU
- Include GPIO connection wires visible
- Side view showing airflow direction

**Photo Tips:**
- Capture the complete Raspberry Pi board
- Show the fan's position relative to the CPU
- Include wire connections for documentation
- Consider showing the fan spinning (motion blur) vs static

**Suggested filename:** `raspberry_pi_with_cooling_fan.jpg`

---

### 3. Complete System Setup

**File:** `complete_system_setup.jpg`

**What to capture:**
- Raspberry Pi connected to power and network
- One or more ESP32-CAM units in position
- WiFi router (if used)
- Physical mounting/positioning of cameras
- Overall deployment configuration

**Photo Tips:**
- Show the complete system in a real-world setting
- Include cable management
- Demonstrate typical deployment scenario (e.g., at entrance)
- Wide shot showing all components together

**Suggested filename:** `complete_system_deployment.jpg`

---

### 4. Wiring Diagrams

**Files:** `led_wiring_diagram.jpg`, `fan_wiring_diagram.jpg`

**What to create:**
- Hand-drawn or digital circuit diagrams
- Show ESP32-CAM to LED panel connections
- Show Raspberry Pi GPIO to fan connections
- Label all pins and voltages clearly

**Tools:**
- Fritzing (open-source circuit design software)
- Draw.io or Lucidchart
- Or hand-drawn and scanned

**Suggested filenames:**
- `esp32_led_wiring_diagram.png`
- `raspberry_pi_fan_wiring_diagram.png`

---

## Required UI Screenshots

### 1. Dashboard Page

**File:** `dashboard_screenshot.png`

**What to capture:**
- Main dashboard of the web application
- Show navigation menu
- Display key statistics (total users, recent attendance)
- Capture at 1920x1080 resolution

**How to capture:**
1. Start the application: `python run.py`
2. Navigate to: `http://localhost:3000`
3. Use browser screenshot tool (F12 → Capture Screenshot)
4. Or use screenshot tool: `import pyautogui; pyautogui.screenshot('dashboard.png')`

**Suggested filename:** `web_dashboard_interface.png`

---

### 2. Add User Page

**File:** `add_user_screenshot.png`

**What to capture:**
- Add User interface
- Show file upload area
- Display form fields (username, etc.)
- Include preview thumbnails if images are uploaded

**Suggested filename:** `add_user_interface.png`

---

### 3. Mark Attendance Page

**File:** `mark_attendance_screenshot.png`

**What to capture:**
- Mark Attendance page with live camera feed
- Show ESP32-CAM stream or local camera
- Include face detection boxes if visible
- Show recognized name and confidence score
- Display attendance log table

**Suggested filename:** `mark_attendance_interface.png`

---

### 4. Face Detection Output

**File:** `face_detection_sample.png`

**What to capture:**
- Sample image showing detected face with bounding box
- Display the recognized person's name
- Show confidence percentage
- Include alignment landmarks if visible

**How to capture:**
- Use the mark_attendance endpoint
- Save a frame with detection overlay
- Or modify the code to save detection outputs

**Suggested filename:** `sample_face_detection_output.png`

---

## Comparison and Conceptual Photos

### 1. Traditional vs Automated Attendance

**File:** `traditional_vs_automated.jpg`

**What to create:**
- Split image showing traditional method (paper register, manual roll call) vs our system
- Use stock photos or create illustration
- Show the contrast in time and efficiency

**Suggested filename:** `attendance_methods_comparison_photo.jpg`

---

### 2. Thermal Camera Comparison

**File:** `thermal_comparison.jpg`

**What to capture (if thermal camera available):**
- Thermal image of Raspberry Pi without fan (showing heat)
- Thermal image of Raspberry Pi with fan (showing cooling)
- Side-by-side comparison

**Alternative:**
- Create visualization based on temperature readings
- Use color gradient overlays on regular photos
- Already generated as `temperature_performance_graph.png`

**Suggested filename:** `thermal_camera_comparison.jpg`

---

## Capturing Screenshots

### Using Browser Developer Tools

```bash
# Chrome/Firefox
1. Press F12 to open Developer Tools
2. Press Ctrl+Shift+P (Cmd+Shift+P on Mac)
3. Type "screenshot"
4. Select "Capture full size screenshot" or "Capture screenshot"
```

### Using Python Script

```python
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--window-size=1920,1080")

driver = webdriver.Chrome(options=chrome_options)
driver.get("http://localhost:3000")
driver.save_screenshot("dashboard_screenshot.png")
driver.quit()
```

### Using Command Line Tools

```bash
# Linux - using scrot
scrot -u screenshot.png

# macOS - using screencapture
screencapture -w screenshot.png

# Windows - using PowerShell
Add-Type -AssemblyName System.Windows.Forms
[System.Windows.Forms.SendKeys]::SendWait("{PrtSc}")
```

---

## Photo Organization

Organize captured photos in this directory structure:

```
thesis_diagrams/
├── hardware/
│   ├── esp32_cam_led_panel_setup.jpg
│   ├── raspberry_pi_with_cooling_fan.jpg
│   ├── complete_system_deployment.jpg
│   ├── esp32_led_wiring_diagram.png
│   └── raspberry_pi_fan_wiring_diagram.png
├── ui_screenshots/
│   ├── web_dashboard_interface.png
│   ├── add_user_interface.png
│   ├── mark_attendance_interface.png
│   └── sample_face_detection_output.png
└── comparison/
    ├── attendance_methods_comparison_photo.jpg
    └── thermal_camera_comparison.jpg
```

---

## Image Specifications

### For Print Quality Thesis

- **Resolution:** Minimum 300 DPI
- **Format:** PNG for screenshots, JPG for photos
- **Dimensions:** 
  - Full-page width: ~2500-3000 pixels wide
  - Half-page: ~1500 pixels wide
  - Small figures: ~800-1000 pixels wide
- **File Size:** Keep under 2MB per image (compress if needed)

### Color and Contrast

- Use good lighting for hardware photos
- Ensure high contrast for visibility
- Avoid shadows and reflections
- Use neutral background (white or light gray)

---

## Tools and Resources

### Photo Editing
- **GIMP:** Free, open-source (https://www.gimp.org/)
- **Paint.NET:** Windows (https://www.getpaint.net/)
- **Preview:** macOS built-in

### Circuit Diagrams
- **Fritzing:** Circuit design (https://fritzing.org/)
- **Draw.io:** Diagramming (https://app.diagrams.net/)
- **KiCad:** Professional PCB design (https://www.kicad.org/)

### Screenshot Tools
- **Flameshot:** Linux (https://flameshot.org/)
- **Snagit:** Commercial, all platforms
- **Greenshot:** Windows, free

---

## Checklist

Before finalizing your thesis, ensure you have:

- [ ] ESP32-CAM hardware photos (multiple angles)
- [ ] Raspberry Pi with fan photos
- [ ] Complete system setup photo
- [ ] LED wiring diagram
- [ ] Fan wiring diagram
- [ ] Dashboard screenshot
- [ ] Add User page screenshot
- [ ] Mark Attendance page screenshot
- [ ] Face detection sample output
- [ ] Traditional vs automated comparison
- [ ] All images properly named and organized
- [ ] Image captions written
- [ ] References updated in THESIS.md

---

## Questions?

If you need help capturing or creating any of these images, refer to:
- Project documentation in docs/
- README.md for system setup
- ESP32_CAM_GUIDE.md for camera setup
- Contact the development team

---

**Last Updated:** January 2026
