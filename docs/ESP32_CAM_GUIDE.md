# ESP32-CAM Setup and Testing Guide

This guide will help you set up your ESP32-CAM, find its IP address, and verify it works correctly with the face recognition attendance system.

## Table of Contents
1. [Hardware Setup](#hardware-setup)
2. [Firmware Upload](#firmware-upload)
3. [Finding ESP32 Cam IP Address](#finding-esp32-cam-ip-address)
4. [Testing ESP32 Cam](#testing-esp32-cam)
5. [Using with Web Interface](#using-with-web-interface)
6. [Troubleshooting](#troubleshooting)

## Hardware Setup

### Required Components
- ESP32-CAM module (AI-Thinker model recommended)
- FTDI programmer or USB-to-TTL adapter
- Jumper wires
- 5V power supply (recommended: separate power supply, not from FTDI)
- Micro SD card (optional, for saving images)

### Wiring Connections

**For Programming (Upload Firmware):**
```
ESP32-CAM    →    FTDI/USB-TTL
GND          →    GND
5V           →    5V (use external 5V power if FTDI cannot provide enough current)
U0R (RX)     →    TX
U0T (TX)     →    RX
GPIO0        →    GND (for programming mode only)
```

**For Normal Operation:**
```
ESP32-CAM    →    Power Supply
GND          →    GND
5V           →    5V (1A or higher recommended)

Note: Disconnect GPIO0 from GND for normal operation
```

## Firmware Upload

### 1. Install Arduino IDE
Download and install from: https://www.arduino.cc/en/software

### 2. Install ESP32 Board Support
1. Open Arduino IDE
2. Go to `File → Preferences`
3. Add this URL to "Additional Board Manager URLs":
   ```
   https://dl.espressif.com/dl/package_esp32_index.json
   ```
4. Go to `Tools → Board → Boards Manager`
5. Search for "esp32" and install "esp32 by Espressif Systems"

### 3. Configure WiFi Credentials
1. Navigate to `esp32-camera/` directory
2. Open `esp32-camera.ino` in Arduino IDE
3. Update WiFi credentials (lines 12-13):
   ```cpp
   const char *ssid = "YOUR_WIFI_SSID";
   const char *password = "YOUR_WIFI_PASSWORD";
   ```

### 4. Select Board and Port
1. Go to `Tools → Board` and select "AI Thinker ESP32-CAM"
2. Go to `Tools → Port` and select your FTDI/USB-TTL port
3. Go to `Tools → Partition Scheme` and select "Huge APP (3MB No OTA/1MB SPIFFS)"

### 5. Upload Firmware
1. Connect GPIO0 to GND on ESP32-CAM (programming mode)
2. Click "Upload" button in Arduino IDE
3. Wait for upload to complete
4. Disconnect GPIO0 from GND
5. Press the RESET button on ESP32-CAM

### 6. Verify Upload
1. Open Serial Monitor (115200 baud)
2. Press RESET button on ESP32-CAM
3. You should see output like:
   ```
   WiFi connecting.......
   WiFi connected
   Camera Ready! Use 'http://192.168.1.100' to connect
   ```
4. Note the IP address displayed

## Finding ESP32 Cam IP Address

### Method 1: Serial Monitor (Recommended)
1. Connect ESP32-CAM to FTDI/USB-TTL adapter
2. Open Arduino IDE Serial Monitor (115200 baud)
3. Press RESET button on ESP32-CAM
4. IP address will be displayed in the output

### Method 2: Check Router's DHCP Table
1. Log into your router's admin interface
2. Look for DHCP client list or connected devices
3. Find device with MAC address starting with `d8:3a:dd` or `80:f3:da`
4. Note the IP address

### Method 3: Use ip.py Script
If you know the MAC address of your ESP32-CAM:

```bash
# Find ESP32 Cam IP by MAC address
python ip.py d8:3a:dd:51:6b:3c

# Or with your specific MAC address
python ip.py <YOUR_ESP32_MAC_ADDRESS>
```

Example output:
```
Found IP(s) for MAC d8:3a:dd:51:6b:3c
192.168.1.100
```

### Common ESP32-CAM URLs
Once you have the IP address, the stream URL will be:
```
http://<ESP32_IP>:81/stream

Examples:
http://192.168.1.100:81/stream
http://10.74.63.131:81/stream
```

## Testing ESP32 Cam

### Quick Test Script
We provide a comprehensive testing script to verify everything works:

```bash
# Test with known ESP32 URL
python test_esp32_cam.py --url http://192.168.1.100:81/stream --all

# Find IP by MAC and test
python test_esp32_cam.py --mac d8:3a:dd:51:6b:3c --all

# Just find the IP
python test_esp32_cam.py --mac d8:3a:dd:51:6b:3c --find-only
```

### Manual Testing Steps

#### 1. Test in Web Browser
Open your ESP32 stream URL in a web browser:
```
http://192.168.1.100:81/stream
```
You should see live video from the camera.

#### 2. Test with curl
```bash
curl -I http://192.168.1.100:81/stream
```

Expected output:
```
HTTP/1.1 200 OK
Content-Type: multipart/x-mixed-replace; boundary=frame
```

#### 3. Test with OpenCV (Python)
```python
import cv2

# Replace with your ESP32 IP
camera_url = "http://192.168.1.100:81/stream"

cap = cv2.VideoCapture(camera_url)

if cap.isOpened():
    print("✅ Camera connection successful!")
    ret, frame = cap.read()
    if ret:
        print(f"✅ Frame captured: {frame.shape}")
    else:
        print("❌ Failed to read frame")
else:
    print("❌ Failed to open camera")

cap.release()
```

## Using with Web Interface

### 1. Start the Application
```bash
# Activate virtual environment
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows

# Run the application
python run.py
```

### 2. Access Web Interface
Open your browser and go to:
```
http://localhost:3000
```

### 3. Configure Camera Settings
1. In the "Mark Attendance" section
2. Select "IP Camera (URL)" option
3. Enter your ESP32 cam URL:
   ```
   http://192.168.1.100:81/stream
   ```

### 4. Test Live Stream
1. Click "Start Live Stream" button
2. You should see live video with face detection boxes
3. Detected faces will be highlighted with bounding boxes

### 5. Mark Attendance
1. Position a registered user in front of the ESP32 camera
2. Click "Mark Attendance" button
3. System will:
   - Capture image from ESP32 cam
   - Detect and recognize face
   - Mark attendance if face is recognized
   - Display result with confidence score

### 6. Enable Auto Mode (Optional)
1. Check "Enable Automatic Recognition Mode"
2. System will automatically capture and recognize faces every 3 seconds
3. Attendance will be marked automatically for recognized users

## Troubleshooting

### Camera Not Connecting

**Problem:** Cannot connect to ESP32 cam stream

**Solutions:**
1. Verify WiFi credentials are correct in firmware
2. Check that ESP32 and computer are on the same network
3. Try pinging ESP32 IP:
   ```bash
   ping 192.168.1.100
   ```
4. Check firewall settings
5. Power cycle ESP32-CAM (disconnect and reconnect power)
6. Check Serial Monitor for error messages

### Poor Video Quality

**Problem:** Video is blurry or low quality

**Solutions:**
1. Clean the camera lens
2. Adjust camera focus (if adjustable)
3. Check lighting conditions
4. Modify frame size in firmware:
   ```cpp
   // In esp32-camera.ino, around line 92
   s->set_framesize(s, FRAMESIZE_SVGA);  // Try different sizes
   // Options: FRAMESIZE_QVGA, FRAMESIZE_VGA, FRAMESIZE_SVGA, FRAMESIZE_XGA
   ```

### Connection Drops

**Problem:** Stream keeps disconnecting

**Solutions:**
1. Use stronger power supply (5V, 1A or higher)
2. Move ESP32 closer to WiFi router
3. Check for WiFi interference
4. Add external WiFi antenna if supported
5. Reduce frame rate in firmware

### Face Not Detected

**Problem:** Faces not being detected in ESP32 stream

**Solutions:**
1. Improve lighting conditions
2. Position face closer to camera (50-100cm recommended)
3. Ensure face is looking directly at camera
4. Check if face detection works with other cameras:
   ```bash
   python test_esp32_cam.py --url http://192.168.1.100:81/stream
   ```
5. Adjust detection threshold in `src/config.py`:
   ```python
   DETECTION_THRESHOLD = 0.3  # Lower = more sensitive
   ```

### Attendance Not Marking

**Problem:** Face detected but attendance not marked

**Solutions:**
1. Verify user is registered in the system
2. Check user list:
   ```python
   from src.attendance_system import AttendanceSystem
   system = AttendanceSystem()
   print(system.get_user_list())
   ```
3. Add more training images for the user
4. Check lighting and image quality
5. Adjust similarity threshold in `src/config.py`:
   ```python
   SIMILARITY_THRESHOLD = 0.3  # Lower = more lenient
   ```

### Serial Monitor Shows Errors

**Common Error Messages:**

**"Camera init failed"**
- Check camera module is properly connected
- Verify board selection is "AI Thinker ESP32-CAM"
- Try power cycling

**"WiFi connection failed"**
- Verify SSID and password are correct
- Check WiFi signal strength
- Ensure 2.4GHz WiFi is enabled (ESP32 doesn't support 5GHz)

**"PSRAM not found"**
- Normal for some boards, but limits resolution
- Check board has PSRAM chip
- Use lower resolution settings

### Network Issues

**Problem:** Cannot find ESP32 on network

**Solutions:**
1. Check ESP32 is powered on (LED should be on)
2. Verify ESP32 connected to WiFi (check Serial Monitor)
3. Use network scanner to find device:
   ```bash
   # Linux/macOS
   arp -a | grep -i "d8:3a:dd\|80:f3:da"
   
   # Windows
   arp -a
   ```
4. Check router's connected devices list
5. Try assigning static IP in firmware

## Best Practices

### For Optimal Performance
1. **Power Supply**: Use 5V power supply with at least 1A capacity
2. **Lighting**: Ensure good lighting for face detection (avoid backlight)
3. **Distance**: Keep face 50-100cm from camera
4. **Network**: Place ESP32 within good WiFi range
5. **Training**: Add 5-10 images per user for better recognition
6. **Positioning**: Mount camera at face height for best results

### For Production Deployment
1. **Security**: Change default WiFi password
2. **Static IP**: Configure static IP for ESP32 to avoid IP changes
3. **Enclosure**: Use protective case for ESP32-CAM
4. **Backup**: Keep backup of working firmware
5. **Monitoring**: Regularly check Serial Monitor for errors
6. **Updates**: Keep firmware updated with latest fixes

## Additional Resources

### Documentation
- [ESP32-CAM Getting Started](https://github.com/espressif/esp32-camera)
- [Arduino ESP32 Documentation](https://docs.espressif.com/projects/arduino-esp32/)
- [Project README](../README.md)

### Sample URLs
```bash
# ESP32-CAM stream
http://192.168.1.100:81/stream

# ESP32-CAM snapshot
http://192.168.1.100:80/capture

# Face detection endpoint
http://localhost:3000/video_feed/http://192.168.1.100:81/stream
```

### Testing Commands
```bash
# Find IP by MAC
python ip.py d8:3a:dd:51:6b:3c

# Comprehensive test
python test_esp32_cam.py --url http://192.168.1.100:81/stream --all

# Test connectivity only
python test_esp32_cam.py --url http://192.168.1.100:81/stream

# Test with auto mode enabled
# (Use web interface with "Enable Automatic Recognition Mode" checked)
```

## Support

If you encounter issues not covered in this guide:
1. Check the main [README.md](../README.md) troubleshooting section
2. Review Serial Monitor output for error messages
3. Test with the provided `test_esp32_cam.py` script
4. Check project issues on GitHub

---

**Quick Reference Card**

```
┌─────────────────────────────────────────────────────────┐
│ ESP32-CAM Quick Reference                               │
├─────────────────────────────────────────────────────────┤
│ Default URL: http://[ESP32_IP]:81/stream               │
│ Example:     http://192.168.1.100:81/stream            │
│                                                         │
│ Find IP:     python ip.py <MAC_ADDRESS>                │
│ Test:        python test_esp32_cam.py --url <URL>      │
│ Web UI:      http://localhost:3000                      │
│                                                         │
│ Programming: Connect GPIO0 → GND                        │
│ Normal:      Disconnect GPIO0 from GND                  │
│                                                         │
│ Power:       5V, 1A minimum                            │
│ WiFi:        2.4GHz only (not 5GHz)                    │
└─────────────────────────────────────────────────────────┘
```
