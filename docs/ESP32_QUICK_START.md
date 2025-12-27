# ESP32-CAM Quick Start Guide

## Overview
This guide helps you quickly test your ESP32-CAM with the face recognition attendance system.

## Prerequisites
- ESP32-CAM module with firmware uploaded
- ESP32-CAM connected to WiFi network
- Python virtual environment activated
- Face recognition system installed

## Quick Testing Steps

### Step 1: Find Your ESP32-CAM IP Address

#### Method A: Check Serial Monitor (Easiest)
1. Connect ESP32-CAM to FTDI/USB-TTL adapter
2. Open Arduino IDE Serial Monitor (115200 baud)
3. Press RESET button on ESP32-CAM
4. Note the IP address shown (e.g., `192.168.1.100`)

#### Method B: Use ip.py Script
```bash
# If you know your ESP32-CAM MAC address
python ip.py d8:3a:dd:51:6b:3c

# Replace with your actual MAC address
# Common ESP32 MAC prefixes: d8:3a:dd, 80:f3:da, 24:0a:c4
```

#### Method C: Check Router
1. Log into your router's admin interface
2. Look for connected devices / DHCP clients
3. Find device named "ESP32-CAM" or similar

### Step 2: Verify ESP32-CAM is Accessible

Test in your web browser:
```
http://<ESP32_IP>:81/stream
```

Example:
```
http://192.168.1.100:81/stream
```

You should see live video from the camera.

### Step 3: Run Automated Tests

```bash
# Activate virtual environment
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows

# Test with your ESP32-CAM URL
python test_esp32_cam.py --url http://192.168.1.100:81/stream --all
```

**What this tests:**
- ✅ Network connectivity to ESP32-CAM
- ✅ HTTP stream availability
- ✅ Video capture quality
- ✅ Face detection with ESP32 stream
- ✅ Attendance marking functionality

Expected output:
```
================================================================================
  Step 1: Find ESP32 Cam IP Address
================================================================================
ℹ️  Using camera URL: http://192.168.1.100:81/stream

================================================================================
  Step 2: Test ESP32 Cam Connectivity
================================================================================
✅ HTTP connection successful
✅ Stream endpoint accessible

================================================================================
  Step 3: Test Video Capture
================================================================================
✅ Video stream opened successfully
✅ Successfully captured all 10 frames

================================================================================
  Step 4: Test Face Detection
================================================================================
✅ Face detection system initialized
💡 TIP: Position your face in front of the ESP32 camera
✅ Frame 1: Detected 1 face(s)
...

================================================================================
  Step 5: Test Attendance Marking
================================================================================
✅ Attendance marked successfully!
  User: John_Doe
  Confidence: 0.85
```

### Step 4: Test Live Stream UI

#### Option A: Use Web Interface

1. Start the web application:
   ```bash
   python run.py
   ```

2. Open browser: `http://localhost:3000`

3. In the "Mark Attendance" section:
   - Select "IP Camera (URL)" option
   - Enter your ESP32 cam URL: `http://192.168.1.100:81/stream`
   - Click "Start Live Stream" button

4. You should see:
   - ✅ Live video from ESP32 camera
   - ✅ Green boxes around recognized faces
   - ✅ Red boxes around unknown faces
   - ✅ Names and confidence scores displayed

5. Click "Mark Attendance" to capture and mark attendance

#### Option B: Use Direct Video Feed URL

Open in browser:
```
http://localhost:3000/video_feed/http://192.168.1.100:81/stream
```

This shows just the video feed with face detection.

#### Option C: Use Live Demo Script

```bash
# View live stream in OpenCV window
python demo_esp32_live.py --url http://192.168.1.100:81/stream
```

**Controls:**
- Press `q` or `ESC` to quit
- Press `s` to save current frame
- Press `SPACE` to mark attendance

### Step 5: Enable Automatic Recognition

1. In web interface, check "Enable Automatic Recognition Mode"
2. System will automatically:
   - Capture frames every 3 seconds
   - Detect and recognize faces
   - Mark attendance for recognized users
   - Update attendance table in real-time

## Common URLs

Replace IP address with your ESP32-CAM's actual IP:

```bash
# ESP32-CAM stream
http://192.168.1.100:81/stream

# Web application (local)
http://localhost:3000

# Web application (remote access)
http://<raspberry-pi-ip>:3000

# Direct video feed
http://localhost:3000/video_feed/http://192.168.1.100:81/stream

# Camera test endpoint
http://localhost:3000/camera_test/http://192.168.1.100:81/stream
```

## Troubleshooting

### "Failed to open camera"
1. Verify ESP32-CAM is powered on
2. Check IP address is correct
3. Test URL in web browser first
4. Ensure ESP32-CAM and computer are on same network
5. Check firewall settings

### "No faces detected"
1. Improve lighting conditions
2. Position face 50-100cm from camera
3. Look directly at camera
4. Check camera focus
5. Ensure good image quality in browser test

### "Unknown person detected"
1. Add user via "Add User" page
2. Upload 3-5 clear images of the person
3. Ensure good lighting in training images
4. Test again after adding user

### "Connection timeout"
1. Check ESP32-CAM is on network (ping test)
2. Power cycle ESP32-CAM
3. Check router settings
4. Move ESP32-CAM closer to WiFi router

### Video stream is slow/laggy
1. Check WiFi signal strength
2. Reduce frame rate in ESP32 firmware
3. Use lower resolution
4. Close other network-heavy applications

## Testing Checklist

- [ ] Found ESP32-CAM IP address
- [ ] Verified stream works in browser
- [ ] Ran automated test script successfully
- [ ] Tested live stream in web interface
- [ ] Face detection shows green boxes for known users
- [ ] Face detection shows red boxes for unknown users
- [ ] Successfully marked attendance
- [ ] Tested automatic recognition mode
- [ ] Attendance records appear in table
- [ ] Saved frames work correctly

## Next Steps

Once everything is working:

1. **Add More Users**
   - Navigate to "Add User" page
   - Upload multiple images per user
   - Test recognition for each user

2. **Optimize Settings**
   - Adjust recognition threshold in `src/config.py`
   - Modify detection threshold if needed
   - Configure auto-save options

3. **Production Deployment**
   - Set up Raspberry Pi auto-start service
   - Configure static IP for ESP32-CAM
   - Mount camera at optimal position
   - Test under various lighting conditions

4. **Monitor Performance**
   - Check attendance logs regularly
   - Review recognized vs unknown faces
   - Adjust thresholds based on results
   - Add more training images if needed

## Additional Resources

- **Complete Setup**: See [ESP32_CAM_GUIDE.md](ESP32_CAM_GUIDE.md)
- **Project README**: See [README.md](../README.md)
- **Troubleshooting**: See main README troubleshooting section
- **API Documentation**: See web_app.py source code

## Support

For issues or questions:
1. Review this guide and ESP32_CAM_GUIDE.md
2. Run test_esp32_cam.py for diagnostics
3. Check Serial Monitor for ESP32-CAM errors
4. Review project issues on GitHub

---

**Quick Command Reference**

```bash
# Find IP
python ip.py <MAC_ADDRESS>

# Test everything
python test_esp32_cam.py --url http://<IP>:81/stream --all

# Live demo
python demo_esp32_live.py --url http://<IP>:81/stream

# Start web app
python run.py

# Test in browser
http://<IP>:81/stream                                        # ESP32-CAM
http://localhost:3000                                         # Web UI
http://localhost:3000/video_feed/http://<IP>:81/stream      # Direct feed
```

---

Last updated: 2025-12-27
