# ESP32-CAM Verification Checklist

Use this checklist to verify your ESP32-CAM is working correctly with the face recognition attendance system.

## Date: _____________    Tester: _____________

---

## Part 1: Hardware Setup ✓

- [ ] ESP32-CAM module is properly connected to power (5V, 1A+)
- [ ] ESP32-CAM LED is on (indicates power)
- [ ] WiFi credentials are configured in firmware
- [ ] Firmware uploaded successfully to ESP32-CAM
- [ ] GPIO0 disconnected from GND (normal operation mode)
- [ ] Camera lens is clean and unobstructed

**Notes:**
```
Power supply voltage: _______V
MAC Address: __:__:__:__:__:__
WiFi SSID: _________________
```

---

## Part 2: Network Connectivity ✓

- [ ] ESP32-CAM connects to WiFi successfully
- [ ] Serial Monitor shows "WiFi connected" message
- [ ] IP address obtained from DHCP or static assignment
- [ ] IP address noted: `____._____._____._____`
- [ ] Can ping ESP32-CAM from computer: `ping <IP>`
- [ ] Same network as testing computer/Raspberry Pi

**Test Commands:**
```bash
# Ping test
ping 192.168.1.100

# Result: _______ (OK / Failed)
```

---

## Part 3: Stream Accessibility ✓

- [ ] ESP32-CAM stream URL works in web browser
- [ ] URL format: `http://<IP>:81/stream`
- [ ] Live video visible in browser
- [ ] Video quality is acceptable (not too blurry)
- [ ] Frame rate is smooth (10+ FPS)
- [ ] No constant buffering or freezing

**Stream URL:**
```
http://_____________________:81/stream
```

**Test in Browser:**
- Video visible: ☐ Yes ☐ No
- Quality: ☐ Good ☐ Fair ☐ Poor
- Frame rate: ☐ Smooth ☐ Choppy

---

## Part 4: Automated Testing ✓

Run the test script:
```bash
python test_esp32_cam.py --url http://<IP>:81/stream --all
```

**Test Results:**

- [ ] **Step 1: Find ESP32 Cam IP Address**
  - Status: ☐ Pass ☐ Fail
  - IP found: _________________

- [ ] **Step 2: Test Connectivity**
  - HTTP connection: ☐ Pass ☐ Fail
  - Stream endpoint: ☐ Pass ☐ Fail

- [ ] **Step 3: Test Video Capture**
  - Video opened: ☐ Pass ☐ Fail
  - Frames captured: _____ / 10
  - Resolution: _____ x _____

- [ ] **Step 4: Test Face Detection**
  - Face manager loaded: ☐ Pass ☐ Fail
  - Faces detected: _____ frames
  - Detection working: ☐ Yes ☐ No

- [ ] **Step 5: Test Attendance Marking**
  - System initialized: ☐ Pass ☐ Fail
  - Registered users: _____ users
  - Attendance marked: ☐ Pass ☐ Fail
  - User recognized: _________________
  - Confidence score: _______

**Overall Test Result:** ☐ ALL PASS ☐ SOME FAILED

---

## Part 5: Live Demo Testing ✓

Run the live demo:
```bash
python demo_esp32_live.py --url http://<IP>:81/stream
```

**Results:**

- [ ] Video window opens successfully
- [ ] Live video feed is visible
- [ ] Face detection boxes appear
  - Green boxes for known users: ☐ Yes ☐ No
  - Red boxes for unknown persons: ☐ Yes ☐ No
- [ ] User names displayed correctly
- [ ] Confidence scores shown
- [ ] Frame counter updates
- [ ] No lag or stuttering

**Performance:**
- FPS approximate: _____ 
- Detection delay: ☐ < 1s ☐ 1-2s ☐ > 2s
- Video quality: ☐ Excellent ☐ Good ☐ Fair ☐ Poor

---

## Part 6: Web Interface Testing ✓

Start the web application:
```bash
python run.py
```

Open browser: `http://localhost:3000`

### 6.1 Initial Load

- [ ] Web page loads successfully
- [ ] Stats cards show correct numbers
- [ ] Registered users count: _____
- [ ] Today's attendance count: _____
- [ ] System initialization completes

### 6.2 Camera Configuration

- [ ] "IP Camera (URL)" option available
- [ ] Can enter ESP32 cam URL
- [ ] URL accepted: `http://<IP>:81/stream`
- [ ] "Test Camera" button works
  - Result: ☐ Pass ☐ Fail
  - Message: _____________________

### 6.3 Live Stream

- [ ] Click "Start Live Stream" button
- [ ] Live video appears in browser
- [ ] Face detection boxes visible
  - Green boxes: ☐ Yes ☐ No
  - Red boxes: ☐ Yes ☐ No
- [ ] Names and scores displayed
- [ ] Video streams smoothly
- [ ] Can stop stream with "Stop Stream" button

**Stream Performance in Browser:**
- Load time: ☐ < 2s ☐ 2-5s ☐ > 5s
- Smoothness: ☐ Smooth ☐ Some lag ☐ Very laggy
- Detection accuracy: ☐ High ☐ Medium ☐ Low

### 6.4 Manual Attendance Marking

- [ ] Position known user in front of camera
- [ ] Click "Mark Attendance" button
- [ ] Result message appears
  - Status: ☐ Success ☐ Failed
  - User: _____________________
  - Confidence: _______
  - Time: _____________________
- [ ] Captured image shown
- [ ] Attendance table updates
- [ ] Record appears in table

### 6.5 Automatic Recognition Mode

- [ ] Check "Enable Automatic Recognition Mode"
- [ ] Auto status shows "Active ✅"
- [ ] System captures frames every 3 seconds
- [ ] Attendance marked automatically
- [ ] Table updates in real-time
- [ ] Can disable auto mode
- [ ] Auto status shows "Inactive" when disabled

**Auto Mode Test:**
- Number of auto-captures in 1 minute: _____
- Successful recognitions: _____
- Failed recognitions: _____
- Accuracy: _____ %

---

## Part 7: Face Detection Quality ✓

### 7.1 Detection Scenarios

Test with different conditions:

**Lighting:**
- [ ] Good lighting (daylight/bright): ☐ Detected ☐ Not detected
- [ ] Medium lighting (indoor): ☐ Detected ☐ Not detected
- [ ] Low lighting (dim): ☐ Detected ☐ Not detected

**Distance:**
- [ ] Close (30-50cm): ☐ Detected ☐ Not detected
- [ ] Optimal (50-100cm): ☐ Detected ☐ Not detected
- [ ] Far (100-150cm): ☐ Detected ☐ Not detected

**Angle:**
- [ ] Front-facing: ☐ Detected ☐ Not detected
- [ ] Slight angle (15-30°): ☐ Detected ☐ Not detected
- [ ] Side profile (45°+): ☐ Detected ☐ Not detected

**Multiple Faces:**
- [ ] Single person: ☐ Detected ☐ Not detected
- [ ] Two people: ☐ Both detected ☐ Some detected ☐ None
- [ ] Three+ people: ☐ All detected ☐ Some detected ☐ None

### 7.2 Recognition Accuracy

Test with registered users:

**User 1:** _________________
- [ ] Recognized correctly
- [ ] Confidence score: _______
- [ ] False negative: ☐ Yes ☐ No

**User 2:** _________________
- [ ] Recognized correctly
- [ ] Confidence score: _______
- [ ] False negative: ☐ Yes ☐ No

**User 3:** _________________
- [ ] Recognized correctly
- [ ] Confidence score: _______
- [ ] False negative: ☐ Yes ☐ No

**Unknown Person:**
- [ ] Correctly marked as "Unknown"
- [ ] Not falsely recognized as registered user
- [ ] Shown with red box

---

## Part 8: Attendance Records ✓

- [ ] Attendance records saved to JSON file
- [ ] File location: `attendance_records/attendance_YYYY-MM-DD.json`
- [ ] Records include:
  - [ ] User name
  - [ ] Date
  - [ ] Time
  - [ ] Confidence score
- [ ] Records appear in web interface table
- [ ] Can refresh attendance table
- [ ] Export to PDF works: ☐ Yes ☐ No ☐ Not tested
- [ ] Export to Excel works: ☐ Yes ☐ No ☐ Not tested

**Sample Record:**
```json
{
  "user_name": "_________________",
  "date": "____-__-__",
  "time": "__:__:__",
  "confidence": _______
}
```

---

## Part 9: Error Handling ✓

Test error scenarios:

- [ ] **Camera disconnected during stream**
  - Behavior: _____________________
  - Error message shown: ☐ Yes ☐ No
  - Recovers when reconnected: ☐ Yes ☐ No

- [ ] **Wrong camera URL entered**
  - Error message: _____________________
  - User-friendly: ☐ Yes ☐ No

- [ ] **No face in frame**
  - Message: _____________________
  - Appropriate: ☐ Yes ☐ No

- [ ] **Network interruption**
  - System behavior: _____________________
  - Recovers automatically: ☐ Yes ☐ No

---

## Part 10: Performance Metrics ✓

Measure system performance:

**Response Times:**
- Single frame capture: _____ seconds
- Face detection: _____ seconds
- Recognition: _____ seconds
- Attendance marking: _____ seconds
- Total (capture to save): _____ seconds

**Resource Usage:**
- CPU usage: _____ %
- Memory usage: _____ MB
- Network bandwidth: _____ Mbps

**Reliability:**
- Test duration: _____ minutes
- Total frames processed: _____
- Successful detections: _____
- Failed detections: _____
- Success rate: _____ %

---

## Final Assessment

### Overall Status

- [ ] ✅ **PASS** - All critical features working
- [ ] ⚠️ **PARTIAL** - Some issues but usable
- [ ] ❌ **FAIL** - Critical issues prevent use

### Critical Issues (if any)

1. _________________________________________________
2. _________________________________________________
3. _________________________________________________

### Recommendations

1. _________________________________________________
2. _________________________________________________
3. _________________________________________________

### Sign-off

**Tested by:** _________________
**Date:** _________________
**Status:** ☐ Approved ☐ Needs work
**Next review date:** _________________

---

## Quick Reference

**ESP32-CAM IP:** `____._____._____._____`
**Stream URL:** `http://____._____._____._____:81/stream`
**Web Interface:** `http://localhost:3000`
**Total Users:** _____
**System Version:** _____

---

**For support and troubleshooting:**
- See `docs/ESP32_CAM_GUIDE.md`
- See `docs/ESP32_QUICK_START.md`
- Run `python test_esp32_cam.py --help`
