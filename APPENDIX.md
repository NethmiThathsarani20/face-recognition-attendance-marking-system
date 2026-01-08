# Appendix

## A. Hardware Specifications

### A.1 Complete Bill of Materials

| Item | Specification | Unit Price (LKR) | Purpose |
|------|--------------|------------------|---------|
| Raspberry Pi 4 Model B | 4GB RAM | Rs. 16,500 | Edge computing host |
| MicroSD Card | 32GB Class 10 UHS-I | Rs. 2,400 | OS and storage |
| USB-C Power Supply | 5V/3A Official | Rs. 3,000 | Raspberry Pi power |
| Cooling Fan | 30mm × 30mm × 7mm, 5V | Rs. 900 | Active CPU cooling |
| ESP32-CAM Module | With OV2640 camera | Rs. 3,000 | Wireless cameras |
| LED Light Panel | 5V White, 5m roll | Rs. 2,400 | Camera illumination |
| 5V/2A Power Supply | For ESP32-CAM | Rs. 1,500 | ESP32 power |
| WiFi Router | TP-Link AC1750 | Rs. 12,000 | Network infrastructure |
| Ethernet Cable | Cat6, 3m | Rs. 900 | Wired connections |
| MicroUSB Cables | For ESP32 programming | Rs. 600 | Programming/power |
| Enclosure/Case | For Raspberry Pi | Rs. 2,400 | Protection |
| **Total** | | **Rs. 45,600** | |

**Exchange rate:** 1 USD ≈ Rs. 300 (LKR)

**Total Cost in USD:** ~$152

---

## B. Software Dependencies

### B.1 Python Package Requirements

```python
# Core dependencies
insightface==0.7.3
onnxruntime==1.12.1
opencv-python==4.7.0.72
numpy==1.24.2

# Web framework
flask==2.2.3
werkzeug==2.2.3

# Machine learning
scikit-learn==1.2.1
joblib==1.2.0

# Image processing
Pillow==9.4.0

# Optional (for CNN training)
tensorflow==2.10.0
matplotlib==3.7.0

# Export functionality
reportlab==3.6.0
openpyxl==3.0.0
pandas==1.3.0

# Development tools
ruff==0.0.260
mypy==1.1.1
pytest==7.2.2
```

### B.2 System Libraries (Debian/Ubuntu)

```bash
sudo apt update
sudo apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    libgl1 \
    libglib2.0-0 \
    python3-dev \
    build-essential \
    git
```

---

## C. System Installation Guide

### C.1 Raspberry Pi Setup

```bash
# 1. Flash Raspberry Pi OS (64-bit) to SD card
# Use Raspberry Pi Imager: https://www.raspberrypi.com/software/

# 2. Boot Raspberry Pi and connect to network

# 3. Update system
sudo apt update && sudo apt upgrade -y

# 4. Install dependencies
sudo apt install -y python3 python3-pip python3-venv git libgl1 libglib2.0-0

# 5. Clone repository
git clone https://github.com/NethmiThathsarani20/face-recognition-attendance-marking-system.git
cd face-recognition-attendance-marking-system

# 6. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 7. Install Python packages
pip install --upgrade pip
pip install -r requirements.txt

# 8. Run the system
python run.py

# Access at: http://<raspberry-pi-ip>:3000
```

### C.2 ESP32-CAM Firmware Upload

```cpp
// 1. Open Arduino IDE
// 2. Install ESP32 board support:
//    File → Preferences → Additional Board Manager URLs:
//    https://dl.espressif.com/dl/package_esp32_index.json
// 3. Tools → Board → ESP32 Arduino → AI Thinker ESP32-CAM
// 4. Edit WiFi credentials in esp32-camera.ino:

const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// 5. Connect GPIO0 to GND for programming mode
// 6. Upload sketch
// 7. Disconnect GPIO0, press reset
// 8. Open Serial Monitor (115200 baud) to see IP address
```

---

## D. API Documentation

### D.1 REST API Endpoints

**Base URL:** `http://<raspberry-pi-ip>:3000`

#### GET /

Returns the HTML dashboard page.

**Response:**
```html
HTML page
```

---

#### POST /add_user

Add a new user with face images to the system.

**Request (Form Data):**
```json
{
  "user_name": "John_Doe",
  "user_images": ["<file1.jpg>", "<file2.jpg>", ...]
}
```

**Request (JSON - Base64 encoded):**
```json
{
  "username": "John_Doe",
  "images": [
    "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAA...",
    "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAA...",
    ...
  ]
}
```

**Response:**
```json
{
  "status": "success",
  "message": "User John_Doe added successfully",
  "images_processed": 5
}
```

**Error Response:**
```json
{
  "status": "error",
  "message": "User name is required"
}
```

---

#### POST /mark_attendance

Mark attendance using a face image.

**Request (Form Data):**
```
image: <file>
```

**Request (JSON - Base64 encoded):**
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAA...",
  "camera_source": "ESP32-CAM-1"
}
```

**Response:**
```json
{
  "status": "success",
  "name": "John_Doe",
  "confidence": 0.925,
  "timestamp": "2025-12-27 09:15:30"
}
```

**Error Response:**
```json
{
  "status": "error",
  "message": "No face detected in image"
}
```

---

#### GET /model_status

Get information about the active recognition model.

**Response:**
```json
{
  "active_model": "embedding_classifier",
  "accuracy": 99.74,
  "num_users": 67,
  "total_samples": 9648,
  "last_trained": "2025-12-27"
}
```

---

#### GET /get_attendance

Get today's attendance records or filtered by date range.

**Query Parameters:**
- `start_date` (optional): Start date in YYYY-MM-DD format
- `end_date` (optional): End date in YYYY-MM-DD format

**Response:**
```json
[
  {
    "user_name": "John_Doe",
    "date": "2025-12-27",
    "time": "09:15:30",
    "confidence": 0.925
  },
  {
    "user_name": "Jane_Smith",
    "date": "2025-12-27",
    "time": "09:20:15",
    "confidence": 0.887
  }
]
```

---

#### GET /get_users

Get list of all registered users.

**Response:**
```json
[
  "John_Doe",
  "Jane_Smith",
  "Bob_Johnson"
]
```

---

#### POST /mark_attendance_camera

Mark attendance using camera capture (internal use by web interface).

**Request:**
```json
{
  "camera_source": "0",
  "auto_mode": false
}
```

**Response:**
```json
{
  "success": true,
  "user_name": "John_Doe",
  "confidence": 0.925,
  "timestamp": "2025-12-27 09:15:30",
  "captured_image": "<base64_encoded_image>"
}
```

---

#### POST /mark_attendance_upload

Mark attendance using uploaded image file (internal use by web interface).

**Request:**
```
image: <file>
```

**Response:**
```json
{
  "success": true,
  "user_name": "John_Doe",
  "confidence": 0.925,
  "timestamp": "2025-12-27 09:15:30"
}
```

---

#### GET /export_attendance_pdf

Export attendance records to PDF format.

**Query Parameters:**
- `start_date` (optional): Start date in YYYY-MM-DD format
- `end_date` (optional): End date in YYYY-MM-DD format

**Response:**
PDF file download

---

#### GET /export_attendance_excel

Export attendance records to Excel format.

**Query Parameters:**
- `start_date` (optional): Start date in YYYY-MM-DD format
- `end_date` (optional): End date in YYYY-MM-DD format

**Response:**
Excel file download

---

### D.2 Error Codes

| Status Code | Description |
|-------------|-------------|
| 200 | Success |
| 400 | Bad Request - Invalid parameters |
| 404 | Not Found - Endpoint does not exist |
| 500 | Internal Server Error |

### D.3 Authentication

Currently, the API does not require authentication. For production deployment, consider implementing:
- API key authentication
- JWT tokens
- OAuth 2.0

### D.4 Rate Limiting

No rate limiting is currently implemented. For production, consider:
- Request throttling
- IP-based rate limiting
- User-based quotas

---

## E. Training Metrics and Performance

### E.1 Embedding Classifier Performance

The embedding classifier (InsightFace + Logistic Regression) demonstrates superior performance:

**Model Specifications:**
- **Architecture:** InsightFace (buffalo_l) + Logistic Regression
- **Training Samples:** 9,648 images
- **Number of Users:** 67
- **Training Accuracy:** 99.94%
- **Validation Accuracy:** 99.74%
- **Top-3 Accuracy:** 99.90%

**Performance Metrics:**
- **Precision:** 99.74%
- **Recall:** 99.74%
- **F1-Score:** 99.74%
- **Average Confidence:** 0.925

### E.2 Training Curves

Training loss and metric curves demonstrate the superior recall performance of the embedding classifier over epochs. These curves are available in the `embedding_models/` directory:

- `embedding_precision_recall_curve.png` - Precision-Recall curve showing optimal trade-off
- `embedding_confidence_curve.png` - Confidence distribution across predictions
- `embedding_precision_confidence_curve.png` - Precision vs Confidence relationship
- `embedding_confusion_matrix.png` - Detailed confusion matrix
- `embedding_confusion_matrix_normalized.png` - Normalized confusion matrix

### E.3 Model Comparison

| Model | Train Accuracy | Validation Accuracy | Top-3 Accuracy | Use Case |
|-------|----------------|---------------------|----------------|----------|
| **Embedding Classifier** | **99.94%** | **99.74%** | **99.90%** | **Production (Default)** |
| Custom Embedding | 98.86% | 98.50% | 99.20% | Experimental |
| Lightweight CNN | 64.04% | 64.04% | 82.80% | Research baseline |

---

## F. Testing and Verification

### F.1 Postman Collection

A comprehensive Postman collection is provided for API testing. Import the collection and test all endpoints.

**Collection Features:**
- Pre-configured environment variables
- Example requests for all endpoints
- Base64 image encoding examples
- Error handling demonstrations

### F.2 Manual Testing

```bash
# Test camera capture
curl -X POST http://localhost:3000/mark_attendance_camera \
  -H "Content-Type: application/json" \
  -d '{"camera_source": "0", "auto_mode": false}'

# Get model status
curl http://localhost:3000/model_status

# Get users list
curl http://localhost:3000/get_users

# Get today's attendance
curl http://localhost:3000/get_attendance
```

### F.3 ESP32-CAM Testing

```bash
# Find ESP32-CAM IP address
python ip.py <MAC_ADDRESS>

# Test ESP32-CAM stream
python test_esp32_cam.py --url http://<ESP32_IP>:81/stream --all

# View live stream with face detection
python demo_esp32_live.py --url http://<ESP32_IP>:81/stream
```

---

## G. Deployment Considerations

### G.1 Production Checklist

- [ ] Change default ports if needed
- [ ] Enable HTTPS/SSL certificates
- [ ] Implement authentication
- [ ] Set up regular database backups
- [ ] Configure firewall rules
- [ ] Enable logging and monitoring
- [ ] Set up auto-start service
- [ ] Configure network access controls

### G.2 Performance Optimization

- Use SSD storage for faster model loading
- Allocate sufficient swap space (2GB minimum)
- Enable GPU acceleration if available
- Optimize camera resolution for network bandwidth
- Implement caching for frequently accessed data

### G.3 Maintenance

- Regular system updates
- Model retraining with new data
- Database cleanup and optimization
- Log rotation and archival
- Performance monitoring

---

## H. Troubleshooting Guide

### H.1 Common Issues

**Camera Not Detected:**
```bash
# Check camera permissions
sudo usermod -a -G video $USER
# Logout and login again

# Test camera
python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"
```

**Model Loading Errors:**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Check model files exist
ls -la embedding_models/
```

**Network Connection Issues:**
```bash
# Check network connectivity
ping <raspberry-pi-ip>

# Test API endpoint
curl http://<raspberry-pi-ip>:3000/model_status
```

### H.2 ESP32-CAM Issues

**Cannot Connect to WiFi:**
- Verify WiFi credentials in firmware
- Check WiFi signal strength
- Ensure 2.4GHz network (ESP32 doesn't support 5GHz)

**Low Frame Rate:**
- Reduce image resolution in ESP32 firmware
- Check network bandwidth
- Minimize WiFi interference

**No IP Address Displayed:**
- Open Serial Monitor at 115200 baud
- Press reset button on ESP32-CAM
- Check for error messages

---

## I. References

### I.1 Documentation
- [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md) - Comprehensive project documentation
- [README.md](README.md) - Quick start guide
- [ESP32_CAM_GUIDE.md](docs/ESP32_CAM_GUIDE.md) - ESP32-CAM setup guide
- [MODEL_TRAINING.md](docs/MODEL_TRAINING.md) - Model training documentation

### I.2 External Resources
- InsightFace: https://github.com/deepinsight/insightface
- Raspberry Pi Documentation: https://www.raspberrypi.com/documentation/
- ESP32-CAM Guide: https://randomnerdtutorials.com/esp32-cam-video-streaming-face-recognition-arduino-ide/
- Flask Documentation: https://flask.palletsprojects.com/

### I.3 Hardware Vendors
- Raspberry Pi Official Store: https://www.raspberrypi.com/products/
- ESP32-CAM Suppliers: AliExpress, Amazon, local electronics stores
- TP-Link Router: https://www.tp-link.com/

---

**Document Version:** 1.0  
**Last Updated:** 2026-01-08  
**Author:** Face Recognition Attendance System Team
