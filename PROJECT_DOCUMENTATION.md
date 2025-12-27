# Face Recognition Attendance System - Complete Project Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [IoT Devices and Hardware](#iot-devices-and-hardware)
3. [System Architecture](#system-architecture)
4. [Dataset Details](#dataset-details)
5. [Models and Algorithms](#models-and-algorithms)
6. [UI Components](#ui-components)
7. [Features and Capabilities](#features-and-capabilities)
8. [Workflow and Processes](#workflow-and-processes)

---

## 1. Project Overview

The Face Recognition Attendance System is an IoT-enabled, edge-assisted attendance solution that combines edge computing (Raspberry Pi) with IoT camera devices (ESP32-CAM) to deliver a production-grade face recognition system. The system uses InsightFace for detection and embedding generation, paired with a lightweight embedding-based classifier for exceptional accuracy.

### Key Highlights
- **Production Accuracy**: 99.74% validation accuracy on 67 users
- **Edge + Cloud Architecture**: Raspberry Pi for edge processing, GitHub Actions for cloud training
- **IoT Integration**: ESP32-CAM for wireless camera streaming
- **Real-time Processing**: Live face detection and attendance marking
- **Scalable Design**: Supports multiple camera inputs and remote access

---

## 2. IoT Devices and Hardware

### 2.1 Raspberry Pi (Edge Computing Host)

#### Why Raspberry Pi?

The Raspberry Pi serves as the **edge computing host** for several critical reasons:

1. **Computational Power**
   - Sufficient processing capability to run InsightFace models
   - Can handle real-time face detection and recognition
   - Supports Python and all required ML libraries
   - Memory adequate for embedding storage and matching (4GB+ recommended)

2. **Network Orchestration**
   - Acts as central hub for ESP32-CAM devices
   - Manages WiFi network connectivity
   - Provides stable web server hosting
   - Handles multiple concurrent camera streams

3. **Edge Processing Benefits**
   - Reduces cloud dependency for real-time recognition
   - Lower latency for immediate attendance marking
   - Privacy preservation (face data stays local)
   - Works without constant internet connection

4. **Storage and Persistence**
   - Stores user face database locally
   - Maintains attendance records in JSON format
   - Caches face embeddings for fast matching
   - Hosts the web application and UI

5. **Cost-Effective Solution**
   - Affordable hardware (~$35-$75)
   - Low power consumption
   - Compact form factor
   - Easy to deploy and maintain

6. **Connectivity**
   - Built-in WiFi and Ethernet
   - USB ports for local cameras
   - GPIO pins for future sensor integration
   - SSH access for remote management

#### Raspberry Pi Process Flow

```
┌─────────────────────────────────────────────────────────┐
│                    RASPBERRY PI                         │
│                  (Edge Computing Host)                  │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  1. WEB SERVER (Flask)                                  │
│     └─> Serves UI on port 3000                          │
│     └─> Handles user requests                           │
│     └─> Manages camera connections                      │
│                                                          │
│  2. FACE RECOGNITION ENGINE (InsightFace)               │
│     └─> Receives frames from ESP32-CAM                  │
│     └─> Detects and aligns faces                        │
│     └─> Generates 512-dim embeddings                    │
│     └─> Matches against stored user embeddings          │
│                                                          │
│  3. DATABASE MANAGEMENT                                 │
│     └─> Stores user images (database/ folder)           │
│     └─> Caches face embeddings (embeddings/ folder)     │
│     └─> Records attendance (attendance_records/)        │
│                                                          │
│  4. DEVICE COORDINATION                                 │
│     └─> Connects to ESP32-CAM via HTTP stream           │
│     └─> Processes video frames in real-time             │
│     └─> Sends results back to UI                        │
│                                                          │
│  5. CLOUD SYNC (Optional)                               │
│     └─> Pushes new user images to GitHub                │
│     └─> Triggers cloud training via GitHub Actions      │
│     └─> Pulls trained models back                       │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

#### Raspberry Pi Setup Requirements

- **Model**: Raspberry Pi 3B+ or newer (Pi 4 4GB+ recommended)
- **OS**: Raspberry Pi OS (64-bit Bullseye or later)
- **Python**: 3.8+ (3.12 recommended)
- **Network**: WiFi or Ethernet connection
- **Storage**: 16GB+ microSD card (32GB recommended)
- **Power**: 5V 3A USB-C power supply

#### Raspberry Pi Key Responsibilities

1. **Host Web Application**: Runs Flask server accessible via browser
2. **Process Face Recognition**: Executes InsightFace detection and matching
3. **Manage Data**: Stores images, embeddings, and attendance records
4. **Coordinate Devices**: Manages ESP32-CAM streams and local cameras
5. **Enable Remote Access**: Provides network-accessible interface
6. **Sync with Cloud**: Optional GitHub integration for model training

---

### 2.2 ESP32-CAM (IoT Camera Module)

#### What is ESP32-CAM?

ESP32-CAM is a low-cost IoT camera module featuring:
- ESP32 microcontroller with WiFi
- OV2640 camera sensor (2MP)
- Compact size (27mm × 40mm)
- WiFi streaming capability
- Low power consumption

#### Why ESP32-CAM?

1. **Wireless Operation**
   - No physical cable connection required
   - WiFi streaming to Raspberry Pi
   - Flexible placement anywhere in WiFi range
   - Easy to relocate or add multiple units

2. **Cost-Effective**
   - Very affordable (~$5-$10 per unit)
   - No need for expensive IP cameras
   - Scalable (add multiple cameras easily)

3. **IoT Integration**
   - Built-in WiFi connectivity
   - HTTP/MJPEG streaming
   - Lightweight firmware
   - Real-time video transmission

4. **Dedicated Camera**
   - Independent operation
   - Doesn't consume Raspberry Pi USB ports
   - Can be positioned optimally for face capture
   - Multiple ESP32-CAM units can be deployed

5. **Sufficient Quality**
   - 2MP resolution adequate for face recognition
   - Adjustable image quality
   - Good low-light performance
   - JPEG compression for network efficiency

#### ESP32-CAM Process Flow

```
┌────────────────────────────────────────────────────┐
│                   ESP32-CAM                        │
│              (IoT Camera Module)                   │
├────────────────────────────────────────────────────┤
│                                                     │
│  1. INITIALIZATION                                 │
│     └─> Connect to WiFi network                    │
│     └─> Obtain IP address via DHCP                 │
│     └─> Initialize OV2640 camera sensor            │
│     └─> Start HTTP server on port 81               │
│                                                     │
│  2. VIDEO CAPTURE                                  │
│     └─> Continuously capture frames from camera    │
│     └─> Compress as JPEG (quality: 12-63)          │
│     └─> Buffer management for smooth streaming     │
│                                                     │
│  3. HTTP STREAMING                                 │
│     └─> Serve MJPEG stream at /stream endpoint     │
│     └─> Stream URL: http://<ESP32-IP>:81/stream    │
│     └─> Handle multiple concurrent connections     │
│                                                     │
│  4. FRAME TRANSMISSION                             │
│     └─> Send frames to Raspberry Pi via HTTP       │
│     └─> Typical frame rate: 10-15 FPS              │
│     └─> Image size: 240×240 to 800×600 pixels      │
│                                                     │
│  5. STATUS MONITORING                              │
│     └─> LED indicator for operation status         │
│     └─> Serial output for debugging                │
│     └─> Web interface for camera settings          │
│                                                     │
└────────────────────────────────────────────────────┘
```

#### ESP32-CAM Hardware Connections

- **Camera**: OV2640 sensor connected via FPC cable
- **Power**: 5V via VCC/GND pins (3.3V regulator onboard)
- **Programming**: GPIO0 to GND for flash mode
- **LED**: Built-in flash LED (GPIO4)
- **WiFi**: Built-in antenna or external antenna connector

#### ESP32-CAM Setup Process

1. **Upload Firmware**
   - Use Arduino IDE with ESP32 board support
   - Configure WiFi SSID and password in code
   - Set camera resolution and quality parameters
   - Flash firmware via FTDI/USB-to-TTL adapter

2. **Network Configuration**
   - Connect GPIO0 to GND for programming mode
   - Upload sketch from `esp32-camera/` directory
   - Disconnect GPIO0 and reset
   - Camera connects to WiFi and displays IP in Serial Monitor

3. **Verify Connection**
   - Note IP address (e.g., 10.74.63.131)
   - Access stream: `http://10.74.63.131:81/stream`
   - Test in web browser to verify video feed
   - Configure in Raspberry Pi web interface

#### ESP32-CAM Key Responsibilities

1. **Capture Video**: Continuously capture frames from OV2640 sensor
2. **Stream via WiFi**: Transmit MJPEG stream over HTTP
3. **Network Connectivity**: Maintain stable WiFi connection
4. **Quality Management**: Balance resolution vs. network bandwidth
5. **Status Indication**: Provide visual feedback via LED

---

### 2.3 IoT System Integration

#### Complete Hardware Architecture

```
┌──────────────────────┐
│   ESP32-CAM #1       │ ──┐
│   (Entry Point)      │   │
└──────────────────────┘   │
                           │
┌──────────────────────┐   │    WiFi Network
│   ESP32-CAM #2       │ ──┤    ═══════════
│   (Exit Point)       │   │
└──────────────────────┘   │         │
                           │         │
┌──────────────────────┐   │         │
│   ESP32-CAM #N       │ ──┘         │
│   (Office Area)      │             │
└──────────────────────┘             │
                                     │
                                     ▼
                          ┌─────────────────────┐
                          │   Raspberry Pi      │
                          │   (Edge Host)       │
                          │                     │
                          │  • Flask Web Server │
                          │  • InsightFace      │
                          │  • Database         │
                          │  • Attendance Logs  │
                          └─────────────────────┘
                                     │
                                     │ Ethernet/WiFi
                                     │
                                     ▼
                          ┌─────────────────────┐
                          │   Router/Network    │
                          └─────────────────────┘
                                     │
                                     │ Internet
                                     │
                                     ▼
                          ┌─────────────────────┐
                          │   GitHub Cloud      │
                          │   (Model Training)  │
                          └─────────────────────┘
```

#### Communication Flow

1. **ESP32-CAM → Raspberry Pi**
   - Protocol: HTTP/MJPEG
   - Port: 81
   - Data: Video frames (JPEG compressed)
   - Direction: One-way (camera to Pi)

2. **Raspberry Pi → User Browser**
   - Protocol: HTTP/WebSocket
   - Port: 3000
   - Data: Web UI, API responses, live video
   - Direction: Bidirectional

3. **Raspberry Pi → GitHub**
   - Protocol: HTTPS/Git
   - Data: Database images, trained models
   - Direction: Bidirectional (push images, pull models)

#### Network Requirements

- **Local Network**: WiFi or Ethernet for all devices
- **IP Addressing**: Static IPs recommended for ESP32-CAM
- **Bandwidth**: Minimum 2 Mbps per camera for smooth streaming
- **Latency**: <50ms preferred for real-time recognition
- **Internet**: Optional, only needed for cloud training

---

## 3. System Architecture

### 3.1 Three-Tier Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  PRESENTATION TIER                      │
│             (Web UI - Browser Access)                   │
│                                                          │
│  • Dashboard with real-time video feed                  │
│  • User management interface                            │
│  • Attendance records viewer                            │
│  • Model status and switching                           │
│                                                          │
└────────────────────┬────────────────────────────────────┘
                     │ HTTP (Port 3000)
                     ▼
┌─────────────────────────────────────────────────────────┐
│                  APPLICATION TIER                       │
│            (Raspberry Pi - Edge Host)                   │
│                                                          │
│  ┌─────────────────────────────────────────────┐        │
│  │  Flask Web Application (web_app.py)         │        │
│  │  • Route handling                            │        │
│  │  • Request processing                        │        │
│  │  • Response generation                       │        │
│  └─────────────────────────────────────────────┘        │
│                                                          │
│  ┌─────────────────────────────────────────────┐        │
│  │  Face Manager (face_manager.py)             │        │
│  │  • InsightFace integration                   │        │
│  │  • Face detection & alignment                │        │
│  │  • Embedding generation (512-dim)            │        │
│  │  • Face matching                             │        │
│  └─────────────────────────────────────────────┘        │
│                                                          │
│  ┌─────────────────────────────────────────────┐        │
│  │  Attendance System (attendance_system.py)   │        │
│  │  • Camera integration                        │        │
│  │  • Recognition logic                         │        │
│  │  • Attendance recording                      │        │
│  │  • Image saving                              │        │
│  └─────────────────────────────────────────────┘        │
│                                                          │
└────────────────────┬────────────────────────────────────┘
                     │ File I/O
                     ▼
┌─────────────────────────────────────────────────────────┐
│                    DATA TIER                            │
│              (Local Storage on Pi)                      │
│                                                          │
│  • database/ - User face images (30MB)                  │
│  • embeddings/ - Cached face embeddings                 │
│  • attendance_records/ - Daily JSON logs                │
│  • embedding_models/ - Trained classifier (2.3MB)       │
│  • cnn_models/ - Optional CNN model (3.2MB)             │
│  • custom_embedding_models/ - Custom embeddings (3.2MB) │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### 3.2 Edge + Cloud Hybrid Architecture

The system employs an **edge-first, cloud-assisted** architecture:

#### Edge Processing (Raspberry Pi)
- **Real-time Operations**: Face detection, recognition, attendance marking
- **Data Storage**: User images, embeddings, attendance records
- **Web Hosting**: User interface and API endpoints
- **Device Management**: ESP32-CAM stream processing

#### Cloud Processing (GitHub Actions)
- **Model Training**: Heavy computational tasks
- **CI/CD Pipeline**: Automated training on dataset updates
- **Model Distribution**: Push trained models back to repository
- **Backup**: Version control for code and models

#### Why This Architecture?

1. **Low Latency**: Face recognition happens instantly at the edge
2. **Privacy**: Face data stays local on Raspberry Pi
3. **Reliability**: Works offline, no cloud dependency for attendance
4. **Scalability**: Cloud handles compute-intensive training
5. **Cost-Effective**: No continuous cloud computing costs
6. **Easy Updates**: Git pull to update models

---

## 4. Dataset Details

### 4.1 Dataset Specifications

#### Current Dataset Composition

- **Location**: `database/` directory on Raspberry Pi
- **Total Users**: 67 individuals
- **Total Images**: 1,595 base images (before augmentation/balancing)
- **Total Samples**: 9,648 samples after balancing for training
- **Image Format**: JPEG with 95% quality
- **Image Size**: 240×240 pixels (optimized for ESP32-CAM resolution)
- **Color**: RGB (3 channels)
- **File Naming**: Timestamp-based for automatic organization

#### Dataset Structure

```
database/
├── Abdullah_Gul/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ... (multiple images per user)
├── Adithya/
│   ├── captured_20231215_143022.jpg
│   └── ... 
├── Alvaro_Uribe/
├── Amelie_Mauresmo/
├── Andre_Agassi/
├── ... (67 users total)
└── Yukiya_Amano/
```

### 4.2 Dataset Characteristics

#### Images Per User
- **Minimum**: 1 image per user
- **Recommended**: 5-10 images per user for better accuracy
- **Average**: ~24 images per user (1,595 ÷ 67)
- **Variation**: Some users have more images for difficult lighting/angles

#### Image Quality Requirements

1. **Face Visibility**
   - Face should be clearly visible
   - Minimal occlusion (no masks, sunglasses preferred to be off)
   - Front-facing or slight angle (±30 degrees optimal)

2. **Lighting Conditions**
   - Well-lit environments preferred
   - Avoid extreme shadows or backlighting
   - Consistent lighting for better recognition

3. **Resolution**
   - 240×240 pixels minimum (matches ESP32-CAM output)
   - Higher resolution images automatically resized
   - Maintains aspect ratio during preprocessing

4. **Image Diversity**
   - Different facial expressions beneficial
   - Varying angles (slight variations)
   - Different lighting conditions
   - With/without glasses (if applicable)

### 4.3 Data Acquisition Process

#### Method 1: Web Interface Upload
1. Navigate to "Add User" page
2. Enter user name
3. Upload multiple images (3-5 recommended)
4. System automatically saves to `database/username/`
5. Embeddings generated and cached

#### Method 2: ESP32-CAM Capture
1. User stands in front of ESP32-CAM
2. Admin triggers capture via web interface
3. Multiple frames captured automatically
4. Best quality frames saved to database
5. Immediate embedding generation

#### Method 3: Video Processing
1. Upload video file via web interface
2. System extracts frames at intervals
3. Face detection on each frame
4. High-quality detections saved
5. Reduces manual photo collection

#### Method 4: Direct File Copy
1. Copy images to `database/username/` on Raspberry Pi
2. SSH or direct SD card access
3. Run sync script: `./scripts/edge_sync.sh "Add new user images"`
4. Triggers cloud training automatically

### 4.4 Data Preprocessing

#### Automatic Preprocessing Pipeline

1. **Face Detection**
   - InsightFace detects faces in uploaded images
   - Rejects images without detectable faces
   - Multiple faces handled (saves each separately)

2. **Face Alignment**
   - InsightFace aligns faces to canonical position
   - Normalizes rotation and scale
   - Ensures consistent face orientation

3. **Face Cropping**
   - Extracts face region with margin
   - Resizes to 240×240 pixels
   - Maintains aspect ratio

4. **Quality Assessment**
   - Blur detection (rejects poor quality)
   - Brightness normalization
   - Contrast enhancement if needed

5. **Embedding Generation**
   - 512-dimensional embedding extracted
   - L2-normalized for cosine similarity
   - Cached in `embeddings/` directory

### 4.5 Data Balancing

For model training, the dataset is balanced:

#### Oversampling Strategy
```python
# Find class with maximum samples
max_samples = max(samples per class)

# For each class with fewer samples:
#   - Duplicate random samples (with replacement)
#   - Until class has max_samples

Result: 9,648 balanced samples for training
```

#### Why Balancing?
- Prevents model bias toward users with more images
- Ensures equal learning for all users
- Improves recognition accuracy for underrepresented users
- Standard practice in classification tasks

### 4.6 Data Augmentation

For CNN training only (not for embedding classifier):

- **Horizontal Flip**: 50% probability
- **Rotation**: ±15 degrees random
- **Zoom**: 0.9-1.1x random
- **Brightness**: ±20% random
- **Contrast**: ±20% random

Augmentation increases training data variability without manual collection.

### 4.7 Train/Validation Split

- **Training Set**: 80% of balanced dataset (7,718 samples)
- **Validation Set**: 20% of balanced dataset (1,930 samples)
- **Stratified Split**: Maintains class distribution in both sets
- **Random Seed**: Fixed for reproducibility

---

## 5. Models and Algorithms

### 5.1 Production Model: Embedding Classifier

#### Architecture Overview

The production model combines **InsightFace** (pre-trained) with **Logistic Regression** classifier:

```
Input Image (240×240×3)
        │
        ▼
┌──────────────────────┐
│   InsightFace        │
│   buffalo_l Model    │
│   (Pre-trained)      │
│                      │
│  • Face Detection    │
│  • Face Alignment    │
│  • Feature Extract   │
└──────────────────────┘
        │
        ▼
   512-D Embedding
   (L2-Normalized)
        │
        ▼
┌──────────────────────┐
│  Logistic Regression │
│  Classifier          │
│  (Trained on Dataset)│
│                      │
│  • OneVsRest         │
│  • L2 Regularization │
│  • SAGA Solver       │
└──────────────────────┘
        │
        ▼
  User Prediction
  + Confidence Score
```

#### Component Details

**1. InsightFace (Feature Extraction)**
- **Model**: buffalo_l (large variant)
- **Input**: 240×240 RGB image
- **Output**: 512-dimensional embedding vector
- **Training**: Pre-trained on massive face dataset
- **Frozen**: No retraining, used as-is
- **Purpose**: Extract discriminative facial features

**2. Logistic Regression (Classification)**
- **Type**: Multinomial Logistic Regression
- **Strategy**: OneVsRestClassifier
- **Solver**: SAGA (scalable, handles L1/L2 penalty)
- **Regularization**: L2 penalty (C=1.0)
- **Max Iterations**: 2000
- **Training**: Trains on 512-D embeddings from InsightFace
- **Purpose**: Map embeddings to user identities

#### Why Embedding Classifier for This Project?

**1. Exceptional Accuracy**
- **99.74% validation accuracy** on 67 users
- **99.90% top-3 accuracy** (correct user in top 3 predictions)
- Far superior to custom CNN (64.04%) on same dataset
- Comparable to custom embedding model (98.86%) but faster

**2. Fast Training**
- Training time: **~30 seconds**
- Only classifier trains (512-D → 67 classes)
- InsightFace embeddings pre-computed
- CNN requires ~32 minutes for comparison

**3. Robust Pre-trained Features**
- InsightFace trained on millions of faces
- Generalizes extremely well to new faces
- Handles varying lighting, angles, expressions
- Production-proven in real-world applications

**4. Low Resource Requirements**
- **Model Size**: ~500 KB (classifier only)
- InsightFace: ~50 MB (one-time download)
- Suitable for Raspberry Pi deployment
- Fast inference (<100ms per face)

**5. Simple Architecture**
- Only one component needs training
- Fewer hyperparameters to tune
- Easy to update with new users (retrain classifier only)
- Straightforward debugging

**6. Proven Technology**
- InsightFace: Industry-standard face recognition
- Logistic Regression: Well-understood, reliable
- Combination widely used in production systems
- Extensive community support

**7. Optimal for Small-Medium Datasets**
- Works excellent with 67 users
- Doesn't require millions of training samples
- Transfer learning leverages pre-trained knowledge
- Scalable to hundreds of users

**8. Real-World Deployment Benefits**
- Reliable daily operation on Raspberry Pi
- Handles ESP32-CAM image quality variations
- Consistent performance across conditions
- Easy to add new users without full retraining

#### Training Process

```bash
# Train embedding classifier
python train.py --only embedding --epochs 30 --validation-split 0.2

# Training steps:
# 1. Load images from database/
# 2. Balance dataset (oversample to 9,648 samples)
# 3. Generate InsightFace embeddings (512-D)
# 4. Split 80/20 train/validation
# 5. Train Logistic Regression on embeddings
# 6. Evaluate on validation set
# 7. Save classifier to embedding_models/
# 8. Generate evaluation charts
```

#### Model Files

- `embedding_classifier.joblib` - Trained LogisticRegression (500 KB)
- `label_encoder.pkl` - Maps class indices to user names
- `training_log.json` - Metadata and metrics

#### Inference Process

```python
# At runtime:
1. Capture image from ESP32-CAM or upload
2. Detect face with InsightFace
3. Generate 512-D embedding
4. Pass to Logistic Regression classifier
5. Get prediction + confidence score
6. If confidence > threshold (0.4):
   - Mark attendance for recognized user
   - Save image to user's folder
7. Else:
   - Mark as "Unknown"
```

---

### 5.2 Experimental Model: Lightweight CNN

#### Purpose
Research and comparison to demonstrate value of transfer learning.

#### Architecture

```
Input (240×240×3)
    ↓
Data Augmentation Layer
    ↓
SeparableConv2D(32, 3×3) → BatchNorm → ReLU → MaxPool → Dropout(0.2)
    ↓
SeparableConv2D(64, 3×3) → BatchNorm → ReLU → MaxPool → Dropout(0.3)
    ↓
SeparableConv2D(128, 3×3) → BatchNorm → ReLU → MaxPool → Dropout(0.4)
    ↓
GlobalAveragePooling2D
    ↓
Dense(256) → BatchNorm → ReLU → Dropout(0.5)
    ↓
Dense(128) → ReLU → Dropout(0.4)
    ↓
Dense(67, softmax)
    ↓
Output (User Prediction)
```

#### Features

- **End-to-End Learning**: Learns features from scratch
- **Separable Convolutions**: Reduced parameters vs. standard Conv2D
- **Batch Normalization**: Stabilizes training
- **Dropout**: Prevents overfitting (rates: 0.2-0.5)
- **Data Augmentation**: Increases training diversity

#### Performance

- **Validation Accuracy**: 64.04%
- **Top-3 Accuracy**: 82.80%
- **Training Time**: ~32 minutes
- **Model Size**: ~2 MB

#### Use Case
- Educational/research purposes
- Demonstrates challenges of training from scratch
- Shows value of pre-trained models
- Not recommended for production

---

### 5.3 Experimental Model: Custom Embedding

#### Purpose
Explore custom embedding spaces without InsightFace dependency.

#### Architecture

```
Input (240×240×3)
    ↓
SeparableConv2D(32) → BatchNorm → MaxPool
    ↓
SeparableConv2D(64) → BatchNorm → MaxPool
    ↓
SeparableConv2D(128) → BatchNorm → GlobalAveragePooling
    ↓
Dense(256) → Dropout(0.4)
    ↓
Dense(128, linear) [Embedding Layer]
    ↓
L2 Normalization
    ↓
Dense(67, softmax) [Training Only]

# Inference uses cosine similarity to class centroids
```

#### Features

- **Custom Embeddings**: Learns 128-D embedding space
- **L2 Normalization**: Enables cosine similarity matching
- **Centroid-Based**: Stores average embedding per class
- **Independent**: No InsightFace dependency

#### Performance

- **Validation Accuracy**: 98.86%
- **Training Time**: ~2-3 minutes
- **Model Size**: ~1.5 MB
- **Embedding Dimension**: 128

#### Use Case
- Research on custom metric learning
- Demonstrates embedding-based recognition
- Good middle ground between CNN and production model
- Not recommended for production (InsightFace better)

---

### 5.4 Model Comparison Summary

| Metric | Embedding Classifier | Custom Embedding | Lightweight CNN |
|--------|---------------------|------------------|-----------------|
| **Validation Accuracy** | **99.74%** ⭐ | 98.86% | 64.04% |
| **Top-3 Accuracy** | **99.90%** ⭐ | N/A | 82.80% |
| **Training Time** | **~30 sec** ⭐ | ~2-3 min | ~32 min |
| **Model Size** | 500 KB ⭐ | 1.5 MB | 2 MB |
| **Dependencies** | InsightFace | None | None |
| **Production Ready** | **YES** ⭐ | No | No |
| **Use Case** | **Production** | Research | Research |

⭐ = Best in category

#### Final Recommendation: Embedding Classifier

The **Embedding Classifier (InsightFace + Logistic Regression)** is the clear choice for production because:

1. ✅ Highest accuracy (99.74%)
2. ✅ Fastest training (30 seconds)
3. ✅ Proven reliability
4. ✅ Optimal for Raspberry Pi
5. ✅ Industry-standard architecture
6. ✅ Easy to maintain and update

---

## 6. UI Components

### 6.1 Web Interface Overview

The system provides a professional Flask-based web interface optimized for Raspberry Pi:

**Access URL**: `http://<raspberry-pi-ip>:3000`
**Example**: `http://10.74.63.231:3000`

### 6.2 Dashboard (Main Page)

**File**: `templates/index.html`
**Route**: `/` (GET)

#### Components

1. **Navigation Bar**
   - Logo/Title: "Face Recognition Attendance System"
   - Links: Dashboard | Add User | Mark Attendance | View Attendance
   - Model Status indicator
   - Active page highlighting

2. **Live Video Feed Section**
   - Camera preview (ESP32-CAM or local)
   - Start/Stop camera buttons
   - Camera source selector (dropdown)
   - Real-time face detection overlay

3. **Quick Actions Panel**
   - "Mark Attendance" button (primary action)
   - "Add New User" button
   - "View Today's Attendance" button

4. **Recent Activity Widget**
   - Last 5 attendance records
   - User name, timestamp, confidence score
   - Auto-refresh every 30 seconds

5. **Statistics Panel**
   - Total registered users: 67
   - Today's attendance count
   - System uptime
   - Active model indicator

#### Process Flow

```
User accesses http://10.74.63.231:3000
    ↓
Flask renders index.html
    ↓
JavaScript loads camera stream
    ↓
User selects ESP32-CAM from dropdown
    ↓
Stream URL: http://10.74.63.131:81/stream
    ↓
Live video displayed in feed section
    ↓
User clicks "Mark Attendance"
    ↓
Current frame captured
    ↓
Sent to /mark_attendance endpoint
    ↓
Face detection and recognition
    ↓
Result displayed in popup
    ↓
Attendance record saved
    ↓
Recent activity updated
```

---

### 6.3 Add User Page

**File**: `templates/add_user.html`
**Route**: `/add_user` (GET/POST)

#### Components

1. **User Information Form**
   - Name input field (required)
   - Validation: alphanumeric and underscores only

2. **Image Upload Section**
   - Multiple file upload (accepts .jpg, .jpeg, .png)
   - Drag-and-drop support
   - Image preview before upload
   - Recommended: 3-5 images per user

3. **Alternative: Camera Capture**
   - Live camera preview
   - "Capture Photo" button
   - Multiple captures supported
   - Automatic face detection check

4. **Progress Indicator**
   - Upload progress bar
   - Processing status messages
   - Success/error notifications

5. **Preview Gallery**
   - Thumbnails of uploaded images
   - Remove button for each image
   - Face detection status (✓ or ✗)

#### Process Flow

```
User navigates to Add User page
    ↓
Enters user name: "John_Doe"
    ↓
Option 1: Upload Images
    ├─> Selects 5 images from computer
    ├─> Images uploaded to server
    ├─> Server validates images
    ├─> Detects faces in each image
    └─> Saves to database/John_Doe/

Option 2: Capture from Camera
    ├─> Opens camera feed (ESP32-CAM)
    ├─> Clicks "Capture" 5 times
    ├─> Each capture saved instantly
    └─> Faces detected in real-time

After upload/capture:
    ↓
InsightFace generates embeddings
    ↓
Embeddings saved to embeddings/John_Doe.pkl
    ↓
Success message: "User John_Doe added!"
    ↓
User can mark attendance immediately
```

---

### 6.4 Mark Attendance Page

**File**: `templates/index.html` (integrated)
**Route**: `/mark_attendance` (POST)

#### Components

1. **Camera Selection**
   - Local Camera: Dropdown (Camera 0, Camera 1, etc.)
   - IP Camera: Text input for URL
   - ESP32-CAM: Pre-configured URL option

2. **Live Preview**
   - Real-time camera feed
   - Face detection boxes overlaid
   - Recognition results displayed

3. **Capture Button**
   - Large, prominent "Mark Attendance" button
   - Keyboard shortcut: Spacebar

4. **Recognition Result Display**
   - User name (if recognized)
   - Confidence score (percentage)
   - Status: Success/Unknown
   - Timestamp

5. **Attendance Log**
   - Real-time list of marked attendance
   - Automatic scrolling
   - Color-coded (green=success, red=unknown)

#### Process Flow

```
User selects camera source
    ↓
ESP32-CAM: http://10.74.63.131:81/stream
    ↓
Live video appears in preview
    ↓
User stands in front of camera
    ↓
Clicks "Mark Attendance"
    ↓
POST request to /mark_attendance
    ├─> Current frame captured
    ├─> Sent to backend
    └─> Base64 encoded image data

Backend processing:
    ↓
1. Decode image
2. InsightFace detects face
3. Extract 512-D embedding
4. Match against database (cosine similarity)
5. If similarity > 0.6:
   ├─> Recognized as "John_Doe"
   ├─> Confidence: 92.5%
   ├─> Save to attendance_records/attendance_2025-12-27.json
   ├─> Save image to database/John_Doe/captured_20251227_101530.jpg
   └─> Return success response
6. Else:
   └─> Return "Unknown person"

Frontend displays result:
    ↓
"Attendance marked for John_Doe (92.5% confident)"
    ↓
Green success notification
    ↓
Attendance log updates
```

---

### 6.5 View Attendance Page

**Route**: `/view_attendance` (GET)

#### Components

1. **Date Selector**
   - Calendar widget
   - Quick links: Today, Yesterday, This Week
   - Date range picker for custom periods

2. **Attendance Table**
   - Columns: Name, Time, Confidence, Image
   - Sortable by each column
   - Search/filter functionality

3. **Export Options**
   - Download as CSV
   - Download as PDF
   - Print friendly view

4. **Statistics**
   - Total attendance for selected date
   - Attendance rate graph
   - Peak hours chart

#### Process Flow

```
User selects date: 2025-12-27
    ↓
GET /view_attendance?date=2025-12-27
    ↓
Server reads attendance_records/attendance_2025-12-27.json
    ↓
Parse JSON:
[
  {
    "name": "John_Doe",
    "time": "09:15:30",
    "confidence": 0.925,
    "image": "database/John_Doe/captured_20251227_091530.jpg"
  },
  ...
]
    ↓
Render HTML table
    ↓
User can:
    ├─> Sort by time
    ├─> Filter by name
    ├─> Export to CSV
    └─> View saved images
```

---

### 6.6 UI Styling and Optimization

#### Design Principles

1. **Raspberry Pi Optimized**
   - Minimal animations (reduced GPU load)
   - Solid colors instead of gradients
   - Lightweight images
   - Efficient JavaScript

2. **Responsive Design**
   - Works on desktop, tablet, mobile
   - Flexible layouts
   - Touch-friendly buttons (48px minimum)

3. **Professional Appearance**
   - Clean, modern design
   - Consistent color scheme
   - Clear typography
   - Intuitive navigation

4. **Accessibility**
   - High contrast text
   - Keyboard navigation support
   - Screen reader compatible
   - Clear error messages

---

## 7. Features and Capabilities

### 7.1 Core Features

#### 1. Face Recognition
- **Technology**: InsightFace buffalo_l model
- **Accuracy**: 99.74% validation accuracy
- **Speed**: <100ms per face on Raspberry Pi 4
- **Robustness**: Handles various lighting, angles, expressions
- **Multi-face**: Detects and recognizes multiple faces in single frame

#### 2. Attendance Marking
- **Real-time**: Instant recognition and logging
- **Automatic Logging**: JSON files with timestamps
- **Image Archival**: Saves recognized faces to user folders
- **Duplicate Prevention**: Won't mark same person multiple times in short interval
- **Confidence Scoring**: Shows recognition certainty (0-100%)

#### 3. User Management
- **Easy Registration**: Web-based upload or camera capture
- **Multi-image Support**: 1-10 images per user
- **Instant Activation**: Users can be recognized immediately after adding
- **Update Capability**: Add more images to existing users
- **Delete Users**: Remove users and their data

#### 4. Camera Integration
- **Local USB Cameras**: Automatic detection (0, 1, 2, etc.)
- **ESP32-CAM**: WiFi streaming via HTTP/MJPEG
- **IP Cameras**: Generic MJPEG/RTSP support
- **Android IP Webcam**: Direct compatibility
- **Image Upload**: Attendance from uploaded photos

#### 5. Data Management
- **Persistent Storage**: All data on Raspberry Pi SD card
- **JSON Format**: Human-readable attendance records
- **Automatic Organization**: Files organized by user and date
- **Export Options**: CSV, PDF export of attendance
- **Backup Ready**: Easy to backup entire database folder

---

## 8. Workflow and Processes

### 8.1 Complete System Workflow

```
ESP32-CAM captures video
    ↓
Streams to Raspberry Pi via WiFi (MJPEG/HTTP)
    ↓
Raspberry Pi processes frames with InsightFace
    ↓
Detects faces, generates 512-D embeddings
    ↓
Matches against stored user embeddings (Logistic Regression)
    ↓
If match found (confidence > threshold):
    ├─> Mark attendance in JSON file
    ├─> Save image to user's database folder
    ├─> Display result in web interface
    └─> Update live attendance feed
Else:
    └─> Display "Unknown person"
```

### 8.2 Daily Operation

1. **Morning**: System auto-starts, ESP32-CAM connects
2. **Attendance**: Users stand in front of camera, automatic recognition
3. **Monitoring**: Admin reviews dashboard for today's attendance
4. **Evening**: Export records, backup if needed

### 8.3 Model Training (Cloud)

1. New user images added on Raspberry Pi
2. Sync to GitHub: `./scripts/edge_sync.sh "Add New_User"`
3. GitHub Actions automatically trains models
4. Pull updates: `git pull` on Raspberry Pi
5. System uses improved models immediately

---

## Appendix: Quick Reference

### Common IP Addresses (Examples)
- **Raspberry Pi**: 10.74.63.231
- **ESP32-CAM #1**: 10.74.63.131
- **ESP32-CAM #2**: 10.74.63.132

### Important URLs
- **Web Interface**: http://10.74.63.231:3000
- **ESP32 Stream**: http://10.74.63.131:81/stream
- **Model Status**: http://10.74.63.231:3000/model_status

### Key Commands
```bash
# Start system
python run.py

# SSH to Pi
ssh pi@10.74.63.231

# Sync database
./scripts/edge_sync.sh "Add new users"

# Find ESP32 IP
python ip.py 80:f3:da:62:14:c0

# Train models
python train.py --only embedding
```

### Model Performance
- **Embedding Classifier**: 99.74% accuracy ⭐ Production
- **Custom Embedding**: 98.86% accuracy (Research)
- **Lightweight CNN**: 64.04% accuracy (Research)

---

**Document Version**: 1.0  
**Last Updated**: December 27, 2025  
**System Version**: Production (67 users, 9,648 samples)  
**Recommended Model**: Embedding Classifier (InsightFace + Logistic Regression)
