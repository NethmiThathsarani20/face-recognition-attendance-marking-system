# Postman API Testing Screenshots - Complete Guide

This document provides detailed instructions for capturing Postman API testing screenshots to demonstrate the API functionality.

## 📋 Overview

You need to capture screenshots showing:
1. ✅ GET /model_status - Model information and accuracy
2. ✅ POST /add_user - User registration with base64 images
3. ✅ POST /mark_attendance - Attendance marking with recognition
4. ✅ GET /get_users - List of registered users
5. ✅ GET /get_attendance - Attendance records

---

## 🚀 Quick Start Guide

### Step 1: Start the Server

```bash
# Navigate to project directory
cd face-recognition-attendance-marking-system

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Start the server
python run.py
```

**Expected output:**
```
Starting Simple Attendance System...
Open your browser and go to: http://0.0.0.0:3000
Press Ctrl+C to stop the server
```

### Step 2: Import Postman Collection

1. Open Postman desktop application
2. Click **Import** button (top-left corner)
3. Select **File** → Browse to `postman_collection.json`
4. Click **Import**
5. Collection appears in left sidebar: "Face Recognition Attendance System API"

### Step 3: Configure Variables

1. Click on collection name
2. Select **Variables** tab
3. Set the following:
   - `base_url`: `http://localhost:3000`
   - `base64_image`: (your base64 encoded test image)
   - `base64_image_1`: (first registration image)
   - `base64_image_2`: (second registration image)
4. Click **Save**

---

## 📸 Screenshot 1: GET /model_status

### Purpose
Demonstrate that the API returns model information including the 99.74% accuracy.

### Steps to Capture

1. **Select Request**:
   - In Postman, expand collection
   - Click "System Information" folder
   - Click "Get Model Status"

2. **Send Request**:
   - Click the blue **Send** button
   - Wait for response (should be < 100ms)

3. **Verify Response**:
   ```json
   {
     "active_model": "embedding_classifier",
     "accuracy": 99.74,
     "num_users": 67,
     "total_samples": 9648,
     "last_trained": "2025-12-27",
     "current_model": "Embedding",
     "cnn_model_available": true,
     "embedding_model_available": true,
     "insightface_available": true
   }
   ```

4. **What to Include in Screenshot**:
   ```
   ┌─────────────────────────────────────────────────────┐
   │ Postman Window                                      │
   ├─────────────────────────────────────────────────────┤
   │ GET │ http://localhost:3000/model_status       [Send]│
   ├─────────────────────────────────────────────────────┤
   │ Params │ Authorization │ Headers │ Body          │
   ├─────────────────────────────────────────────────────┤
   │ Response                                            │
   │ Status: 200 OK    Time: 45ms    Size: 245B         │
   │ ┌─────────────────────────────────────────────┐   │
   │ │ Pretty │ Raw │ Preview │                     │   │
   │ │ {                                            │   │
   │ │   "active_model": "embedding_classifier",   │   │
   │ │   "accuracy": 99.74,                        │   │
   │ │   "num_users": 67,                          │   │
   │ │   "total_samples": 9648,                    │   │
   │ │   "last_trained": "2025-12-27"              │   │
   │ │ }                                            │   │
   │ └─────────────────────────────────────────────┘   │
   └─────────────────────────────────────────────────────┘
   ```

5. **Capture Screenshot**:
   - Use full window capture
   - Ensure status is green (200 OK)
   - Verify "accuracy": 99.74 is visible
   - Save as: `postman_1_model_status.png`

### ✅ Checklist
- [ ] URL visible: `http://localhost:3000/model_status`
- [ ] Method: GET
- [ ] Status: 200 OK (in green)
- [ ] Response time shown
- [ ] JSON response visible with "accuracy": 99.74
- [ ] Pretty view selected

---

## 📸 Screenshot 2: POST /add_user

### Purpose
Demonstrate adding a new user with base64-encoded images.

### Steps to Capture

1. **Prepare Base64 Images**:
   ```python
   # Use this script to convert your image
   import base64
   
   with open('face_image.jpg', 'rb') as f:
       img_data = base64.b64encode(f.read()).decode('utf-8')
       print(f"data:image/jpeg;base64,{img_data}")
   ```

2. **Select Request**:
   - Click "User Management" folder
   - Click "Add User (JSON Base64)"

3. **Configure Request Body**:
   - Click **Body** tab
   - Select **raw** and **JSON**
   - Enter:
   ```json
   {
     "username": "Test_User",
     "images": [
       "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
       "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
     ]
   }
   ```

4. **Send Request**:
   - Click **Send**
   - Wait for response

5. **Expected Response**:
   ```json
   {
     "status": "success",
     "message": "User Test_User added successfully",
     "images_processed": 2
   }
   ```

6. **What to Include in Screenshot**:
   ```
   ┌─────────────────────────────────────────────────────┐
   │ POST │ http://localhost:3000/add_user        [Send] │
   ├─────────────────────────────────────────────────────┤
   │ Params │ Authorization │ Headers │ Body       ▼    │
   ├─────────────────────────────────────────────────────┤
   │ Body                                                │
   │ ○ none  ○ form-data  ○ x-www-form-urlencoded       │
   │ ● raw   ○ binary                          JSON ▼   │
   │ ┌─────────────────────────────────────────────┐   │
   │ │ {                                            │   │
   │ │   "username": "Test_User",                  │   │
   │ │   "images": [                                │   │
   │ │     "data:image/jpeg;base64,/9j/4AAQ...",  │   │
   │ │     "data:image/jpeg;base64,/9j/4AAQ..."   │   │
   │ │   ]                                          │   │
   │ │ }                                            │   │
   │ └─────────────────────────────────────────────┘   │
   ├─────────────────────────────────────────────────────┤
   │ Response                                            │
   │ Status: 200 OK    Time: 520ms                      │
   │ {                                                   │
   │   "status": "success",                             │
   │   "message": "User Test_User added successfully",  │
   │   "images_processed": 2                            │
   │ }                                                   │
   └─────────────────────────────────────────────────────┘
   ```

7. **Capture Screenshot**:
   - Show both request body AND response
   - Base64 can be truncated with "..." for readability
   - Save as: `postman_2_add_user.png`

### ✅ Checklist
- [ ] URL: `http://localhost:3000/add_user`
- [ ] Method: POST
- [ ] Body tab selected showing JSON request
- [ ] Base64 images visible in request
- [ ] Response shows "status": "success"
- [ ] "images_processed": 2 visible

---

## 📸 Screenshot 3: POST /mark_attendance

### Purpose
Demonstrate attendance marking with face recognition and confidence score.

### Steps to Capture

1. **Select Request**:
   - Click "Attendance" folder
   - Click "Mark Attendance (JSON Base64)"

2. **Configure Request**:
   ```json
   {
     "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
     "camera_source": "ESP32-CAM-1"
   }
   ```

3. **Send Request**:
   - Click **Send**

4. **Expected Response**:
   ```json
   {
     "status": "success",
     "name": "Test_User",
     "confidence": 0.925,
     "timestamp": "2025-12-27 09:15:30"
   }
   ```

5. **What to Include**:
   ```
   ┌─────────────────────────────────────────────────────┐
   │ POST │ http://localhost:3000/mark_attendance [Send] │
   ├─────────────────────────────────────────────────────┤
   │ Body: raw JSON                                      │
   │ {                                                   │
   │   "image": "data:image/jpeg;base64,/9j/4AAQ...",  │
   │   "camera_source": "ESP32-CAM-1"                   │
   │ }                                                   │
   ├─────────────────────────────────────────────────────┤
   │ Response: 200 OK    Time: 340ms                    │
   │ {                                                   │
   │   "status": "success",                             │
   │   "name": "Test_User",                             │
   │   "confidence": 0.925,                             │
   │   "timestamp": "2025-12-27 09:15:30"               │
   │ }                                                   │
   └─────────────────────────────────────────────────────┘
   ```

6. **Save as**: `postman_3_mark_attendance.png`

### ✅ Checklist
- [ ] URL: `http://localhost:3000/mark_attendance`
- [ ] Method: POST
- [ ] Request shows image and camera_source
- [ ] Response shows recognized name
- [ ] Confidence score visible (e.g., 0.925)
- [ ] Timestamp in correct format

---

## 📸 Screenshot 4: GET /get_users

### Purpose
Show the list of registered users in the system.

### Steps to Capture

1. **Select Request**:
   - Click "System Information" folder
   - Click "Get Users List"

2. **Send Request**

3. **Expected Response**:
   ```json
   [
     "Test_User",
     "John_Doe",
     "Jane_Smith"
   ]
   ```

4. **Save as**: `postman_4_get_users.png`

### ✅ Checklist
- [ ] URL: `http://localhost:3000/get_users`
- [ ] Method: GET
- [ ] Status: 200 OK
- [ ] Response is array of user names

---

## 📸 Screenshot 5: GET /get_attendance

### Purpose
Display attendance records with user names, dates, times, and confidence scores.

### Steps to Capture

1. **Select Request**:
   - Click "Attendance" folder
   - Click "Get Today's Attendance"

2. **Send Request**

3. **Expected Response**:
   ```json
   [
     {
       "user_name": "Test_User",
       "date": "2025-12-27",
       "time": "09:15:30",
       "confidence": 0.925
     },
     {
       "user_name": "John_Doe",
       "date": "2025-12-27",
       "time": "09:20:15",
       "confidence": 0.887
     }
   ]
   ```

4. **Save as**: `postman_5_get_attendance.png`

### ✅ Checklist
- [ ] URL: `http://localhost:3000/get_attendance`
- [ ] Method: GET
- [ ] Response is array of records
- [ ] Each record has user_name, date, time, confidence

---

## 🎨 Screenshot Quality Guidelines

### Recommended Settings

1. **Window Size**:
   - Postman in maximized window
   - Or consistent size for all screenshots (1920x1080 recommended)

2. **Zoom Level**:
   - 100% zoom in Postman
   - Ensure text is readable

3. **View Settings**:
   - Always use **Pretty** view for JSON responses
   - Enable JSON syntax highlighting

4. **What to Include**:
   ✅ Full Postman window
   ✅ URL bar with complete endpoint
   ✅ HTTP method (GET/POST)
   ✅ Status code with color (green for 200 OK)
   ✅ Response time
   ✅ Request body (for POST)
   ✅ Full response body

5. **What to Exclude**:
   ❌ Personal information in headers
   ❌ Sensitive tokens or keys
   ❌ Other browser windows
   ❌ Desktop background

---

## 🔧 Troubleshooting

### Issue: Connection Refused

**Solution:**
```bash
# Check if server is running
curl http://localhost:3000/model_status

# If not, start it:
python run.py
```

### Issue: Base64 Image Too Long

**Solution:**
- In screenshot, you can show truncated base64
- Add "..." to indicate truncation
- Example: `"data:image/jpeg;base64,/9j/4AAQSkZJRg... [truncated]"`

### Issue: Response Not Formatted

**Solution:**
- Click "Pretty" tab in response section
- Ensure JSON is selected in dropdown

### Issue: No Users or Attendance

**Solution:**
- First run "Add User" request
- Then run "Mark Attendance" request
- Then check "Get Attendance"

---

## 📦 Complete Checklist

Before submission, verify you have:

- [ ] 5 main screenshots captured
- [ ] All screenshots are clear and readable
- [ ] Status codes visible (200 OK in green)
- [ ] JSON responses formatted (Pretty view)
- [ ] Request bodies shown for POST requests
- [ ] Response times visible
- [ ] All URLs correct (http://localhost:3000)
- [ ] File names descriptive

---

## 🎯 Expected Screenshot Files

After completion, you should have:

1. `postman_1_model_status.png` - Shows 99.74% accuracy
2. `postman_2_add_user.png` - User registration with base64
3. `postman_3_mark_attendance.png` - Recognition with confidence
4. `postman_4_get_users.png` - List of users
5. `postman_5_get_attendance.png` - Attendance records

---

## 📚 Additional Resources

- **Postman Collection**: `postman_collection.json`
- **API Documentation**: `APPENDIX.md` Section D
- **Testing Guide**: `POSTMAN_TESTING.md`
- **Screenshot Instructions**: `SCREENSHOT_INSTRUCTIONS.md`
- **Implementation Summary**: `IMPLEMENTATION_SUMMARY.md`

---

## ✨ Pro Tips

1. **Use Collection Runner**: Run all requests sequentially
2. **Save Responses**: Use "Save Response" feature
3. **Export Results**: Export collection with responses
4. **Document Issues**: Note any errors encountered
5. **Test Order**: Follow the order: Status → Add User → Mark Attendance → Get Records

---

**Last Updated**: 2026-01-08  
**Version**: 1.0  
**Status**: Ready for Testing
