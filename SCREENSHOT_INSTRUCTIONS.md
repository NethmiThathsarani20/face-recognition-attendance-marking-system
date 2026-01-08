# Postman API Testing - Screenshot Guide

This guide shows exactly how to capture the screenshots requested in the problem statement.

## Prerequisites

1. **Start the Server**
   ```bash
   # Activate virtual environment
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   
   # Start server
   python run.py
   ```

2. **Import Postman Collection**
   - Open Postman
   - Click **Import** → Select `postman_collection.json`
   - Collection will appear in left sidebar

3. **Prepare Test Images**
   - Convert images to base64 (see instructions below)
   - Set variables in collection

## Converting Images to Base64

### Method 1: Python Script
```python
import base64

# Read image and convert to base64
with open('test_face.jpg', 'rb') as f:
    img_data = base64.b64encode(f.read()).decode('utf-8')
    
# Print with data URI prefix
print(f"data:image/jpeg;base64,{img_data}")
```

### Method 2: Online Tool
1. Visit: https://www.base64-image.de/
2. Upload your image
3. Copy the base64 string (entire data URI)

### Method 3: Command Line
```bash
# Linux/Mac
base64 test_face.jpg | tr -d '\n'

# Add data URI prefix manually:
# data:image/jpeg;base64,<paste_base64_here>
```

## Screenshot 1: GET /model_status

**Purpose**: Show model information and performance metrics

### Steps:
1. In Postman, select "Get Model Status" request
2. Click **Send** button
3. Wait for response

### What to Capture:
- Full Postman window showing:
  - Request URL: `http://localhost:3000/model_status`
  - Method: GET
  - Response Status: `200 OK`
  - Response time
  - Response body with JSON

### Expected Response:
```json
{
  "active_model": "embedding_classifier",
  "accuracy": 99.74,
  "num_users": 67,
  "total_samples": 9648,
  "last_trained": "2025-12-27",
  "current_model": "Embedding",
  "cnn_model_available": true,
  "custom_embedding_model_available": true,
  "embedding_model_available": true,
  "insightface_available": true
}
```

### Screenshot Checklist:
- [ ] URL visible at top
- [ ] GET method shown
- [ ] Status 200 OK in green
- [ ] Response time displayed
- [ ] Full JSON response visible
- [ ] JSON properly formatted (use Pretty view)

---

## Screenshot 2: POST /add_user

**Purpose**: Demonstrate adding a new user with base64 images

### Steps:
1. Select "Add User (JSON Base64)" request
2. Go to **Body** tab
3. Ensure raw JSON is selected
4. Update the JSON with your data:
   ```json
   {
     "username": "Test_User",
     "images": [
       "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
       "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
     ]
   }
   ```
5. Click **Send**

### What to Capture:
- Full window showing:
  - Request URL: `http://localhost:3000/add_user`
  - Method: POST
  - Body tab selected with JSON visible
  - Response status: `200 OK`
  - Success response

### Expected Response:
```json
{
  "status": "success",
  "message": "User Test_User added successfully",
  "images_processed": 2
}
```

### Screenshot Checklist:
- [ ] POST method visible
- [ ] Body tab showing JSON request
- [ ] Base64 images in request (can truncate for visibility)
- [ ] Success response with correct count
- [ ] Status 200 OK

---

## Screenshot 3: POST /mark_attendance

**Purpose**: Show attendance marking with face recognition

### Steps:
1. Select "Mark Attendance (JSON Base64)" request
2. Go to **Body** tab
3. Update JSON with test image:
   ```json
   {
     "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
     "camera_source": "ESP32-CAM-1"
   }
   ```
4. Click **Send**

### What to Capture:
- Full window showing:
  - Request URL: `http://localhost:3000/mark_attendance`
  - Method: POST
  - Request body with image and camera_source
  - Response with recognized name and confidence

### Expected Response:
```json
{
  "status": "success",
  "name": "Test_User",
  "confidence": 0.925,
  "timestamp": "2025-12-27 09:15:30"
}
```

### Screenshot Checklist:
- [ ] POST method visible
- [ ] Request body showing image data
- [ ] camera_source parameter visible
- [ ] Response shows recognized name
- [ ] Confidence score displayed
- [ ] Timestamp in correct format

---

## Screenshot 4: Training Loss Curves

**Purpose**: Show training loss and validation loss over epochs

### Steps:
1. Navigate to `embedding_models/` directory
2. Open `embedding_training_loss_and_metrics.png`
3. Take screenshot of the entire image

### What to Show:
- **Top-left panel**: Training and Validation Loss
  - Blue line: Training Loss (decreasing)
  - Red line: Validation Loss (decreasing)
  - Both converging to low values
  
- **Top-right panel**: Training and Validation Accuracy
  - Green line: Training Accuracy (99.94%)
  - Orange line: Validation Accuracy (99.74%)
  - Purple dashed line: Target 99.74%

- **Bottom-left panel**: Precision, Recall, F1-Score
  - All metrics converging to 99.74%
  - Recall line emphasized (red, thicker)

- **Bottom-right panel**: Recall Performance
  - Bar chart with recall percentages
  - Annotation showing "Final Recall: 99.74%"

### Screenshot Checklist:
- [ ] All 4 panels visible
- [ ] Title readable: "Embedding Classifier Training Metrics - Superior Recall Performance"
- [ ] Legend visible in each panel
- [ ] Axes labels clear
- [ ] Final values readable (99.74%)

---

## Screenshot 5: Recall Performance Over Epochs

**Purpose**: Show superior recall performance specifically

### Steps:
1. Open `embedding_models/embedding_recall_performance_epochs.png`
2. Take screenshot of the entire image

### What to Show:
- Large plot showing recall over 30 epochs
- Red line with circular markers
- Green shaded area below the line
- Purple dashed line at 99.74%
- Key metrics box in top-left:
  - Final Recall: 99.74%
  - Final Precision: 99.74%
  - Final F1-Score: 99.74%
  - Validation Acc: 99.74%
  - Training Acc: 99.94%

### Screenshot Checklist:
- [ ] Title visible: "Embedding Classifier - Superior Recall Performance Over Epochs"
- [ ] Key metrics box readable
- [ ] Recall line prominent and clear
- [ ] Target line (99.74%) visible
- [ ] X-axis: Epochs (1-30)
- [ ] Y-axis: Recall (98.5-100.5%)

---

## Additional Screenshots (Optional)

### GET /get_users
Shows list of registered users
```json
["Test_User", "John_Doe", "Jane_Smith"]
```

### GET /get_attendance
Shows today's attendance records
```json
[
  {
    "user_name": "Test_User",
    "date": "2025-12-27",
    "time": "09:15:30",
    "confidence": 0.925
  }
]
```

---

## Screenshot Organization

Recommended naming:
1. `screenshot_1_model_status.png` - Model status API
2. `screenshot_2_add_user.png` - Add user API
3. `screenshot_3_mark_attendance.png` - Mark attendance API
4. `screenshot_4_training_loss.png` - Training loss curves
5. `screenshot_5_recall_performance.png` - Recall performance

---

## Tips for Good Screenshots

1. **Use Full Window**
   - Capture entire Postman window
   - Include URL, method, status code
   - Show both request and response

2. **Format JSON**
   - Click "Pretty" view in response
   - Ensure JSON is readable
   - Expand all nested objects

3. **Highlight Key Information**
   - Ensure status codes are visible
   - Make sure metrics are clear
   - Verify timestamps and values

4. **High Resolution**
   - Use high DPI if available
   - Ensure text is readable
   - Don't compress too much

5. **Consistent Size**
   - Try to keep all screenshots same width
   - Maintain aspect ratio
   - Crop unnecessary parts

---

## Verification Checklist

Before submitting screenshots, verify:

- [ ] All 5 main screenshots captured
- [ ] All screenshots are clear and readable
- [ ] Status codes visible (200 OK)
- [ ] JSON responses properly formatted
- [ ] Training curves show all panels
- [ ] Recall performance clearly visible
- [ ] Metrics match documentation (99.74%)
- [ ] Timestamps in correct format
- [ ] File names are descriptive

---

## Troubleshooting

### Server Not Responding
```bash
# Check if server is running
curl http://localhost:3000/model_status

# Restart server if needed
python run.py
```

### Base64 Too Long for Screenshot
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg... [truncated]",
  "camera_source": "ESP32-CAM-1"
}
```

### Response Not Formatted
- Click "Pretty" button in Postman
- Or use online JSON formatter

### Images Don't Load
- Check base64 is complete
- Verify no line breaks in string
- Ensure data URI prefix is included

---

## Summary

You should have:
1. ✅ Model status showing 99.74% accuracy
2. ✅ Add user showing successful registration
3. ✅ Mark attendance showing recognition with confidence
4. ✅ Training loss curves (4 panels)
5. ✅ Recall performance curve with 99.74% achievement

All screenshots should clearly demonstrate:
- Working API endpoints
- Base64 image support
- Superior recall performance (99.74%)
- Comprehensive training metrics

---

**Last Updated**: 2026-01-08  
**Version**: 1.0
