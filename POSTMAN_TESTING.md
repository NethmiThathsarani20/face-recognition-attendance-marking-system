# API Testing with Postman

This guide explains how to test the Face Recognition Attendance System API using Postman.

## Quick Start

### 1. Import Postman Collection

1. Open Postman
2. Click **Import** button
3. Select `postman_collection.json` from this repository
4. The collection will be imported with all endpoints pre-configured

### 2. Configure Environment

The collection uses the following variables:

- `base_url`: API base URL (default: `http://localhost:3000`)
- `base64_image`: Base64 encoded image for testing
- `base64_image_1`: First image for user registration
- `base64_image_2`: Second image for user registration

**To update variables:**
1. Click on the collection name
2. Go to **Variables** tab
3. Update values as needed

### 3. Start the Server

Before testing, ensure the server is running:

```bash
# Activate virtual environment (if using)
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Start the server
python run.py

# Server will start at http://localhost:3000
```

## API Endpoints

### System Information

#### Get Model Status
```
GET /model_status
```

Returns information about the active model, accuracy, and dataset statistics.

**Example Response:**
```json
{
  "active_model": "embedding_classifier",
  "accuracy": 99.74,
  "num_users": 67,
  "total_samples": 9648,
  "last_trained": "2025-12-27"
}
```

#### Get Users List
```
GET /get_users
```

Returns list of all registered users.

### User Management

#### Add User (JSON with Base64)
```
POST /add_user
Content-Type: application/json

{
  "username": "John_Doe",
  "images": [
    "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAA...",
    "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAA..."
  ]
}
```

#### Add User (Form Data)
```
POST /add_user
Content-Type: multipart/form-data

user_name: John_Doe
user_images: [file1.jpg, file2.jpg, ...]
```

### Attendance Marking

#### Mark Attendance (JSON with Base64)
```
POST /mark_attendance
Content-Type: application/json

{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAA...",
  "camera_source": "ESP32-CAM-1"
}
```

**Example Response:**
```json
{
  "status": "success",
  "name": "John_Doe",
  "confidence": 0.925,
  "timestamp": "2025-12-27 09:15:30"
}
```

#### Get Attendance Records
```
GET /get_attendance
```

Optional query parameters:
- `start_date`: Start date (YYYY-MM-DD)
- `end_date`: End date (YYYY-MM-DD)

### Export Functions

#### Export to PDF
```
GET /export_attendance_pdf?start_date=2025-12-01&end_date=2025-12-31
```

#### Export to Excel
```
GET /export_attendance_excel?start_date=2025-12-01&end_date=2025-12-31
```

## Testing with Base64 Images

### Converting Image to Base64

**Python:**
```python
import base64

with open('image.jpg', 'rb') as f:
    img_data = base64.b64encode(f.read()).decode('utf-8')
    base64_string = f"data:image/jpeg;base64,{img_data}"
    print(base64_string)
```

**Online Tools:**
- https://www.base64-image.de/
- https://base64.guru/converter/encode/image

**Command Line:**
```bash
# Linux/Mac
base64 image.jpg

# Windows PowerShell
[Convert]::ToBase64String([IO.File]::ReadAllBytes("image.jpg"))
```

### Using Base64 in Postman

1. Convert your image to base64
2. Copy the base64 string
3. In Postman, go to collection variables
4. Set `base64_image` to your base64 string (with or without data URI prefix)
5. Use `{{base64_image}}` in your request body

## Sample Test Workflow

### 1. Verify System Status
```
GET /model_status
```
Should return model information.

### 2. Check Registered Users
```
GET /get_users
```
Should return array of user names.

### 3. Add a New User
```
POST /add_user
{
  "username": "Test_User",
  "images": ["{{base64_image_1}}", "{{base64_image_2}}"]
}
```

### 4. Mark Attendance
```
POST /mark_attendance
{
  "image": "{{base64_image}}",
  "camera_source": "Postman-Test"
}
```

### 5. Get Today's Attendance
```
GET /get_attendance
```

### 6. Export Attendance
```
GET /export_attendance_pdf
```

## Expected Screenshots

When testing with Postman, you should capture:

1. **Model Status Response**
   - Shows active_model: "embedding_classifier"
   - Shows accuracy: 99.74
   - Shows num_users and total_samples

2. **Add User Success**
   - Shows status: "success"
   - Shows images_processed count

3. **Mark Attendance Success**
   - Shows recognized name
   - Shows confidence score
   - Shows timestamp

4. **Get Attendance Records**
   - Shows array of attendance records
   - Each with user_name, date, time, confidence

## Troubleshooting

### Connection Refused
- Ensure server is running: `python run.py`
- Check if port 3000 is available
- Update `base_url` if using different port

### Invalid Image Data
- Verify base64 string is complete
- Remove any whitespace/newlines
- Check if data URI prefix is included

### No Face Detected
- Ensure image contains a clear face
- Check image quality and lighting
- Try with different images

### User Already Exists
- Use different username
- Or delete existing user first

## Advanced Testing

### Testing with cURL

```bash
# Get model status
curl http://localhost:3000/model_status

# Mark attendance with base64
curl -X POST http://localhost:3000/mark_attendance \
  -H "Content-Type: application/json" \
  -d '{
    "image": "data:image/jpeg;base64,/9j/4AAQ...",
    "camera_source": "cURL-Test"
  }'

# Get users
curl http://localhost:3000/get_users

# Get attendance
curl http://localhost:3000/get_attendance
```

### Testing with Python

```python
import requests
import base64

# Base URL
base_url = "http://localhost:3000"

# Get model status
response = requests.get(f"{base_url}/model_status")
print(response.json())

# Mark attendance with base64 image
with open('test_image.jpg', 'rb') as f:
    img_data = base64.b64encode(f.read()).decode('utf-8')

response = requests.post(
    f"{base_url}/mark_attendance",
    json={
        "image": f"data:image/jpeg;base64,{img_data}",
        "camera_source": "Python-Test"
    }
)
print(response.json())
```

## API Response Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request - Invalid parameters |
| 404 | Not Found |
| 500 | Internal Server Error |

## Security Notes

- This is a development/demonstration API
- For production, implement:
  - API authentication (API keys, JWT)
  - Rate limiting
  - Input validation
  - HTTPS/SSL
  - CORS configuration

## Support

For issues or questions:
- Check [APPENDIX.md](APPENDIX.md) for complete API documentation
- See [README.md](README.md) for general setup
- Review server logs for error details

---

**Last Updated:** 2026-01-08
