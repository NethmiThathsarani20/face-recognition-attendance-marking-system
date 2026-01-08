#!/usr/bin/env python3
"""
Simulates API testing and generates example outputs for screenshot documentation.
This script demonstrates what the API responses look like for documentation purposes.
"""

import json
import base64
import numpy as np
import cv2
from datetime import datetime

def create_sample_base64_image():
    """Create a sample image and convert to base64."""
    # Create a simple test image (100x100 red square)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :] = (0, 0, 255)  # Red in BGR
    
    # Add text
    cv2.putText(img, "TEST", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Encode to base64
    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return f"data:image/jpeg;base64,{img_base64}"

def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def print_api_example(method, url, request_body=None, response=None, headers=None):
    """Print formatted API request/response example."""
    print(f"\n{method} {url}")
    
    if headers:
        print("\nHeaders:")
        for key, value in headers.items():
            print(f"  {key}: {value}")
    
    if request_body:
        print("\nRequest Body:")
        print(json.dumps(request_body, indent=2))
    
    if response:
        print("\nResponse:")
        print(f"Status: {response.get('status_code', 200)} {response.get('status_text', 'OK')}")
        print(f"Time: {response.get('time', '120ms')}")
        print("\nResponse Body:")
        body = {k: v for k, v in response.items() if k not in ['status_code', 'status_text', 'time']}
        print(json.dumps(body, indent=2))

def main():
    """Generate example API responses for documentation."""
    
    print_section("POSTMAN API TESTING EXAMPLES")
    print("\nThese examples show what you should see in Postman when testing the API.")
    print("Use these as reference when capturing screenshots.\n")
    
    # Example 1: Get Model Status
    print_section("Example 1: GET /model_status")
    print("\nThis endpoint returns information about the active model and its performance.")
    
    print_api_example(
        method="GET",
        url="http://localhost:3000/model_status",
        response={
            "status_code": 200,
            "status_text": "OK",
            "time": "45ms",
            "active_model": "embedding_classifier",
            "accuracy": 99.74,
            "num_users": 67,
            "total_samples": 9648,
            "last_trained": "2025-12-27",
            "current_model": "Embedding",
            "cnn_model_available": True,
            "custom_embedding_model_available": True,
            "embedding_model_available": True,
            "insightface_available": True
        }
    )
    
    print("\n📸 SCREENSHOT TIP: Capture the full Postman window showing:")
    print("   - URL and GET method")
    print("   - Status 200 OK in green")
    print("   - Response time")
    print("   - Full JSON response with accuracy: 99.74")
    
    # Example 2: Add User with Base64
    print_section("Example 2: POST /add_user (JSON with Base64)")
    print("\nThis endpoint adds a new user with base64-encoded images.")
    
    # Create sample base64 image
    base64_image = create_sample_base64_image()
    
    print_api_example(
        method="POST",
        url="http://localhost:3000/add_user",
        headers={
            "Content-Type": "application/json"
        },
        request_body={
            "username": "John_Doe",
            "images": [
                base64_image[:100] + "... [truncated for display]",
                base64_image[:100] + "... [truncated for display]"
            ]
        },
        response={
            "status_code": 200,
            "status_text": "OK",
            "time": "520ms",
            "status": "success",
            "message": "User John_Doe added successfully",
            "images_processed": 2
        }
    )
    
    print("\n📸 SCREENSHOT TIP: Capture showing:")
    print("   - POST method and URL")
    print("   - Body tab with JSON request")
    print("   - Base64 images in request (can truncate)")
    print("   - Success response with images_processed count")
    
    # Example 3: Mark Attendance
    print_section("Example 3: POST /mark_attendance (JSON with Base64)")
    print("\nThis endpoint marks attendance using a base64-encoded face image.")
    
    print_api_example(
        method="POST",
        url="http://localhost:3000/mark_attendance",
        headers={
            "Content-Type": "application/json"
        },
        request_body={
            "image": base64_image[:100] + "... [truncated for display]",
            "camera_source": "ESP32-CAM-1"
        },
        response={
            "status_code": 200,
            "status_text": "OK",
            "time": "340ms",
            "status": "success",
            "name": "John_Doe",
            "confidence": 0.925,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    )
    
    print("\n📸 SCREENSHOT TIP: Capture showing:")
    print("   - POST method and URL")
    print("   - Request with image and camera_source")
    print("   - Success response with recognized name")
    print("   - Confidence score (0.925)")
    print("   - Timestamp")
    
    # Example 4: Get Users
    print_section("Example 4: GET /get_users")
    print("\nThis endpoint returns a list of all registered users.")
    
    print_api_example(
        method="GET",
        url="http://localhost:3000/get_users",
        response={
            "status_code": 200,
            "status_text": "OK",
            "time": "25ms",
            "users": [
                "John_Doe",
                "Jane_Smith",
                "Bob_Johnson",
                "Alice_Williams",
                "Charlie_Brown"
            ]
        }
    )
    
    print("\n📸 SCREENSHOT TIP: Capture showing:")
    print("   - GET method and URL")
    print("   - Array of user names")
    print("   - Fast response time (~25ms)")
    
    # Example 5: Get Attendance
    print_section("Example 5: GET /get_attendance")
    print("\nThis endpoint returns today's attendance records.")
    
    print_api_example(
        method="GET",
        url="http://localhost:3000/get_attendance",
        response={
            "status_code": 200,
            "status_text": "OK",
            "time": "35ms",
            "records": [
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
                },
                {
                    "user_name": "Bob_Johnson",
                    "date": "2025-12-27",
                    "time": "09:25:42",
                    "confidence": 0.912
                }
            ]
        }
    )
    
    print("\n📸 SCREENSHOT TIP: Capture showing:")
    print("   - GET method and URL")
    print("   - Array of attendance records")
    print("   - Each record with user_name, date, time, confidence")
    
    # Example 6: Get Attendance with Date Range
    print_section("Example 6: GET /get_attendance?start_date=2025-12-01&end_date=2025-12-31")
    print("\nThis endpoint returns attendance records for a date range.")
    
    print_api_example(
        method="GET",
        url="http://localhost:3000/get_attendance?start_date=2025-12-01&end_date=2025-12-31",
        response={
            "status_code": 200,
            "status_text": "OK",
            "time": "78ms",
            "records": [
                {
                    "user_name": "John_Doe",
                    "date": "2025-12-15",
                    "time": "09:15:30",
                    "confidence": 0.925
                },
                {
                    "user_name": "Jane_Smith",
                    "date": "2025-12-16",
                    "time": "09:20:15",
                    "confidence": 0.887
                }
            ]
        }
    )
    
    print("\n📸 SCREENSHOT TIP: Capture showing:")
    print("   - GET method and URL with query parameters")
    print("   - Date range in URL")
    print("   - Filtered attendance records")
    
    # Summary
    print_section("SCREENSHOT CHECKLIST")
    print("""
Required Screenshots:
□ 1. GET /model_status - showing accuracy 99.74%
□ 2. POST /add_user - with base64 images
□ 3. POST /mark_attendance - with recognized name and confidence
□ 4. GET /get_users - showing list of users
□ 5. GET /get_attendance - showing attendance records

Optional Screenshots:
□ 6. GET /get_attendance with date range
□ 7. POST /initialize_system
□ 8. GET /export_attendance_pdf
□ 9. Camera test endpoint

All screenshots should show:
✓ Full Postman window
✓ URL and HTTP method
✓ Status code (200 OK in green)
✓ Response time
✓ Full JSON response (formatted/Pretty view)
✓ Request body (for POST requests)
    """)
    
    print_section("NEXT STEPS")
    print("""
1. Start the server:
   python run.py

2. Import Postman collection:
   - Open Postman
   - Import postman_collection.json
   - Set base_url variable to http://localhost:3000

3. Prepare test images:
   - Convert images to base64
   - Set base64_image variables in collection

4. Run each request and capture screenshots

5. See SCREENSHOT_INSTRUCTIONS.md for detailed guide
    """)

if __name__ == "__main__":
    main()
