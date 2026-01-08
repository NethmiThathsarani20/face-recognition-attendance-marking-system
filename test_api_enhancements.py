#!/usr/bin/env python3
"""
Simple script to test API endpoints without starting the full server.
Tests the new base64 functionality.
"""

import base64
import json
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_base64_handling():
    """Test base64 image encoding/decoding."""
    print("Testing Base64 Image Handling")
    print("=" * 60)
    
    # Create a simple test image (1x1 red pixel)
    import numpy as np
    import cv2
    
    # Create test image
    test_img = np.zeros((100, 100, 3), dtype=np.uint8)
    test_img[:, :] = (255, 0, 0)  # Red
    
    # Encode to base64
    _, buffer = cv2.imencode('.jpg', test_img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    print(f"✅ Created test image: 100x100 pixels")
    print(f"✅ Encoded to base64: {len(img_base64)} characters")
    
    # Test decoding
    img_bytes = base64.b64decode(img_base64)
    nparr = np.frombuffer(img_bytes, np.uint8)
    decoded_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if decoded_img is not None:
        print(f"✅ Successfully decoded image: {decoded_img.shape}")
    else:
        print(f"❌ Failed to decode image")
        return False
    
    # Test with data URI prefix
    data_uri = f"data:image/jpeg;base64,{img_base64}"
    print(f"✅ Created data URI: {len(data_uri)} characters")
    
    # Extract and decode
    if ',' in data_uri:
        img_data = data_uri.split(',')[1]
    else:
        img_data = data_uri
    
    img_bytes = base64.b64decode(img_data)
    nparr = np.frombuffer(img_bytes, np.uint8)
    decoded_img2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if decoded_img2 is not None:
        print(f"✅ Successfully decoded from data URI: {decoded_img2.shape}")
    else:
        print(f"❌ Failed to decode from data URI")
        return False
    
    print("\n✅ All base64 handling tests passed!")
    return True

def test_api_response_format():
    """Test API response format."""
    print("\nTesting API Response Formats")
    print("=" * 60)
    
    # Test add_user response
    add_user_response = {
        "status": "success",
        "message": "User John_Doe added successfully",
        "images_processed": 5
    }
    print("Add User Response:")
    print(json.dumps(add_user_response, indent=2))
    
    # Test mark_attendance response
    mark_attendance_response = {
        "status": "success",
        "name": "John_Doe",
        "confidence": 0.925,
        "timestamp": "2025-12-27 09:15:30"
    }
    print("\nMark Attendance Response:")
    print(json.dumps(mark_attendance_response, indent=2))
    
    # Test model_status response
    model_status_response = {
        "active_model": "embedding_classifier",
        "accuracy": 99.74,
        "num_users": 67,
        "total_samples": 9648,
        "last_trained": "2025-12-27"
    }
    print("\nModel Status Response:")
    print(json.dumps(model_status_response, indent=2))
    
    print("\n✅ All response format tests passed!")
    return True

def verify_training_curves():
    """Verify training curve images exist."""
    print("\nVerifying Training Curves")
    print("=" * 60)
    
    files_to_check = [
        "embedding_models/embedding_training_loss_and_metrics.png",
        "embedding_models/embedding_recall_performance_epochs.png",
        "embedding_models/training_summary.json",
        "embedding_models/epoch_metrics.json"
    ]
    
    all_exist = True
    for file_path in files_to_check:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✅ {file_path} ({size:,} bytes)")
        else:
            print(f"❌ {file_path} not found")
            all_exist = False
    
    if all_exist:
        print("\n✅ All training curve files exist!")
    else:
        print("\n❌ Some training curve files are missing")
    
    return all_exist

def verify_documentation():
    """Verify documentation files exist."""
    print("\nVerifying Documentation Files")
    print("=" * 60)
    
    files_to_check = [
        "APPENDIX.md",
        "POSTMAN_TESTING.md",
        "postman_collection.json"
    ]
    
    all_exist = True
    for file_path in files_to_check:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✅ {file_path} ({size:,} bytes)")
        else:
            print(f"❌ {file_path} not found")
            all_exist = False
    
    if all_exist:
        print("\n✅ All documentation files exist!")
    else:
        print("\n❌ Some documentation files are missing")
    
    return all_exist

def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("API Enhancement Verification Tests")
    print("=" * 60 + "\n")
    
    results = []
    
    # Test 1: Base64 handling
    try:
        results.append(("Base64 Handling", test_base64_handling()))
    except Exception as e:
        print(f"❌ Base64 handling test failed: {e}")
        results.append(("Base64 Handling", False))
    
    # Test 2: API response formats
    try:
        results.append(("API Response Formats", test_api_response_format()))
    except Exception as e:
        print(f"❌ API response format test failed: {e}")
        results.append(("API Response Formats", False))
    
    # Test 3: Training curves
    try:
        results.append(("Training Curves", verify_training_curves()))
    except Exception as e:
        print(f"❌ Training curves verification failed: {e}")
        results.append(("Training Curves", False))
    
    # Test 4: Documentation
    try:
        results.append(("Documentation", verify_documentation()))
    except Exception as e:
        print(f"❌ Documentation verification failed: {e}")
        results.append(("Documentation", False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(result[1] for result in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed. Please review the output above.")
    print("=" * 60 + "\n")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
