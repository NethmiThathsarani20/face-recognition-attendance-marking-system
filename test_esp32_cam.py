#!/usr/bin/env python3
"""
ESP32-CAM Testing and Verification Script

This script helps you:
1. Find your ESP32 Cam IP address by MAC address
2. Test ESP32 cam connectivity and video stream
3. Verify face detection works correctly with ESP32 cam
4. Test attendance marking functionality
"""

import argparse
import sys
import os
import time
import cv2
import requests

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.attendance_system import AttendanceSystem
from src.face_manager import FaceManager


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)


def print_success(text):
    """Print success message."""
    print(f"✅ {text}")


def print_error(text):
    """Print error message."""
    print(f"❌ {text}")


def print_info(text):
    """Print info message."""
    print(f"ℹ️  {text}")


def find_esp32_ip(mac_address=None):
    """Find ESP32 Cam IP address using MAC address."""
    print_header("Step 1: Find ESP32 Cam IP Address")
    
    if mac_address:
        print_info(f"Searching for device with MAC address: {mac_address}")
        # Use the ip.py script
        import subprocess
        try:
            result = subprocess.run(
                ["python", "ip.py", mac_address],
                capture_output=True,
                text=True,
                timeout=60
            )
            if result.returncode == 0:
                # Extract IP from output
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if '.' in line and not line.startswith('Found'):
                        ip = line.strip()
                        print_success(f"Found ESP32 Cam IP: {ip}")
                        return ip
            print_error("Could not find device with given MAC address")
        except Exception as e:
            print_error(f"Error finding IP: {e}")
    else:
        print_info("No MAC address provided. Common ESP32 Cam URLs:")
        print("  • http://192.168.1.100:81/stream")
        print("  • http://10.74.63.131:81/stream")
        print("  • Check your router's DHCP table for ESP32-CAM")
        print("  • Look for device with MAC address starting with: d8:3a:dd or 80:f3:da")
    
    return None


def test_esp32_connectivity(camera_url):
    """Test ESP32 cam connectivity and HTTP stream."""
    print_header("Step 2: Test ESP32 Cam Connectivity")
    
    print_info(f"Testing connection to: {camera_url}")
    
    # Test 1: HTTP connectivity
    try:
        # Try to access the root page first
        base_url = camera_url.replace('/stream', '')
        print_info(f"Checking base URL: {base_url}")
        response = requests.get(base_url, timeout=5)
        if response.status_code == 200:
            print_success(f"HTTP connection successful (Status: {response.status_code})")
        else:
            print_error(f"HTTP connection failed (Status: {response.status_code})")
    except Exception as e:
        print_error(f"HTTP connection failed: {e}")
    
    # Test 2: Stream endpoint
    try:
        print_info(f"Checking stream endpoint: {camera_url}")
        response = requests.get(camera_url, timeout=5, stream=True)
        if response.status_code == 200:
            print_success(f"Stream endpoint accessible (Status: {response.status_code})")
            print_success(f"Content-Type: {response.headers.get('Content-Type', 'N/A')}")
            return True
        else:
            print_error(f"Stream endpoint failed (Status: {response.status_code})")
            return False
    except Exception as e:
        print_error(f"Stream endpoint failed: {e}")
        return False


def test_video_capture(camera_url):
    """Test video capture from ESP32 cam."""
    print_header("Step 3: Test Video Capture")
    
    print_info(f"Opening video stream: {camera_url}")
    
    # Use CAP_FFMPEG backend with timeout settings for network streams
    cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
    
    # Set timeout properties (in milliseconds)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)  # 5 second open timeout
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)  # 5 second read timeout
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer for lower latency
    
    if not cap.isOpened():
        print_error("Failed to open video stream")
        print_info("💡 Troubleshooting tips:")
        print_info("   - Verify camera URL is correct and accessible")
        print_info("   - Check that camera is streaming (not just taking photos)")
        print_info("   - Try opening the URL in a web browser first")
        print_info("   - Ensure camera is on the same network")
        return False
    
    print_success("Video stream opened successfully")
    
    # Try to read a few frames
    frame_count = 0
    max_frames = 10
    
    print_info(f"Reading {max_frames} frames to verify stream quality...")
    
    for i in range(max_frames):
        ret, frame = cap.read()
        if ret:
            frame_count += 1
            height, width = frame.shape[:2]
            print_success(f"Frame {i+1}/{max_frames}: {width}x{height} pixels")
        else:
            print_error(f"Failed to read frame {i+1}/{max_frames}")
        time.sleep(0.1)
    
    cap.release()
    
    if frame_count == max_frames:
        print_success(f"Successfully captured all {max_frames} frames")
        return True
    else:
        print_error(f"Only captured {frame_count}/{max_frames} frames")
        return False


def test_face_detection(camera_url):
    """Test face detection with ESP32 cam."""
    print_header("Step 4: Test Face Detection")
    
    print_info("Initializing face detection system...")
    
    try:
        face_manager = FaceManager()
        print_success("Face detection system initialized")
    except Exception as e:
        print_error(f"Failed to initialize face detection: {e}")
        return False
    
    print_info(f"Opening video stream: {camera_url}")
    
    # Use CAP_FFMPEG backend with timeout settings for network streams
    cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
    
    # Set timeout properties (in milliseconds)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)  # 5 second open timeout
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)  # 5 second read timeout
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer for lower latency
    
    if not cap.isOpened():
        print_error("Failed to open video stream")
        return False
    
    print_info("Attempting to detect faces in 10 frames...")
    print_info("💡 TIP: Position your face in front of the ESP32 camera")
    
    faces_detected = 0
    frames_tested = 0
    max_frames = 10
    
    for i in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            print_error(f"Failed to read frame {i+1}")
            continue
        
        frames_tested += 1
        
        # Detect faces
        faces = face_manager.detect_faces(frame)
        
        if faces is not None and len(faces) > 0:
            faces_detected += 1
            print_success(f"Frame {i+1}: Detected {len(faces)} face(s)")
        else:
            print_info(f"Frame {i+1}: No faces detected")
        
        time.sleep(0.2)
    
    cap.release()
    
    print_info(f"Summary: Detected faces in {faces_detected}/{frames_tested} frames")
    
    if faces_detected > 0:
        print_success("Face detection is working correctly!")
        return True
    else:
        print_error("No faces detected. Check camera positioning and lighting.")
        return False


def test_attendance_marking(camera_url):
    """Test attendance marking with ESP32 cam."""
    print_header("Step 5: Test Attendance Marking")
    
    print_info("Initializing attendance system...")
    
    try:
        attendance_system = AttendanceSystem()
        print_success("Attendance system initialized")
    except Exception as e:
        print_error(f"Failed to initialize attendance system: {e}")
        return False
    
    # Check if there are any registered users
    users = attendance_system.get_user_list()
    if not users:
        print_error("No registered users found!")
        print_info("Please add users via the web interface before testing attendance")
        return False
    
    print_success(f"Found {len(users)} registered user(s): {', '.join(users)}")
    
    print_info("Attempting to mark attendance...")
    print_info("💡 TIP: Position a registered user's face in front of the camera")
    
    try:
        # Capture image from camera
        image = attendance_system.capture_from_camera(camera_url)
        
        if image is None:
            print_error("Failed to capture image from camera")
            return False
        
        print_success("Image captured successfully")
        
        # Mark attendance
        result = attendance_system.mark_attendance(image, save_captured=True)
        
        if result["success"]:
            print_success(f"Attendance marked successfully!")
            print_success(f"  User: {result['user_name']}")
            print_success(f"  Confidence: {result['confidence']:.2f}")
            print_success(f"  Time: {result['attendance_record']['time']}")
            return True
        else:
            print_error(f"Failed to mark attendance: {result['message']}")
            return False
            
    except Exception as e:
        print_error(f"Error during attendance marking: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_live_stream_ui():
    """Test live stream UI feature."""
    print_header("Step 6: Test Live Stream UI Feature")
    
    print_info("To test the live stream UI:")
    print("  1. Start the web application: python run.py")
    print("  2. Open browser: http://localhost:3000")
    print("  3. Select 'IP Camera (URL)' option")
    print("  4. Enter your ESP32 cam URL (e.g., http://192.168.1.100:81/stream)")
    print("  5. Click 'Start Live Stream' button")
    print("  6. You should see live video with face detection boxes")
    print()
    print_info("Alternative: Use the /video_feed endpoint directly:")
    print("  http://localhost:3000/video_feed/http://192.168.1.100:81/stream")


def main():
    parser = argparse.ArgumentParser(
        description="ESP32-CAM Testing and Verification Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Find ESP32 by MAC and test everything
  python test_esp32_cam.py --mac d8:3a:dd:51:6b:3c

  # Test specific ESP32 cam URL
  python test_esp32_cam.py --url http://192.168.1.100:81/stream

  # Test all features with known URL
  python test_esp32_cam.py --url http://10.74.63.131:81/stream --all

  # Just find the IP
  python test_esp32_cam.py --mac d8:3a:dd:51:6b:3c --find-only
        """
    )
    
    parser.add_argument(
        '--mac',
        help='MAC address of ESP32 cam (e.g., d8:3a:dd:51:6b:3c)'
    )
    
    parser.add_argument(
        '--url',
        help='ESP32 cam stream URL (e.g., http://192.168.1.100:81/stream)'
    )
    
    parser.add_argument(
        '--find-only',
        action='store_true',
        help='Only find IP address, skip testing'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all tests (connectivity, video, face detection, attendance)'
    )
    
    args = parser.parse_args()
    
    # Show header
    print_header("ESP32-CAM Testing and Verification Tool")
    
    # Step 1: Find IP if MAC provided
    camera_url = args.url
    
    if args.mac:
        found_ip = find_esp32_ip(args.mac)
        if found_ip and not camera_url:
            camera_url = f"http://{found_ip}:81/stream"
            print_info(f"Using camera URL: {camera_url}")
    elif not camera_url:
        find_esp32_ip(None)  # Show info
    
    if args.find_only:
        return
    
    if not camera_url:
        print_error("No camera URL provided!")
        print_info("Use --url to specify ESP32 cam URL or --mac to find it")
        sys.exit(1)
    
    # Run tests
    all_tests_passed = True
    
    # Test connectivity
    if not test_esp32_connectivity(camera_url):
        all_tests_passed = False
        if not args.all:
            print_error("Connectivity test failed. Fix connectivity before proceeding.")
            sys.exit(1)
    
    # Test video capture
    if not test_video_capture(camera_url):
        all_tests_passed = False
        if not args.all:
            print_error("Video capture test failed.")
            sys.exit(1)
    
    # Test face detection (always run if we got here or if --all)
    if args.all or all_tests_passed:
        if not test_face_detection(camera_url):
            all_tests_passed = False
    
    # Test attendance marking (always run if we got here or if --all)
    if args.all or all_tests_passed:
        if not test_attendance_marking(camera_url):
            all_tests_passed = False
    
    # Show UI testing instructions
    test_live_stream_ui()
    
    # Final summary
    print_header("Test Summary")
    
    if all_tests_passed:
        print_success("All tests passed! ESP32 cam is working correctly.")
        print_info("Your ESP32 cam is ready for attendance marking.")
    else:
        print_error("Some tests failed. Please check the errors above.")
    
    print()


if __name__ == '__main__':
    main()
