#!/usr/bin/env python3
"""
Test script for IP camera functionality.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.attendance_system import AttendanceSystem


def main():
    """
    Test IP camera functionality.
    """
    print("🎯 IP Camera Test Script")
    print("=" * 40)
    
    # Initialize system
    print("📋 Initializing attendance system...")
    try:
        attendance_system = AttendanceSystem()
        print("✅ System initialized successfully!")
    except Exception as e:
        print(f"❌ Error initializing system: {e}")
        return
    
    # Test different IP camera formats
    test_urls = [
        "http://192.168.1.100:8080/video",  # IP Webcam (Android)
        "http://192.168.1.100:81/stream",   # ESP32-CAM
        "rtsp://192.168.1.100:554/stream",  # RTSP stream
        "http://admin:password@192.168.1.100:8080/video",  # With authentication
    ]
    
    print("\n📱 Common IP Camera URL Formats:")
    for i, url in enumerate(test_urls, 1):
        print(f"   {i}. {url}")
    
    print("\n💡 To test your IP camera:")
    print("   1. Replace IP address with your camera's IP")
    print("   2. Make sure the camera is accessible on your network")
    print("   3. Use the correct port and path for your camera")
    
    # Interactive test
    print("\n🔧 Interactive Test:")
    try:
        camera_url = input("Enter your IP camera URL (or press Enter to skip): ").strip()
        
        if camera_url:
            print(f"📷 Testing IP camera: {camera_url}")
            test_image = attendance_system.capture_from_camera(camera_url)
            
            if test_image is not None:
                print("✅ IP camera capture successful!")
                print(f"   Image shape: {test_image.shape}")
                
                # Test attendance marking
                print("🔍 Testing attendance marking with IP camera image...")
                result = attendance_system.mark_attendance(test_image)
                
                if result['success']:
                    print(f"✅ Attendance marked successfully!")
                    print(f"   User: {result['user_name']}")
                    print(f"   Confidence: {result['confidence']:.2f}")
                    print(f"   Time: {result['attendance_record']['time']}")
                else:
                    print(f"❌ Attendance marking failed: {result['message']}")
                    
            else:
                print("❌ IP camera capture failed.")
                print("   Common issues:")
                print("   - Camera not accessible on network")
                print("   - Incorrect URL format")
                print("   - Camera requires authentication")
                print("   - Port blocked by firewall")
        else:
            print("⏭️ Skipping IP camera test")
            
    except Exception as e:
        print(f"❌ Error during IP camera test: {e}")
    
    print("\n🚀 Test completed!")
    print("   For web interface with IP camera support, run: python run.py")


if __name__ == '__main__':
    main()
