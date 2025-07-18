#!/usr/bin/env python3
"""
Demo script to test the attendance system with existing database.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.attendance_system import AttendanceSystem


def main():
    """
    Demo function to test the attendance system.
    """
    print("🎯 Simple Attendance System Demo")
    print("=" * 40)
    
    # Initialize system
    print("📋 Initializing attendance system...")
    try:
        attendance_system = AttendanceSystem()
        print("✅ System initialized successfully!")
    except Exception as e:
        print(f"❌ Error initializing system: {e}")
        return
    
    # Show loaded users
    users = attendance_system.get_user_list()
    print(f"👥 Loaded {len(users)} users from database:")
    for i, user in enumerate(users, 1):
        print(f"   {i}. {user}")
    
    if not users:
        print("⚠️  No users found in database. Please add users first.")
        print("   You can add users via the web interface or by")
        print("   organizing images in the database/ folder by name.")
        return
    
    # Test camera capture
    print("\n📷 Testing camera capture...")
    try:
        # Test local camera first
        print("   Testing local camera (index 0)...")
        test_image = attendance_system.capture_from_camera(0)
        if test_image is not None:
            print("✅ Local camera capture successful!")
            
            # Test attendance marking
            print("🔍 Testing attendance marking with captured image...")
            result = attendance_system.mark_attendance(test_image)
            
            if result['success']:
                print(f"✅ Attendance marked successfully!")
                print(f"   User: {result['user_name']}")
                print(f"   Confidence: {result['confidence']:.2f}")
                print(f"   Time: {result['attendance_record']['time']}")
            else:
                print(f"❌ Attendance marking failed: {result['message']}")
        else:
            print("❌ Local camera capture failed.")
            
            # Suggest IP camera testing
            print("\n💡 If you have an IP camera, you can test it like this:")
            print("   - IP Webcam (Android): http://192.168.1.100:8080/video")
            print("   - ESP32-CAM: http://192.168.1.100:81/stream")
            print("   - Generic MJPEG: http://192.168.1.100:PORT/video")
            print("   - Generic RTSP: rtsp://192.168.1.100:PORT/stream")
            print("   Example:")
            print("   test_image = attendance_system.capture_from_camera('http://192.168.1.100:8080/video')")
            
    except Exception as e:
        print(f"❌ Error during camera test: {e}")
    
    # Show today's attendance
    print("\n📊 Today's Attendance Records:")
    attendance_records = attendance_system.get_today_attendance()
    if attendance_records:
        for i, record in enumerate(attendance_records, 1):
            print(f"   {i}. {record['user_name']} - {record['time']} (confidence: {record['confidence']:.2f})")
    else:
        print("   No attendance records for today.")
    
    print("\n🚀 Demo completed!")
    print("   To start the web interface, run: python run.py")


if __name__ == '__main__':
    main()
