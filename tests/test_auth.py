#!/usr/bin/env python3
"""
Quick IP Camera Authentication Test
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.attendance_system import AttendanceSystem


def main():
    """Test IP camera with authentication."""
    print("🔐 IP Camera Authentication Test")
    print("=" * 40)
    
    # Initialize system
    print("📋 Initializing attendance system...")
    try:
        attendance_system = AttendanceSystem()
        print("✅ System initialized successfully!")
    except Exception as e:
        print(f"❌ Error initializing system: {e}")
        return
    
    # Your camera IP
    camera_ip = "192.168.1.4"
    camera_port = "8554"
    
    print(f"\n🎯 Testing camera at {camera_ip}:{camera_port}")
    print("Since the camera returned 401 (authentication required),")
    print("we need to add username and password to the URL.")
    print()
    
    # Common default credentials
    common_credentials = [
        ("admin", "admin"),
        ("admin", "password"),
        ("admin", ""),
        ("root", "root"),
        ("user", "user"),
        ("admin", "123456"),
        ("admin", "admin123"),
    ]
    
    print("🔑 Trying common default credentials:")
    for username, password in common_credentials:
        auth_url = f"http://{username}:{password}@{camera_ip}:{camera_port}/video"
        print(f"   Testing: {username}:{password}")
        
        try:
            image = attendance_system.capture_from_camera(auth_url)
            if image is not None:
                print(f"✅ SUCCESS! Working credentials: {username}:{password}")
                print(f"✅ Working URL: {auth_url}")
                print(f"✅ Image captured: {image.shape}")
                return
            else:
                print("❌ Failed")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n🔒 Manual credential entry:")
    print("If none of the default credentials worked, try entering them manually:")
    
    username = input("Enter username (or press Enter to skip): ").strip()
    if username:
        password = input("Enter password: ").strip()
        
        auth_url = f"http://{username}:{password}@{camera_ip}:{camera_port}/video"
        print(f"Testing: {auth_url}")
        
        try:
            image = attendance_system.capture_from_camera(auth_url)
            if image is not None:
                print(f"✅ SUCCESS! Working URL: {auth_url}")
                print(f"✅ Image captured: {image.shape}")
                
                # Test attendance marking
                print("\n🔍 Testing attendance marking...")
                result = attendance_system.mark_attendance(image)
                if result['success']:
                    print(f"✅ Attendance marked successfully!")
                    print(f"   User: {result['user_name']}")
                    print(f"   Confidence: {result['confidence']:.2f}")
                else:
                    print(f"❌ Attendance marking failed: {result['message']}")
                    
                return
            else:
                print("❌ Authentication failed")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n💡 Next steps:")
    print("1. Check your camera's documentation for default credentials")
    print("2. Look for camera setup interface (usually web-based)")
    print("3. Common camera web interfaces:")
    print(f"   - http://{camera_ip}:{camera_port}")
    print(f"   - http://{camera_ip}")
    print(f"   - http://{camera_ip}:80")
    print("4. Try different streaming endpoints:")
    print(f"   - http://user:pass@{camera_ip}:{camera_port}/stream")
    print(f"   - http://user:pass@{camera_ip}:{camera_port}/mjpeg")
    print(f"   - http://user:pass@{camera_ip}:{camera_port}/video.cgi")


if __name__ == '__main__':
    main()
