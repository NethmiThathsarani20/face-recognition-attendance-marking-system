#!/usr/bin/env python3
"""
Setup script for the attendance system.
Prepares the system for first use.
"""

import os
import sys
import subprocess

def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def setup_directories():
    """Create necessary directories."""
    print("📁 Setting up directories...")
    directories = [
        'database',
        'embeddings', 
        'attendance_records',
        'static',
        'templates',
        'temp'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"   ✅ {directory}/")
    
    print("✅ Directories created successfully!")

def load_existing_users():
    """Load users from existing database folder."""
    print("👥 Loading existing users...")
    
    # Add src to path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    
    try:
        from src.attendance_system import AttendanceSystem
        
        attendance_system = AttendanceSystem()
        users = attendance_system.get_user_list()
        
        print(f"✅ Loaded {len(users)} users:")
        for i, user in enumerate(users, 1):
            print(f"   {i}. {user}")
        
        return len(users)
    except Exception as e:
        print(f"❌ Error loading users: {e}")
        return 0

def main():
    """Main setup function."""
    print("🎯 Simple Attendance System Setup")
    print("=" * 40)
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required")
        sys.exit(1)
    
    # Setup directories
    setup_directories()
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Setup failed. Please install dependencies manually.")
        sys.exit(1)
    
    # Load existing users
    user_count = load_existing_users()
    
    print("\n🚀 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Add users via web interface: python run.py")
    print("2. Or organize images in database/ folder by name")
    print("3. Run demo: python demo.py")
    print("4. Run tests: python tests/run_tests.py")
    print("\n🌐 Web interface will be available at: http://localhost:3000")
    
    if user_count > 0:
        print(f"\n✅ {user_count} users already loaded from database")

if __name__ == '__main__':
    main()
