#!/usr/bin/env python3
"""
Folder Structure Verification Script

This script verifies that all required folders and files are in place
for the Face Recognition Attendance System.
"""

import os
import sys
from pathlib import Path


def check_directory_structure():
    """Verify that all required directories exist."""
    
    print("=" * 70)
    print("Face Recognition Attendance System - Folder Structure Verification")
    print("=" * 70)
    print()
    
    base_dir = Path(__file__).parent
    
    # Required directories
    required_dirs = {
        ".vscode": "VS Code configuration (NEW)",
        "src": "Python source code",
        "templates": "HTML templates",
        "static": "CSS and JavaScript",
        "tests": "Test suite",
        "docs": "Documentation",
        "scripts": "Helper scripts",
        "esp32-camera": "ESP32-CAM firmware",
        ".github/workflows": "GitHub Actions workflows",
    }
    
    # Auto-created directories (may not exist initially)
    auto_created_dirs = {
        "database": "User face images (auto-created)",
        "embeddings": "Face embeddings (auto-created)",
        "attendance_records": "Daily attendance logs (auto-created)",
        "cnn_models": "Trained CNN models (auto-created)",
        "embedding_models": "Embedding classifier models (auto-created)",
        "custom_embedding_models": "Custom embedding models (auto-created)",
    }
    
    # Required files
    required_files = {
        "run.py": "Main application entry point",
        "requirements.txt": "Python dependencies",
        "README.md": "Main project README",
        "INSTRUCTIONS.md": "Detailed setup instructions",
        "VS_CODE_SETUP.md": "VS Code comprehensive guide",
        "QUICK_START_VS_CODE.md": "VS Code quick start",
        "FOLDER_STRUCTURE.md": "Folder structure documentation",
        ".gitignore": "Git ignore patterns",
        "Makefile": "Build and development commands",
        ".vscode/settings.json": "VS Code settings",
        ".vscode/launch.json": "VS Code debug configurations",
        ".vscode/tasks.json": "VS Code tasks",
        ".vscode/extensions.json": "Recommended VS Code extensions",
    }
    
    all_passed = True
    
    # Check required directories
    print("Checking Required Directories:")
    print("-" * 70)
    for dir_name, description in required_dirs.items():
        dir_path = base_dir / dir_name
        if dir_path.exists() and dir_path.is_dir():
            print(f"✅ {dir_name:30s} - {description}")
        else:
            print(f"❌ {dir_name:30s} - MISSING! {description}")
            all_passed = False
    
    print()
    
    # Check auto-created directories
    print("Checking Auto-Created Directories:")
    print("-" * 70)
    for dir_name, description in auto_created_dirs.items():
        dir_path = base_dir / dir_name
        if dir_path.exists() and dir_path.is_dir():
            print(f"✅ {dir_name:30s} - {description}")
        else:
            print(f"ℹ️  {dir_name:30s} - Not yet created. {description}")
    
    print()
    
    # Check required files
    print("Checking Required Files:")
    print("-" * 70)
    for file_name, description in required_files.items():
        file_path = base_dir / file_name
        if file_path.exists() and file_path.is_file():
            print(f"✅ {file_name:30s} - {description}")
        else:
            print(f"❌ {file_name:30s} - MISSING! {description}")
            all_passed = False
    
    print()
    
    # Check virtual environment
    print("Checking Virtual Environment:")
    print("-" * 70)
    venv_dirs = ["venv", ".venv", "env"]
    venv_found = False
    for venv_dir in venv_dirs:
        venv_path = base_dir / venv_dir
        if venv_path.exists() and venv_path.is_dir():
            print(f"✅ Virtual environment found: {venv_dir}")
            venv_found = True
            break
    
    if not venv_found:
        print("ℹ️  No virtual environment found. Create one with:")
        print("   python -m venv venv")
        print("   source venv/bin/activate  (Linux/Mac)")
        print("   venv\\Scripts\\activate     (Windows)")
    
    print()
    
    # Check source code files
    print("Checking Source Code Files:")
    print("-" * 70)
    src_files = {
        "src/config.py": "Configuration settings",
        "src/face_manager.py": "InsightFace integration",
        "src/attendance_system.py": "Attendance logic",
        "src/web_app.py": "Flask web application",
        "src/exceptions.py": "Custom exception framework",
    }
    
    for file_name, description in src_files.items():
        file_path = base_dir / file_name
        if file_path.exists() and file_path.is_file():
            print(f"✅ {file_name:30s} - {description}")
        else:
            print(f"❌ {file_name:30s} - MISSING! {description}")
            all_passed = False
    
    print()
    print("=" * 70)
    
    if all_passed:
        print("✅ All required folders and files are in place!")
        print()
        print("Next Steps:")
        print("1. Create virtual environment (if not exists): python -m venv venv")
        print("2. Activate virtual environment")
        print("3. Install dependencies: pip install -r requirements.txt")
        print("4. Run application: python run.py")
        print()
        print("For VS Code users:")
        print("- See QUICK_START_VS_CODE.md for quick setup")
        print("- See VS_CODE_SETUP.md for comprehensive guide")
        return 0
    else:
        print("❌ Some required files or folders are missing!")
        print("Please ensure you have cloned the complete repository.")
        return 1


def main():
    """Main entry point."""
    try:
        exit_code = check_directory_structure()
        sys.exit(exit_code)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
