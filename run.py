#!/usr/bin/env python3
"""
Main entry point for the attendance system.
Simple launcher that starts the web application.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.web_app import run_app

if __name__ == '__main__':
    print("Starting Simple Attendance System...")
    print("Open your browser and go to: http://localhost:3000")
    print("Press Ctrl+C to stop the server")
    
    try:
        run_app()
    except KeyboardInterrupt:
        print("\nShutting down...")
        sys.exit(0)
    except Exception as e:
        print(f"Error starting application: {e}")
        sys.exit(1)
