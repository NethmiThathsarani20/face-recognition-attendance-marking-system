#!/usr/bin/env python3
"""
Main entry point for the attendance system.
Simple launcher that starts the web application.
"""

import sys
import os

# Suppress TensorFlow and other verbose logging BEFORE importing anything else
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimization (can be slow on startup)
os.environ['ALBUMENTATIONS_DEBUG'] = '0'  # Disable albumentations debug
os.environ['ALBUMENTATIONS_SKIP_VERSION_CHECK'] = '1'  # Skip albumentations version check

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.web_app import run_app

if __name__ == '__main__':
    try:
        run_app()
    except ImportError as e:
        print(f"Error starting application: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting application: {e}")
        sys.exit(1)
