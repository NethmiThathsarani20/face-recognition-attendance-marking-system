"""
Test configuration and utilities for the attendance system.
"""

import os
import sys
import tempfile
import shutil
import numpy as np
import cv2

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

# Test configuration
TEST_CONFIG = {
    'test_images_dir': os.path.join(os.path.dirname(__file__), 'test_images'),
    'temp_dir': tempfile.mkdtemp(),
    'similarity_threshold': 0.4,
    'test_users': ['test_user1', 'test_user2']
}


def create_test_image(width=640, height=480, color=(100, 100, 100)):
    """
    Create a simple test image.
    
    Args:
        width: Image width
        height: Image height
        color: BGR color tuple
        
    Returns:
        Test image as numpy array
    """
    image = np.full((height, width, 3), color, dtype=np.uint8)
    
    # Add some simple features (rectangles) to make it more realistic
    cv2.rectangle(image, (width//4, height//4), (3*width//4, 3*height//4), (200, 200, 200), -1)
    cv2.circle(image, (width//2, height//2), min(width, height)//8, (50, 50, 50), -1)
    
    return image


def setup_test_database():
    """
    Create test database with sample images.
    """
    test_db_dir = os.path.join(TEST_CONFIG['temp_dir'], 'test_database')
    os.makedirs(test_db_dir, exist_ok=True)
    
    for user in TEST_CONFIG['test_users']:
        user_dir = os.path.join(test_db_dir, user)
        os.makedirs(user_dir, exist_ok=True)
        
        # Create 3 test images per user
        for i in range(3):
            image = create_test_image(color=(100 + i*20, 100 + i*20, 100 + i*20))
            image_path = os.path.join(user_dir, f'{user}_{i+1}.jpg')
            cv2.imwrite(image_path, image)
    
    return test_db_dir


def cleanup_test_data():
    """
    Clean up test data.
    """
    if os.path.exists(TEST_CONFIG['temp_dir']):
        shutil.rmtree(TEST_CONFIG['temp_dir'])


def get_test_image_path(user_name, image_number=1):
    """
    Get path to test image.
    
    Args:
        user_name: Name of test user
        image_number: Image number (1-3)
        
    Returns:
        Path to test image
    """
    test_db_dir = os.path.join(TEST_CONFIG['temp_dir'], 'test_database')
    return os.path.join(test_db_dir, user_name, f'{user_name}_{image_number}.jpg')
