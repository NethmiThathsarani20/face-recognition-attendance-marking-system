#!/usr/bin/env python3
"""
Script to resize large images in the database to ESP32-CAM compatible size.
Resizes images larger than max_size while maintaining aspect ratio.
"""

import os
import sys
from pathlib import Path
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import DATABASE_DIR

# ESP32-CAM configuration
MAX_SIZE = 640  # Maximum dimension (width or height)
JPEG_QUALITY = 95  # Quality for JPEG compression (1-100)
SUPPORTED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp')  # Image file extensions


def resize_if_needed(image_path: str, max_size: int = MAX_SIZE) -> bool:
    """
    Resize image if it's larger than max_size.
    
    Args:
        image_path: Path to the image
        max_size: Maximum dimension (width or height)
    
    Returns:
        True if resized, False otherwise
    """
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            
            # Check if resize needed
            if width <= max_size and height <= max_size:
                return False
            
            # Calculate new size maintaining aspect ratio
            if width > height:
                new_width = max_size
                new_height = int((max_size / width) * height)
            else:
                new_height = max_size
                new_width = int((max_size / height) * width)
            
            # Resize and save
            img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            img_resized.save(image_path, quality=JPEG_QUALITY)
            
            print(f"  Resized {os.path.basename(image_path)}: {width}x{height} → {new_width}x{new_height}")
            return True
            
    except Exception as e:
        print(f"  ⚠️ Error processing {image_path}: {e}")
        return False


def main():
    """Resize all large images in the database."""
    if not os.path.exists(DATABASE_DIR):
        print(f"❌ Database directory not found: {DATABASE_DIR}")
        sys.exit(1)
    
    print(f"🔍 Scanning database for large images (>{MAX_SIZE}x{MAX_SIZE})...")
    print(f"Database: {DATABASE_DIR}\n")
    
    total_images = 0
    resized_count = 0
    
    # Process each user folder
    for user_name in sorted(os.listdir(DATABASE_DIR)):
        user_folder = os.path.join(DATABASE_DIR, user_name)
        
        if not os.path.isdir(user_folder) or user_name.startswith("."):
            continue
        
        user_resized = 0
        for image_file in os.listdir(user_folder):
            if image_file.lower().endswith(SUPPORTED_EXTENSIONS):
                image_path = os.path.join(user_folder, image_file)
                total_images += 1
                
                if resize_if_needed(image_path, MAX_SIZE):
                    user_resized += 1
        
        if user_resized > 0:
            print(f"  {user_name}: resized {user_resized} images")
            resized_count += user_resized
    
    print(f"\n✅ Done!")
    print(f"Total images processed: {total_images}")
    print(f"Images resized: {resized_count}")
    
    if resized_count == 0:
        print("All images are already suitable for ESP32-CAM!")


if __name__ == "__main__":
    main()
