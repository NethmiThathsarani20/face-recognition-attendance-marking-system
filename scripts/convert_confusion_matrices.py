#!/usr/bin/env python3
"""
Convert confusion matrix PNG files to JPG format for thesis inclusion.
This script converts the existing PNG confusion matrices to JPG format
as referenced in the thesis Results and Discussion chapter.
"""

import os
import sys
from pathlib import Path

from PIL import Image

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
EMBEDDING_MODELS_DIR = PROJECT_ROOT / "embedding_models"

def convert_png_to_jpg(png_path, jpg_path, quality=95):
    """
    Convert a PNG image to JPG format with high quality.
    
    Args:
        png_path: Path to source PNG file
        jpg_path: Path to output JPG file
        quality: JPEG quality (1-100, default 95 for high quality)
    """
    try:
        # Open PNG image
        img = Image.open(png_path)
        
        # Convert RGBA to RGB if necessary (JPG doesn't support transparency)
        if img.mode in ('RGBA', 'LA', 'P'):
            # Create a white background
            rgb_img = Image.new('RGB', img.size, (255, 255, 255))
            # Paste the image on the white background
            if img.mode == 'P':
                img = img.convert('RGBA')
            rgb_img.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
            img = rgb_img
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Save as JPG with high quality
        img.save(jpg_path, 'JPEG', quality=quality, optimize=True)
        
        # Get file sizes
        png_size = os.path.getsize(png_path) / (1024 * 1024)  # MB
        jpg_size = os.path.getsize(jpg_path) / (1024 * 1024)  # MB
        
        print(f"✓ Converted: {png_path.name}")
        print(f"  PNG size: {png_size:.2f} MB")
        print(f"  JPG size: {jpg_size:.2f} MB")
        print(f"  Compression: {((png_size - jpg_size) / png_size * 100):.1f}% smaller")
        
        return True
    except Exception as e:
        print(f"✗ Error converting {png_path.name}: {e}")
        return False

def main():
    """Convert confusion matrix PNG files to JPG format."""
    print("\n" + "=" * 80)
    print("Converting Confusion Matrix Files from PNG to JPG")
    print("=" * 80 + "\n")
    
    # Files to convert
    conversions = [
        ("embedding_confusion_matrix.png", "embedding_confusion_matrix.jpg"),
        ("embedding_confusion_matrix_normalized.png", "embedding_confusion_matrix_normalized.jpg"),
    ]
    
    success_count = 0
    for png_filename, jpg_filename in conversions:
        png_path = EMBEDDING_MODELS_DIR / png_filename
        jpg_path = EMBEDDING_MODELS_DIR / jpg_filename
        
        if not png_path.exists():
            print(f"⚠ Warning: Source file not found: {png_path}")
            continue
        
        if convert_png_to_jpg(png_path, jpg_path, quality=95):
            success_count += 1
        print()
    
    print("=" * 80)
    print(f"Conversion Complete: {success_count}/{len(conversions)} files converted")
    print("=" * 80)
    
    if success_count == len(conversions):
        print("\n✓ All confusion matrix files are now available in JPG format!")
        print(f"Location: {EMBEDDING_MODELS_DIR}/")
        print("\nGenerated files:")
        for _, jpg_filename in conversions:
            jpg_path = EMBEDDING_MODELS_DIR / jpg_filename
            if jpg_path.exists():
                size_mb = os.path.getsize(jpg_path) / (1024 * 1024)
                print(f"  - {jpg_filename} ({size_mb:.2f} MB)")
    
    return success_count == len(conversions)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
