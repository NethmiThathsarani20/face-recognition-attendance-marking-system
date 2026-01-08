#!/usr/bin/env python3
"""
Copy Results and Discussion chapter figures to thesis_diagrams directory.
This script copies the embedding classifier performance figures from the 
embedding_models directory to the thesis_diagrams/embedding_classifier directory
for inclusion in the thesis Results and Discussion chapter.
"""

import os
import shutil
import sys
from pathlib import Path

from PIL import Image


# Project root
PROJECT_ROOT = Path(__file__).parent.parent
EMBEDDING_MODELS_DIR = PROJECT_ROOT / "embedding_models"
THESIS_DIAGRAMS_DIR = PROJECT_ROOT / "thesis_diagrams" / "embedding_classifier"

# Ensure output directory exists
THESIS_DIAGRAMS_DIR.mkdir(exist_ok=True, parents=True)


def convert_and_copy(src_path, dest_path, convert_to_jpg=False, quality=95):
    """
    Copy a file, optionally converting PNG to JPG.
    
    Args:
        src_path: Source file path
        dest_path: Destination file path
        convert_to_jpg: If True and source is PNG, convert to JPG
        quality: JPEG quality (1-100)
    """
    try:
        if convert_to_jpg and src_path.suffix.lower() == '.png':
            # Convert PNG to JPG
            img = Image.open(src_path)
            
            # Convert RGBA to RGB if necessary
            if img.mode in ('RGBA', 'LA', 'P'):
                rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                if img.mode == 'RGBA':
                    rgb_img.paste(img, mask=img.split()[-1])
                else:
                    rgb_img.paste(img)
                img = rgb_img
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Save as JPG
            img.save(dest_path, 'JPEG', quality=quality, optimize=True)
            
            src_size = src_path.stat().st_size / 1024  # KB
            dest_size = dest_path.stat().st_size / 1024  # KB
            
            print(f"✓ Converted and copied: {src_path.name}")
            print(f"  Source: {src_size:.1f} KB (PNG)")
            print(f"  Dest:   {dest_size:.1f} KB (JPG)")
        else:
            # Simple copy
            shutil.copy2(src_path, dest_path)
            size = dest_path.stat().st_size / 1024  # KB
            print(f"✓ Copied: {src_path.name}")
            print(f"  Size: {size:.1f} KB")
        
        return True
    except Exception as e:
        print(f"✗ Error processing {src_path.name}: {e}")
        return False


def main():
    """Copy Results and Discussion figures to thesis_diagrams directory."""
    print("\n" + "=" * 80)
    print("Copying Results and Discussion Chapter Figures")
    print("=" * 80 + "\n")
    
    # Files to copy (source, destination, convert_to_jpg)
    # Performance curves from embedding_models/
    files_to_copy = [
        # Precision-Recall Curve
        ("embedding_precision_recall_curve.png", 
         "embedding_precision_recall_curve.png", False),
        
        # Confidence Distribution
        ("embedding_confidence_curve.png", 
         "embedding_confidence_curve.png", False),
        
        # Precision/Recall vs Confidence Threshold
        ("embedding_precision_confidence_curve.png", 
         "embedding_precision_confidence_curve.png", False),
        
        # Confusion Matrix (convert to JPG as referenced in thesis)
        ("embedding_confusion_matrix.png", 
         "embedding_confusion_matrix.jpg", True),
        
        # Normalized Confusion Matrix (convert to JPG as referenced in thesis)
        ("embedding_confusion_matrix_normalized.png", 
         "embedding_confusion_matrix_normalized.jpg", True),
    ]
    
    # Model comparison diagrams from main thesis_diagrams/
    comparison_diagrams = [
        "model_accuracy_comparison.png",
        "inference_speed_comparison.png", 
        "accuracy_vs_training_time.png",
    ]
    
    success_count = 0
    
    # Copy performance curves from embedding_models
    for src_filename, dest_filename, convert_to_jpg in files_to_copy:
        src_path = EMBEDDING_MODELS_DIR / src_filename
        dest_path = THESIS_DIAGRAMS_DIR / dest_filename
        
        if not src_path.exists():
            print(f"⚠ Warning: Source file not found: {src_path}")
            continue
        
        if convert_and_copy(src_path, dest_path, convert_to_jpg, quality=95):
            success_count += 1
        print()
    
    # Copy comparison diagrams from main thesis_diagrams
    print("Copying model comparison diagrams...")
    print()
    for diagram_filename in comparison_diagrams:
        src_path = PROJECT_ROOT / "thesis_diagrams" / diagram_filename
        dest_path = THESIS_DIAGRAMS_DIR / diagram_filename
        
        if not src_path.exists():
            print(f"⚠ Warning: Source file not found: {src_path}")
            continue
        
        if convert_and_copy(src_path, dest_path, False, quality=95):
            success_count += 1
        print()
    
    total_files = len(files_to_copy) + len(comparison_diagrams)
    
    print("=" * 80)
    print(f"Copy Complete: {success_count}/{total_files} files processed")
    print("=" * 80)
    
    if success_count == total_files:
        print("\n✓ All Results and Discussion figures are now in thesis_diagrams!")
        print(f"Location: {THESIS_DIAGRAMS_DIR}/")
        print("\nFigures for thesis Results and Discussion chapter:")
        print("\nPerformance Analysis:")
        print("1. embedding_precision_recall_curve.png")
        print("   - Shows near-perfect precision-recall performance")
        print("   - Area under curve ≈ 1.0")
        print()
        print("2. embedding_confidence_curve.png")
        print("   - Confidence distribution histogram")
        print("   - Sharp peak between 0.85-0.95")
        print("   - Justifies 60% threshold choice")
        print()
        print("3. embedding_precision_confidence_curve.png")
        print("   - Precision/Recall vs Confidence Threshold")
        print("   - Shows optimal threshold range (0.6-0.8)")
        print()
        print("4. embedding_confusion_matrix.jpg")
        print("   - Detailed confusion matrix with raw counts")
        print("   - Shows 1,930 validation samples with only 5 errors")
        print()
        print("5. embedding_confusion_matrix_normalized.jpg")
        print("   - Normalized confusion matrix (percentage view)")
        print("   - Deep blue diagonal confirms 99.74% accuracy")
        print()
        print("\nModel Comparison:")
        print("6. model_accuracy_comparison.png")
        print("   - Validation accuracy: 99.74% vs 98.86% vs 64.04%")
        print()
        print("7. inference_speed_comparison.png")
        print("   - Real-time speed: 80-100ms vs 90-110ms vs 120-150ms")
        print()
        print("8. accuracy_vs_training_time.png")
        print("   - Trade-off analysis: 99.74% in 30 seconds")
        print()
    
    return success_count == total_files


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
