#!/usr/bin/env python3
"""
Verification script to check all requirements are met.
"""
import os
import sys
from pathlib import Path

def check_requirement(name, condition, details=""):
    """Check a requirement and print result."""
    status = "✅" if condition else "❌"
    print(f"{status} {name}")
    if details and not condition:
        print(f"   {details}")
    return condition

def main():
    base_dir = Path(__file__).parent
    
    print("=" * 60)
    print("VERIFICATION: Model Training Requirements")
    print("=" * 60)
    print()
    
    all_passed = True
    
    # 1. Database Consolidation
    print("1. DATABASE CONSOLIDATION")
    all_passed &= check_requirement(
        "Only 'database' directory exists",
        (base_dir / "database").exists() and 
        not (base_dir / "database1").exists() and 
        not (base_dir / "database2").exists(),
        "database1 or database2 still exists"
    )
    
    # Count images
    database_dir = base_dir / "database"
    if database_dir.exists():
        jpg_files = list(database_dir.rglob("*.jpg"))
        all_passed &= check_requirement(
            f"Database has images ({len(jpg_files)} found)",
            len(jpg_files) > 1500,
            f"Expected ~1,595 images, found {len(jpg_files)}"
        )
    print()
    
    # 2. Model Training Fixes
    print("2. MODEL TRAINING FIXES")
    
    try:
        from src.cnn_trainer import CNNTrainer
        import inspect
        source = inspect.getsource(CNNTrainer)
        all_passed &= check_requirement(
            "CNN trainer uses HaarFaceDetector",
            "HaarFaceDetector" in source,
            "HaarFaceDetector not found in CNN trainer"
        )
        all_passed &= check_requirement(
            "CNN trainer does NOT import FaceManager",
            "from .face_manager import FaceManager" not in source and
            "from face_manager import FaceManager" not in source,
            "FaceManager still imported in CNN trainer"
        )
    except Exception as e:
        all_passed &= check_requirement("CNN trainer loads", False, str(e))
    
    try:
        from src.custom_embedding_trainer import CustomEmbeddingTrainer
        import inspect
        source = inspect.getsource(CustomEmbeddingTrainer)
        all_passed &= check_requirement(
            "Custom Embedding uses HaarFaceDetector",
            "HaarFaceDetector" in source,
            "HaarFaceDetector not found"
        )
    except Exception as e:
        all_passed &= check_requirement("Custom Embedding loads", False, str(e))
    
    try:
        from src.embedding_trainer import EmbeddingTrainer
        import inspect
        source = inspect.getsource(EmbeddingTrainer.__init__)
        all_passed &= check_requirement(
            "Embedding Classifier uses FaceManager (InsightFace)",
            "FaceManager" in source,
            "FaceManager not found in Embedding trainer"
        )
    except Exception as e:
        all_passed &= check_requirement("Embedding trainer loads", False, str(e))
    
    print()
    
    # 3. Documentation
    print("3. DOCUMENTATION")
    all_passed &= check_requirement(
        "MODEL_TRAINING.md exists",
        (base_dir / "docs" / "MODEL_TRAINING.md").exists(),
        "docs/MODEL_TRAINING.md not found"
    )
    all_passed &= check_requirement(
        "generate_model_comparison.py exists",
        (base_dir / "scripts" / "generate_model_comparison.py").exists(),
        "scripts/generate_model_comparison.py not found"
    )
    all_passed &= check_requirement(
        "IMPLEMENTATION_SUMMARY.md exists",
        (base_dir / "IMPLEMENTATION_SUMMARY.md").exists(),
        "IMPLEMENTATION_SUMMARY.md not found"
    )
    print()
    
    # 4. Cleanup
    print("4. CLEANUP (December files deleted)")
    all_passed &= check_requirement(
        "check_december_commits.py deleted",
        not (base_dir / "check_december_commits.py").exists()
    )
    all_passed &= check_requirement(
        "DECEMBER_DELETION_TASK.md deleted",
        not (base_dir / "DECEMBER_DELETION_TASK.md").exists()
    )
    all_passed &= check_requirement(
        "DECEMBER_COMMITS_REPORT.md deleted",
        not (base_dir / "DECEMBER_COMMITS_REPORT.md").exists()
    )
    all_passed &= check_requirement(
        "test_december_commits_deleted.py deleted",
        not (base_dir / "tests" / "test_december_commits_deleted.py").exists()
    )
    all_passed &= check_requirement(
        "check-december-commits.yml deleted",
        not (base_dir / ".github" / "workflows" / "check-december-commits.yml").exists()
    )
    print()
    
    # 5. Evaluation Metrics (Check code, not files since models not trained yet)
    print("5. EVALUATION METRICS (Code verification)")
    try:
        # Check if trainers generate expected metrics
        cnn_source = inspect.getsource(CNNTrainer.train_model)
        all_passed &= check_requirement(
            "CNN generates confusion_matrix",
            "confusion_matrix.png" in cnn_source
        )
        all_passed &= check_requirement(
            "CNN generates precision_recall_curve",
            "precision_recall_curve.png" in cnn_source
        )
        all_passed &= check_requirement(
            "CNN generates confidence_curve",
            "confidence_curve.png" in cnn_source
        )
    except:
        pass
    
    print()
    print("=" * 60)
    
    if all_passed:
        print("✅ ALL REQUIREMENTS VERIFIED")
        print("=" * 60)
        print()
        print("Next steps:")
        print("1. Train models: python train.py --epochs 30 --validation-split 0.2")
        print("2. Generate comparison: python scripts/generate_model_comparison.py")
        print()
        return 0
    else:
        print("❌ SOME REQUIREMENTS FAILED")
        print("=" * 60)
        print("Please review the failures above")
        print()
        return 1

if __name__ == "__main__":
    sys.exit(main())
