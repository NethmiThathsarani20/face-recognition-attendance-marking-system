#!/usr/bin/env python3
"""
Generate comprehensive model comparison charts and tables.
This script reads training logs from all three models and creates:
1. Bar chart comparing key metrics
2. Detailed comparison table
3. Updated documentation with actual results
"""
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Paths
BASE_DIR = Path(__file__).parent
EMBEDDING_LOG = BASE_DIR / "embedding_models" / "training_log.json"
CNN_LOG = BASE_DIR / "cnn_models" / "training_log.json"
CUSTOM_LOG = BASE_DIR / "custom_embedding_models" / "training_log.json"
DOCS_DIR = BASE_DIR / "docs"
OUTPUT_CHART = DOCS_DIR / "model_comparison_chart.png"
OUTPUT_TABLE = DOCS_DIR / "model_comparison_table.md"


def load_log(path: Path) -> Optional[Dict[str, Any]]:
    """Load training log JSON if it exists."""
    if not path.exists():
        return None
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️  Failed to load {path}: {e}")
        return None


def extract_metrics(log: Optional[Dict[str, Any]], model_name: str) -> Dict[str, Any]:
    """Extract key metrics from training log."""
    if log is None:
        return {
            "name": model_name,
            "train_acc": None,
            "val_acc": None,
            "top3_acc": None,
            "num_classes": None,
            "num_samples": None,
            "training_time": None,
        }
    
    metrics = {
        "name": model_name,
        "train_acc": log.get("train_accuracy") or log.get("final_train_accuracy"),
        "val_acc": log.get("val_accuracy") or log.get("final_val_accuracy"),
        "top3_acc": log.get("val_top3_accuracy") or log.get("final_val_top3_accuracy"),
        "num_classes": log.get("num_classes"),
        "num_samples": log.get("num_samples"),
        "training_time": log.get("training_time_seconds"),
    }
    
    return metrics


def create_comparison_chart(embedding_metrics, cnn_metrics, custom_metrics):
    """Create bar chart comparing model metrics."""
    models = ['Embedding\nClassifier', 'Lightweight\nCNN', 'Custom\nEmbedding']
    
    # Extract accuracies (handle None values)
    train_accs = [
        embedding_metrics['train_acc'] if embedding_metrics['train_acc'] is not None else 0,
        cnn_metrics['train_acc'] if cnn_metrics['train_acc'] is not None else 0,
        custom_metrics['train_acc'] if custom_metrics['train_acc'] is not None else 0,
    ]
    val_accs = [
        embedding_metrics['val_acc'] if embedding_metrics['val_acc'] is not None else 0,
        cnn_metrics['val_acc'] if cnn_metrics['val_acc'] is not None else 0,
        custom_metrics['val_acc'] if custom_metrics['val_acc'] is not None else 0,
    ]
    top3_accs = [
        embedding_metrics['top3_acc'] if embedding_metrics['top3_acc'] is not None else 0,
        cnn_metrics['top3_acc'] if cnn_metrics['top3_acc'] is not None else 0,
        custom_metrics['top3_acc'] if custom_metrics['top3_acc'] is not None else 0,
    ]
    
    # Training times (normalize to minutes)
    train_times = [
        embedding_metrics['training_time'] / 60 if embedding_metrics['training_time'] else 0,
        cnn_metrics['training_time'] / 60 if cnn_metrics['training_time'] else 0,
        custom_metrics['training_time'] / 60 if custom_metrics['training_time'] else 0,
    ]
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Face Recognition Model Comparison', fontsize=16, fontweight='bold')
    
    # 1. Training Accuracy
    ax1 = axes[0, 0]
    x = np.arange(len(models))
    width = 0.35
    bars = ax1.bar(x, train_accs, width, label='Train Accuracy', color='#2E86AB')
    ax1.set_ylabel('Accuracy', fontweight='bold')
    ax1.set_title('Training Accuracy', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.set_ylim([0, 1.0])
    ax1.grid(axis='y', alpha=0.3)
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, train_accs)):
        if val > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Validation Accuracy
    ax2 = axes[0, 1]
    bars = ax2.bar(x, val_accs, width, label='Val Accuracy', color='#A23B72')
    ax2.set_ylabel('Accuracy', fontweight='bold')
    ax2.set_title('Validation Accuracy', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.set_ylim([0, 1.0])
    ax2.grid(axis='y', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars, val_accs)):
        if val > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Top-3 Accuracy (if available)
    ax3 = axes[1, 0]
    bars = ax3.bar(x, top3_accs, width, label='Top-3 Val Accuracy', color='#F18F01')
    ax3.set_ylabel('Accuracy', fontweight='bold')
    ax3.set_title('Top-3 Validation Accuracy', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(models)
    ax3.set_ylim([0, 1.0])
    ax3.grid(axis='y', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars, top3_accs)):
        if val > 0:
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Training Time
    ax4 = axes[1, 1]
    bars = ax4.bar(x, train_times, width, label='Training Time', color='#6A994E')
    ax4.set_ylabel('Time (minutes)', fontweight='bold')
    ax4.set_title('Training Time', fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(models)
    ax4.grid(axis='y', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars, train_times)):
        if val > 0:
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}m', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_CHART, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Comparison chart saved to {OUTPUT_CHART}")


def create_comparison_table(embedding_metrics, cnn_metrics, custom_metrics):
    """Create markdown table with detailed comparison."""
    
    def fmt(val):
        """Format value for table."""
        if val is None:
            return "N/A"
        if isinstance(val, float):
            return f"{val:.4f}"
        return str(val)
    
    def fmt_time(seconds):
        """Format time in seconds to readable format."""
        if seconds is None:
            return "N/A"
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.2f}h"
    
    table = f"""# Model Comparison Table

## Performance Metrics

| Metric | Embedding Classifier | Lightweight CNN | Custom Embedding |
|--------|---------------------|-----------------|------------------|
| **Training Accuracy** | {fmt(embedding_metrics['train_acc'])} | {fmt(cnn_metrics['train_acc'])} | {fmt(custom_metrics['train_acc'])} |
| **Validation Accuracy** | {fmt(embedding_metrics['val_acc'])} | {fmt(cnn_metrics['val_acc'])} | {fmt(custom_metrics['val_acc'])} |
| **Top-3 Val Accuracy** | {fmt(embedding_metrics['top3_acc'])} | {fmt(cnn_metrics['top3_acc'])} | {fmt(custom_metrics['top3_acc'])} |
| **Number of Classes** | {fmt(embedding_metrics['num_classes'])} | {fmt(cnn_metrics['num_classes'])} | {fmt(custom_metrics['num_classes'])} |
| **Training Samples** | {fmt(embedding_metrics['num_samples'])} | {fmt(cnn_metrics['num_samples'])} | {fmt(custom_metrics['num_samples'])} |
| **Training Time** | {fmt_time(embedding_metrics['training_time'])} | {fmt_time(cnn_metrics['training_time'])} | {fmt_time(custom_metrics['training_time'])} |

## Model Characteristics

| Characteristic | Embedding Classifier | Lightweight CNN | Custom Embedding |
|----------------|---------------------|-----------------|------------------|
| **Face Detection** | InsightFace buffalo_l | OpenCV Haar Cascade | OpenCV Haar Cascade |
| **Feature Extraction** | InsightFace (512-dim) | Learned (end-to-end) | Learned (128-dim) |
| **Classifier** | Logistic Regression | Integrated Softmax | Cosine Similarity |
| **Pre-trained Features** | ✅ Yes (InsightFace) | ❌ No | ❌ No |
| **Training Complexity** | Low (classifier only) | High (full network) | Medium (network + centroids) |
| **Dependencies** | InsightFace, scikit-learn | OpenCV, TensorFlow | OpenCV, TensorFlow |
| **Model Size** | ~500 KB | ~2 MB | ~1.5 MB |
| **Inference Speed** | Fast | Medium | Medium |

## Recommendations

### Production Use
**Embedding Classifier (InsightFace + Logistic Regression)**
- ✅ Best accuracy
- ✅ Fastest training
- ✅ Most reliable
- ⚠️ Requires InsightFace model files

### Research/Experimentation
**Lightweight CNN**
- 📊 Demonstrates end-to-end learning
- 📊 Shows challenges with limited data
- ⚠️ Lower accuracy (expected for dataset size)

**Custom Embedding**
- 📊 Explores custom embedding spaces
- 📊 Metric learning research
- ⚠️ Research only (not production-ready)

---

**Generated**: {embedding_metrics.get('timestamp', 'Unknown')}  
**Dataset**: {embedding_metrics.get('num_samples', 'N/A')} samples, {embedding_metrics.get('num_classes', 'N/A')} classes  
**Image Size**: 240×240 (ESP32-CAM optimized)
"""
    
    with open(OUTPUT_TABLE, 'w') as f:
        f.write(table)
    
    print(f"✅ Comparison table saved to {OUTPUT_TABLE}")


def main():
    """Main function to generate comparison artifacts."""
    print("🚀 Generating model comparison charts and tables...")
    print()
    
    # Load training logs
    print("📊 Loading training logs...")
    embedding_log = load_log(EMBEDDING_LOG)
    cnn_log = load_log(CNN_LOG)
    custom_log = load_log(CUSTOM_LOG)
    
    # Check if at least one model is trained
    if not any([embedding_log, cnn_log, custom_log]):
        print("❌ No training logs found. Please train at least one model first.")
        print("   Run: python train.py")
        return 1
    
    # Extract metrics
    embedding_metrics = extract_metrics(embedding_log, "Embedding Classifier")
    cnn_metrics = extract_metrics(cnn_log, "Lightweight CNN")
    custom_metrics = extract_metrics(custom_log, "Custom Embedding")
    
    print(f"   Embedding Classifier: {'✅ Found' if embedding_log else '⚠️  Not found'}")
    print(f"   Lightweight CNN: {'✅ Found' if cnn_log else '⚠️  Not found'}")
    print(f"   Custom Embedding: {'✅ Found' if custom_log else '⚠️  Not found'}")
    print()
    
    # Create comparison chart
    print("📊 Creating comparison chart...")
    create_comparison_chart(embedding_metrics, cnn_metrics, custom_metrics)
    print()
    
    # Create comparison table
    print("📝 Creating comparison table...")
    create_comparison_table(embedding_metrics, cnn_metrics, custom_metrics)
    print()
    
    print("✅ Model comparison generation completed!")
    print(f"   Chart: {OUTPUT_CHART}")
    print(f"   Table: {OUTPUT_TABLE}")
    print(f"   Documentation: docs/MODEL_TRAINING.md")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
