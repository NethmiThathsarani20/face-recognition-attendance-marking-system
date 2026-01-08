# Thesis Diagrams - Complete Index

This directory contains all diagrams, charts, and visualizations for the Face Recognition Attendance System thesis.

## Main Focus: Embedding Classifier Model (Production)

The **Embedding Classifier** (InsightFace + Logistic Regression) is the production model used in the system.

### Performance Highlights
- **Validation Accuracy**: 99.89%
- **Top-3 Accuracy**: 100%
- **Training Time**: ~30 seconds
- **Inference Time**: 80-100 ms

### Generated Diagrams (High Priority)

All Embedding Classifier diagrams are located in `embedding_classifier/`:

1. **proposed_methodology_diagram.png** - System workflow and pipeline
2. **dataset_class_distribution.png** - Dataset statistics and class distribution
3. **training_validation_accuracy.png** - Accuracy metrics comparison
4. **training_validation_loss.png** - Loss convergence during training
5. **classification_performance_table.png** - Detailed performance table
6. **overall_performance_metrics.png** - Comprehensive metrics evaluation
7. **classifier_architecture_diagram.png** - Architecture diagram

**Location**: `thesis_diagrams/embedding_classifier/`

**See**: [Embedding Classifier README](embedding_classifier/README.md) for detailed documentation

---

## Training-Generated Visualizations

These are automatically generated during model training:

### Embedding Classifier (embedding_models/)
- **embedding_confusion_matrix.png** - Confusion matrix (unnormalized)
- **embedding_confusion_matrix_normalized.png** - Confusion matrix (normalized)
- **embedding_precision_recall_curve.png** - Precision-Recall curve
- **embedding_precision_confidence_curve.png** - Precision/Recall vs Confidence
- **embedding_confidence_curve.png** - Confidence distribution

### CNN Model (cnn_models/) - Experimental
- **cnn_confusion_matrix.png** - Confusion matrix
- **cnn_confusion_matrix_normalized.png** - Normalized confusion matrix
- **cnn_precision_recall_curve.png** - Precision-Recall curve
- **cnn_precision_confidence_curve.png** - Precision/Recall vs Confidence
- **cnn_confidence_curve.png** - Confidence distribution

### Custom Embedding Model (custom_embedding_models/) - Experimental
- **custom_embedding_confusion_matrix.png** - Confusion matrix
- **custom_embedding_confusion_matrix_normalized.png** - Normalized confusion matrix
- **custom_embedding_precision_recall_curve.png** - Precision-Recall curve
- **custom_embedding_precision_confidence_curve.png** - Precision/Recall vs Confidence
- **custom_embedding_confidence_curve.png** - Confidence distribution

---

## General System Diagrams

Located in the main `thesis_diagrams/` directory:

1. **system_architecture_diagram.png** - Three-tier architecture (Cloud, Edge, IoT)
2. **model_accuracy_comparison.png** - Comparison of all three models
3. **training_time_comparison.png** - Training time across models
4. **inference_speed_comparison.png** - Real-time inference speed
5. **accuracy_vs_training_time.png** - Accuracy vs Training Time trade-off
6. **temperature_performance_graph.png** - Raspberry Pi thermal performance
7. **lighting_accuracy_chart.png** - Recognition under different lighting
8. **cost_breakdown_pie.png** - Hardware cost breakdown
9. **annual_cost_comparison.png** - Cost comparison with alternatives
10. **roi_timeline.png** - Return on Investment timeline
11. **attendance_methods_comparison.png** - Attendance methods comparison table

---

## Quick Reference

### For Thesis Inclusion

**Main Model (Embedding Classifier)**:
- Architecture: `embedding_classifier/classifier_architecture_diagram.png`
- Methodology: `embedding_classifier/proposed_methodology_diagram.png`
- Performance: `embedding_classifier/classification_performance_table.png`
- Metrics: `embedding_classifier/overall_performance_metrics.png`
- Accuracy: `embedding_classifier/training_validation_accuracy.png`
- Confusion Matrix: `../embedding_models/embedding_confusion_matrix.png`

**System-Level**:
- Architecture: `system_architecture_diagram.png`
- Cost Analysis: `cost_breakdown_pie.png`, `annual_cost_comparison.png`, `roi_timeline.png`
- Comparison: `model_accuracy_comparison.png`

### Model Comparison

| Model | Val Accuracy | Training Time | Use Case |
|-------|-------------|---------------|----------|
| **Embedding Classifier** | **99.89%** | **30 sec** | **Production** |
| Custom Embedding | 99.00% | 2-3 min | Experimental |
| Lightweight CNN | 57.23% | 32 min | Experimental |

---

## Regenerating Diagrams

### Embedding Classifier Diagrams (Recommended)
```bash
python scripts/generate_classifier_diagrams.py
```

### General Thesis Diagrams
```bash
python scripts/generate_thesis_diagrams.py
```

---

## Image Specifications

- **Resolution**: 300 DPI (print quality)
- **Format**: PNG with transparency support
- **Color Space**: RGB
- **Quality**: Optimized for academic publications

---

## Additional Resources

- **Project Documentation**: See [PROJECT_DOCUMENTATION.md](../PROJECT_DOCUMENTATION.md)
- **Thesis Document**: See [THESIS.md](../THESIS.md)
- **Model Training**: See [docs/MODEL_TRAINING.md](../docs/MODEL_TRAINING.md)

---

## Citation

When using these diagrams, please cite:

```
Face Recognition Based Attendance Marking System Using IoT Devices
InsightFace + Embedding Classifier Approach
Nethmi Thathsarani, 2026
```

---

Last Updated: January 2026
