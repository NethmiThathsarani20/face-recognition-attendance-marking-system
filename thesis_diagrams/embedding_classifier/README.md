# Embedding Classifier Model Diagrams and Visualizations

This directory contains comprehensive diagrams and visualizations for the **Embedding Classifier model** (InsightFace + Logistic Regression) - the **production model** with 99.89% validation accuracy.

## Model Performance Summary

- **Training Accuracy**: 99.9342%
- **Validation Accuracy**: 99.8948%
- **Top-3 Accuracy**: 100.00%
- **Number of Classes**: 66
- **Total Samples**: 9,504
- **Training Time**: ~30 seconds
- **Inference Time**: 80-100 ms

## Generated Diagrams

1. **proposed_methodology_diagram.png**
   - Proposed Methodology - Complete system workflow with InsightFace pipeline

2. **dataset_class_distribution.png**
   - Dataset Class Distribution - Statistics and sample distribution across 66 classes

3. **training_validation_accuracy.png**
   - Training vs Validation Accuracy - Bar chart and performance metrics

4. **training_validation_loss.png**
   - Training vs Validation Loss - Convergence curve for logistic regression

5. **classification_performance_table.png**
   - Classification Performance Table - Comprehensive training configuration and metrics

6. **overall_performance_metrics.png**
   - Overall Performance Metrics - Complete evaluation with interpretations

7. **classifier_architecture_diagram.png**
   - Classifier Architecture - InsightFace + Logistic Regression pipeline diagram


## Results and Discussion Chapter Figures

The following figures are referenced in the thesis Results and Discussion chapter:

8. **embedding_precision_recall_curve.png**
   - Precision-Recall Curve showing near-perfect performance
   - Area under curve ≈ 1.0
   - Demonstrates exceptional reliability at finding correct matches

9. **embedding_confidence_curve.png**
   - Confidence Distribution histogram
   - Sharp peak between 0.85-0.95 confidence scores
   - Justifies 60% threshold for rejecting "Unknown" individuals

10. **embedding_precision_confidence_curve.png**
    - Precision/Recall vs Confidence Threshold graph
    - Shows precision stays at 1.0 regardless of threshold
    - Identifies optimal threshold range (0.6-0.8) as "sweet spot"

11. **embedding_confusion_matrix.jpg**
    - Detailed Confusion Matrix with raw counts
    - Shows 1,930 validation samples with only 5 errors
    - Provides granular view of classification performance

12. **embedding_confusion_matrix_normalized.jpg**
    - Normalized Confusion Matrix (percentage view)
    - Deep blue diagonal confirms nearly 100% correct identification
    - Light-blue pixels outside diagonal represent 0.26% error rate

13. **model_accuracy_comparison.png**
    - Bar chart comparing validation accuracy across three models
    - Embedding Classifier: 99.74%, Custom Embedding: 98.86%, CNN: 64.04%
    - Demonstrates superiority of transfer learning approach

14. **inference_speed_comparison.png**
    - Real-time recognition speed comparison across models
    - Shows inference times: Embedding (80-100ms), Custom (90-110ms), CNN (120-150ms)
    - Validates real-time performance capability

15. **accuracy_vs_training_time.png**
    - Trade-off analysis between accuracy and training time
    - Highlights Embedding Classifier's optimal balance
    - Shows 99.74% accuracy achieved in only 30 seconds training time


## Model Architecture

The Embedding Classifier uses a two-stage approach:

1. **Feature Extraction** (InsightFace buffalo_l - frozen):
   - SCRFD face detector
   - 5-point landmark alignment
   - ArcFace ResNet-based embedding model
   - Output: 512-dimensional embeddings

2. **Classification** (Logistic Regression - trainable):
   - One-vs-Rest (OvR) multi-class strategy
   - SAGA solver with L2 regularization
   - ~34K parameters (66 classes × 512 features)
   - Fast training (~30 seconds)


## Why This Model is Used in Production

The Embedding Classifier is chosen as the production model because:

1. **Excellent Accuracy**: 99.89% validation accuracy, nearly perfect performance
2. **Fast Training**: Only ~30 seconds to train on 9,504 samples
3. **Lightweight**: ~200 KB model size, minimal memory footprint
4. **Real-time Inference**: 80-100 ms per face, suitable for live recognition
5. **Easy to Update**: Adding new users requires only retraining the classifier layer
6. **Proven Technology**: Uses state-of-the-art InsightFace pre-trained features


## Usage

All diagrams are generated at 300 DPI for high-quality printing and are suitable for inclusion in academic documents.

## Generation

To regenerate these diagrams, run:

```bash
# Generate methodology and architecture diagrams
python scripts/generate_classifier_diagrams.py

# Copy Results and Discussion figures from embedding_models/
python scripts/copy_results_figures.py
```

Note: The Results and Discussion figures (precision-recall curves, confidence distributions, and confusion matrices) are generated during model training. The `copy_results_figures.py` script copies them from `embedding_models/` to this directory for thesis inclusion.

## Comparison with Other Models

| Model | Val Accuracy | Top-3 Accuracy | Training Time | Inference Time |
|-------|-------------|----------------|---------------|----------------|
| **Embedding Classifier** (Production) | **99.89%** | **100%** | **30 sec** | **80-100 ms** |
| Custom Embedding | 99.00% | N/A | 2-3 min | 90-110 ms |
| Lightweight CNN | 57.23% | 77.49% | 32 min | 120-150 ms |
