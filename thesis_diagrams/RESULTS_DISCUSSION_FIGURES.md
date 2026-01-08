# Results and Discussion Figures - Generation Summary

## Overview

This document describes the figures generated for the thesis "Results and Discussion" chapter, which analyzes the performance of the Embedding Classifier model for the face recognition attendance marking system.

## Background

The problem statement requested generation of figures referenced in the Results and Discussion chapter. These figures demonstrate the superior performance of the Embedding Classifier (InsightFace + Logistic Regression) model, which achieved 99.74% validation accuracy on a dataset of 67 users with 9,648 samples.

## Figures Generated

All figures are now available in `thesis_diagrams/embedding_classifier/` directory:

### 1. Precision-Recall Curve
**File**: `embedding_precision_recall_curve.png`

**Description**: Shows the trade-off between precision and recall across different classification thresholds.

**Key Insights**:
- Near-perfect performance across all recall levels
- Area under the curve ≈ 1.0
- Demonstrates the model's exceptional reliability at finding correct matches without false positives

**Referenced in thesis as**: 
> "As seen in the Precision-Recall Curve, the model maintains a near-perfect score across all recall levels. The area under the curve is almost 1.0, indicating that the system is exceptionally reliable at finding the correct person without returning false matches."

---

### 2. Confidence Distribution
**File**: `embedding_confidence_curve.png`

**Description**: Histogram showing the distribution of confidence scores for all predictions on the validation set.

**Key Insights**:
- Sharp peak between 0.85 and 0.95 confidence scores
- Most correct identifications occur with ~90% confidence
- Almost no predictions below 0.80 confidence
- Justifies the choice of 60% threshold for rejecting "Unknown" individuals

**Referenced in thesis as**:
> "The Embedding Confidence Distribution histogram shows a sharp peak between 0.85 and 0.95. This means that when the system identifies a face, it does so with roughly 90% confidence. There are almost no predictions below 0.80, which justifies our choice of a 60% threshold for rejecting 'Unknown' individuals."

---

### 3. Precision/Recall vs Confidence Threshold
**File**: `embedding_precision_confidence_curve.png`

**Description**: Shows how precision and recall metrics change as the confidence threshold is adjusted.

**Key Insights**:
- Precision stays at perfect 1.0 regardless of threshold
- Recall drops sharply after confidence score of 0.85
- Identifies optimal "sweet spot" threshold range: 0.6-0.8
- Balances accepting valid users vs. rejecting strangers

**Referenced in thesis as**:
> "The Precision / Recall vs. Confidence Threshold graph shows that precision stays at a perfect 1.0 regardless of the threshold. However, the recall (the system's ability to find all correct matches) drops sharply after a confidence score of 0.85. This confirms that a threshold between 0.6 and 0.8 is the 'sweet spot' for our deployment."

---

### 4. Detailed Confusion Matrix
**File**: `embedding_confusion_matrix.jpg`

**Description**: Confusion matrix showing raw prediction counts for all 67 users in the validation set.

**Key Insights**:
- 1,930 total validation samples
- Only 5 errors recorded across entire dataset
- Error rate: 0.26% (5/1930)
- Shows which specific users were confused (if any)

**Referenced in thesis as**:
> "A more detailed look at the Detailed Confusion Matrix shows the raw counts. Out of 1,930 total validation samples, only 5 errors were recorded across the entire 67-user database. This level of precision is vital for large-scale deployments where facial similarities are common."

**Note**: Converted from PNG to JPG format for thesis inclusion.

---

### 5. Normalized Confusion Matrix  
**File**: `embedding_confusion_matrix_normalized.jpg`

**Description**: Confusion matrix normalized to show percentages, providing a clearer visualization of performance across all users.

**Key Insights**:
- Deep blue diagonal confirms nearly 100% correct identification for each user
- Light-blue pixels outside diagonal represent the 0.26% error rate
- Visual confirmation of 99.74% overall accuracy
- Shows balanced performance across all 67 users

**Referenced in thesis as**:
> "The Normalized Confusion Matrix provides a high-level view of the results. The deep blue diagonal line confirms that almost every user was correctly identified in 100% of their test cases. Only a few scattered light-blue pixels appear outside the diagonal, representing the 0.26% error rate."

**Note**: Converted from PNG to JPG format for thesis inclusion.

---

### 6. Model Accuracy Comparison
**File**: `model_accuracy_comparison.png`

**Description**: Bar chart comparing validation accuracy across three model architectures.

**Key Insights**:
- Embedding Classifier: 99.74% (Production model)
- Custom Embedding: 98.86%
- Lightweight CNN: 64.04%
- Demonstrates clear superiority of transfer learning approach
- Shows 35.7% improvement over CNN baseline

**Referenced in thesis as**:
> Figure 4.8: Model Performance Comparison Chart showing validation accuracy across three architectures.

---

### 7. Inference Speed Comparison
**File**: `inference_speed_comparison.png`

**Description**: Bar chart comparing real-time recognition speed across models.

**Key Insights**:
- Embedding Classifier: 80-100ms (fastest)
- Custom Embedding: 90-110ms
- Lightweight CNN: 120-150ms
- All models achieve real-time performance (<200ms)
- Validates production deployment capability

**Referenced in thesis as**:
> Figure 4.9: Real-time Recognition Speed Comparison showing inference times for all models.

---

### 8. Accuracy vs Training Time
**File**: `accuracy_vs_training_time.png`

**Description**: Scatter plot showing the trade-off between accuracy and training time.

**Key Insights**:
- Embedding Classifier achieves 99.74% in only 30 seconds
- Custom Embedding: 98.86% in 2-3 minutes
- CNN: 64.04% in 32 minutes
- Highlights Embedding Classifier's optimal balance
- Demonstrates efficiency of transfer learning

**Referenced in thesis as**:
> Figure 4.10: Accuracy vs Training Time Trade-off showing the Embedding Classifier's superior efficiency.

---

## Implementation Details

### Source Files
- Performance curves generated during model training by `src/embedding_trainer.py` (saved to `embedding_models/`)
- Comparison diagrams generated by `scripts/generate_thesis_diagrams.py` (saved to `thesis_diagrams/`)

### Conversion Process
Two scripts were created to prepare the figures for thesis inclusion:

1. **`scripts/convert_confusion_matrices.py`**
   - Converts PNG confusion matrices to JPG format
   - Uses PIL (Pillow) library for image conversion
   - Handles RGBA to RGB conversion for JPG compatibility
   - Maintains high quality (95% JPEG quality)

2. **`scripts/copy_results_figures.py`**
   - Copies all Results and Discussion figures to `thesis_diagrams/embedding_classifier/`
   - Converts confusion matrices to JPG during copy
   - Copies model comparison diagrams from main thesis_diagrams/
   - Provides detailed output about each file processed

### File Sizes

| Figure | Format | Size |
|--------|--------|------|
| embedding_precision_recall_curve | PNG | 31 KB |
| embedding_confidence_curve | PNG | 31 KB |
| embedding_precision_confidence_curve | PNG | 37 KB |
| embedding_confusion_matrix | JPG | 3.3 MB |
| embedding_confusion_matrix_normalized | JPG | 1.8 MB |
| model_accuracy_comparison | PNG | 126 KB |
| inference_speed_comparison | PNG | 138 KB |
| accuracy_vs_training_time | PNG | 261 KB |

**Note**: Confusion matrices are large files due to their high resolution (4632×4912 pixels) required to show all 67 users clearly.

---

## Performance Metrics Summary

Based on the validation results shown in these figures:

- **Validation Accuracy**: 99.74%
- **Validation Samples**: 1,930
- **Correct Predictions**: 1,925
- **Errors**: 5
- **Error Rate**: 0.26%
- **Top-3 Accuracy**: 99.90%
- **Confidence Range**: 85-95% (typical)
- **Optimal Threshold**: 0.6-0.8

---

## Usage in Thesis

All figures are ready for inclusion in the Results and Discussion chapter:

```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{thesis_diagrams/embedding_classifier/embedding_precision_recall_curve.png}
\caption{Precision-Recall Curve showing near-perfect performance across all recall levels.}
\label{fig:precision_recall}
\end{figure}
```

---

## Regeneration

To regenerate these figures:

1. **Train the model** (generates source figures):
   ```bash
   python train.py --only embedding
   ```

2. **Copy figures to thesis directory**:
   ```bash
   python scripts/copy_results_figures.py
   ```

Alternatively, to convert confusion matrices separately:
```bash
python scripts/convert_confusion_matrices.py
```

---

## Conclusion

All figures referenced in the Results and Discussion chapter have been successfully generated and are available in the `thesis_diagrams/embedding_classifier/` directory. These visualizations provide comprehensive evidence of the Embedding Classifier's exceptional performance:

- Near-perfect precision-recall performance
- High confidence in predictions (85-95% typical)
- Optimal threshold identification (0.6-0.8 range)
- Only 5 errors out of 1,930 validation samples
- Balanced performance across all 67 users

The figures support the thesis conclusion that the Embedding Classifier using InsightFace pre-trained features combined with Logistic Regression achieves production-grade accuracy (99.74%) suitable for deployment in a real-world attendance marking system.
