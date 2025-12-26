# Model Training Documentation

## Overview

This document provides comprehensive details about the three face recognition models used in this project, their training procedures, evaluation metrics, and comparative analysis.

## Three Model Architectures

### 1. Embedding Classifier (InsightFace + Logistic Regression)
**Production Model - Recommended for deployment**

- **Architecture**: Pre-trained InsightFace (buffalo_l) for feature extraction + Logistic Regression classifier
- **Face Detection**: InsightFace buffalo_l model
- **Feature Extraction**: 512-dimensional embeddings from InsightFace
- **Classification**: Multinomial Logistic Regression with L2 regularization
- **Training Dataset**: Balanced dataset with oversampling
- **Key Features**:
  - Uses frozen InsightFace embeddings (no training of feature extractor)
  - Fast training (only classifier is trained)
  - Excellent generalization due to pre-trained features
  - Class weighting for imbalanced datasets
  - OneVsRest strategy for robust multi-class support

**Training Command**:
```bash
python train.py --only embedding --epochs 30 --validation-split 0.2
```

**Use Case**: Production deployment, best accuracy, minimal training time

---

### 2. Lightweight CNN (End-to-End)
**Experimental Model - For research purposes**

- **Architecture**: Custom lightweight CNN trained from scratch
- **Face Detection**: OpenCV Haar Cascade (no InsightFace)
- **Feature Extraction**: Learned during training
- **Classification**: Integrated softmax classifier
- **Training Dataset**: Balanced dataset with oversampling
- **Key Features**:
  - Complete end-to-end training
  - No dependency on pre-trained models
  - Separable convolutions for reduced parameters
  - Data augmentation (flip, rotation, zoom, contrast)
  - Batch normalization and dropout for regularization
  - Early stopping and learning rate reduction

**Network Architecture**:
```
Input (112x112x3)
  ↓
Data Augmentation (flip, rotate, zoom, contrast)
  ↓
SeparableConv2D (32 filters) → BatchNorm → MaxPool → Dropout(0.2)
  ↓
SeparableConv2D (64 filters) → BatchNorm → MaxPool → Dropout(0.3)
  ↓
SeparableConv2D (128 filters) → BatchNorm → MaxPool → Dropout(0.4)
  ↓
GlobalAveragePooling2D
  ↓
Dense(256) → BatchNorm → Dropout(0.5)
  ↓
Dense(128) → Dropout(0.4)
  ↓
Dense(num_classes, softmax)
```

**Training Command**:
```bash
python train.py --only cnn --epochs 50 --validation-split 0.2 \
  --early-stopping-patience 8 --reduce-lr-patience 4
```

**Use Case**: Research, completely standalone system without pre-trained models

---

### 3. Custom Embedding Model
**Experimental Model - For research purposes**

- **Architecture**: Custom CNN backbone + embedding layer + cosine similarity classifier
- **Face Detection**: OpenCV Haar Cascade (no InsightFace)
- **Feature Extraction**: Learned 128-dimensional embeddings
- **Classification**: Cosine similarity to class centroids
- **Training Dataset**: Balanced dataset with oversampling
- **Key Features**:
  - Learns custom embeddings independently
  - L2-normalized embedding space
  - Centroid-based classification
  - No dependency on InsightFace embeddings
  - Sparse categorical cross-entropy loss
  - Early stopping and learning rate reduction

**Network Architecture**:
```
Input (112x112x3)
  ↓
SeparableConv2D (32 filters) → BatchNorm → MaxPool
  ↓
SeparableConv2D (64 filters) → BatchNorm → MaxPool
  ↓
SeparableConv2D (128 filters) → BatchNorm → GlobalAveragePooling
  ↓
Dense(256) → Dropout(0.4)
  ↓
Dense(128, no activation) [embedding layer]
  ↓
L2 Normalization
  ↓
Dense(num_classes, softmax) [for training only]

At inference: Use cosine similarity between normalized embeddings and class centroids
```

**Training Command**:
```bash
python train.py --only custom-embedding --epochs 30 --validation-split 0.2 \
  --embedding-dim 128
```

**Use Case**: Research, custom embedding space exploration

---

## Dataset Specifications

### Current Dataset
- **Location**: `database/`
- **Total Users**: 67
- **Total Images**: 1,595 (after balancing, varies by model)
- **Image Size**: 240×240 pixels (optimized for ESP32-CAM)
- **Format**: JPEG with 95% quality
- **Balancing**: Oversampling applied to ensure equal samples per class

### Dataset Preparation
Each model performs its own balancing:
- Identifies class with maximum samples
- Oversamples minority classes (with replacement)
- Ensures equal representation during training
- Splits into train/validation sets (80/20 by default)

---

## Evaluation Metrics

All three models generate the following evaluation artifacts:

### 1. Confusion Matrices
- **Regular Confusion Matrix**: Shows absolute prediction counts
- **Normalized Confusion Matrix**: Shows proportions (0-1 scale)
- **Format**: High-resolution PNG (300 DPI for normalized, 150 DPI for regular)
- **Features**: 
  - Color-coded cells (darker = more predictions)
  - Cell annotations with counts/proportions
  - Improved readability for many classes (dynamic font sizing)
  - Grid lines for better visibility

### 2. Precision-Recall Curve
- **Type**: Micro-averaged across all classes
- **Purpose**: Evaluate trade-off between precision and recall
- **X-axis**: Recall (proportion of true positives identified)
- **Y-axis**: Precision (proportion of positive predictions that are correct)
- **Use**: Assess overall classification quality

### 3. Precision/Recall vs Confidence Threshold
- **Purpose**: Understand impact of confidence threshold on metrics
- **X-axis**: Confidence threshold (0.0 to 1.0)
- **Y-axis**: Precision and Recall scores
- **Lines**: 
  - Precision: How accurate predictions are above threshold
  - Recall: What proportion of samples are classified above threshold
- **Use**: Select optimal threshold for deployment

### 4. Confidence Distribution
- **Type**: Histogram of predicted maximum confidences
- **Purpose**: Understand model certainty
- **X-axis**: Predicted confidence (0.0 to 1.0)
- **Y-axis**: Count of validation samples
- **Bins**: 20 bins across [0, 1]
- **Use**: Identify if model is well-calibrated or over/under-confident

---

## Training Logs

Each model generates a JSON training log with the following information:

### Common Fields
```json
{
  "timestamp": "2024-12-26T12:34:56",
  "num_classes": 67,
  "num_samples": 1595,
  "validation_split": 0.2,
  "train_accuracy": 0.994,
  "val_accuracy": 0.984,
  "classes": ["Abdullah_Gul", "Adithya", ...]
}
```

### Model-Specific Fields

**Embedding Classifier**:
```json
{
  "solver": "saga",
  "penalty": "l2",
  "C": 1.0,
  "max_iter": 2000,
  "val_top3_accuracy": 0.984,
  "stratified_split": true
}
```

**CNN Model**:
```json
{
  "epochs": 50,
  "training_time_seconds": 1234.56,
  "final_train_loss": 0.123,
  "final_val_loss": 0.234,
  "final_train_top3_accuracy": 0.95,
  "final_val_top3_accuracy": 0.89
}
```

**Custom Embedding**:
```json
{
  "embedding_dim": 128,
  "epochs": 30
}
```

---

## Model Comparison

### Performance Summary

| Model | Train Acc | Val Acc | Top-3 Val Acc | Training Time | Model Size |
|-------|-----------|---------|---------------|---------------|------------|
| **Embedding Classifier** | 0.994 | 0.984 | 0.984 | ~30 sec | ~500 KB |
| **Lightweight CNN** | 0.039 | 0.039 | 0.061 | ~5-10 min | ~2 MB |
| **Custom Embedding** | 0.010 | 0.010 | N/A | ~2-3 min | ~1.5 MB |

*Note: Actual metrics depend on dataset and training configuration. Run training to get exact values.*

### Key Insights

**Embedding Classifier (Production)**:
- ✅ **Best Performance**: Leverages pre-trained InsightFace features
- ✅ **Fast Training**: Only classifier needs training
- ✅ **Reliable**: Proven architecture with robust features
- ⚠️ **Dependency**: Requires InsightFace model files (~50 MB)

**Lightweight CNN (Experimental)**:
- ⚠️ **Lower Accuracy**: Limited by dataset size (1,595 samples for 67 classes)
- ✅ **Independent**: No pre-trained model dependency
- ✅ **Lightweight**: Smaller model size
- ⚠️ **Slow Training**: Requires more epochs and time
- 📊 **Research Value**: Useful for understanding limitations of end-to-end training

**Custom Embedding (Experimental)**:
- ⚠️ **Lowest Accuracy**: Limited by dataset size and training from scratch
- ✅ **Independent**: No InsightFace dependency
- ✅ **Flexible**: Custom embedding dimension
- ⚠️ **Research Only**: Not suitable for production
- 📊 **Learning Tool**: Demonstrates embedding-based recognition without pre-trained features

---

## Recommendations

### For Production Deployment
**Use the Embedding Classifier (InsightFace + Logistic Regression)**

Reasons:
1. Highest accuracy (>98% validation)
2. Fast training and inference
3. Robust pre-trained features
4. Proven architecture
5. Good generalization

### For Research and Experimentation
**Try all three models to understand trade-offs**

Use cases:
- **CNN**: Understanding end-to-end learning challenges
- **Custom Embedding**: Exploring embedding spaces and metric learning
- **Comparison**: Demonstrates value of transfer learning

### For Resource-Constrained Devices (ESP32)
**Use Embedding Classifier with edge deployment considerations**

Recommendations:
1. Run InsightFace on Raspberry Pi or cloud
2. Send embeddings to ESP32-CAM for lightweight matching
3. Alternatively, use simpler CNN if offline operation required
4. Optimize image size (240×240 balances quality and memory)

---

## Training Procedures

### Initial Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Verify database
ls -la database/
```

### Train All Models
```bash
# Train all three models with default settings
python train.py --epochs 30 --validation-split 0.2

# Or train individually
python train.py --only embedding
python train.py --only cnn --epochs 50
python train.py --only custom-embedding
```

### Cloud Training (GitHub Actions)
The repository includes automated cloud training:
- Triggers on database changes
- Trains all models automatically
- Commits results back to repo
- See `.github/workflows/train.yml`

### Edge Training (Raspberry Pi)
For manual edge training:
```bash
# On Raspberry Pi
cd face-recognition-attendance-marking-system
python train.py --epochs 30 --validation-split 0.2

# Push results (optional)
git add cnn_models/ embedding_models/ custom_embedding_models/
git commit -m "Update models from edge training"
git push
```

---

## Model Artifacts

### Embedding Classifier
**Directory**: `embedding_models/`
- `embedding_classifier.joblib` - Trained LogisticRegression model
- `label_encoder.pkl` - Class label encoder
- `training_log.json` - Training metadata
- `embedding_confusion_matrix.png`
- `embedding_confusion_matrix_normalized.png`
- `embedding_precision_recall_curve.png`
- `embedding_precision_confidence_curve.png`
- `embedding_confidence_curve.png`

### Lightweight CNN
**Directory**: `cnn_models/`
- `custom_face_model.keras` - Trained Keras model
- `label_encoder.pkl` - Class label encoder
- `training_log.json` - Training metadata
- `cnn_confusion_matrix.png`
- `cnn_confusion_matrix_normalized.png`
- `cnn_precision_recall_curve.png`
- `cnn_precision_confidence_curve.png`
- `cnn_confidence_curve.png`

### Custom Embedding
**Directory**: `custom_embedding_models/`
- `custom_embedding_model.keras` - Trained Keras model
- `label_encoder.pkl` - Class label encoder
- `class_centroids.npy` - Per-class embedding centroids
- `training_log.json` - Training metadata
- `custom_embedding_confusion_matrix.png`
- `custom_embedding_confusion_matrix_normalized.png`
- `custom_embedding_precision_recall_curve.png`
- `custom_embedding_precision_confidence_curve.png`
- `custom_embedding_confidence_curve.png`

---

## Troubleshooting

### Low Accuracy
- **Symptom**: Val accuracy < 50%
- **Solutions**:
  1. Increase training epochs
  2. Add more training images per user
  3. Check data quality (blur, lighting, occlusion)
  4. Verify face detection is working
  5. Use Embedding Classifier for best results

### Out of Memory (OOM)
- **Symptom**: Training crashes with memory error
- **Solutions**:
  1. Reduce batch size (--batch-size 16 or 8)
  2. Reduce image size (already 240×240)
  3. Train on cloud/desktop instead of Pi
  4. Use embedding classifier (lower memory)

### Long Training Time
- **Symptom**: Training takes hours
- **Solutions**:
  1. Use cloud training (GitHub Actions)
  2. Reduce epochs for CNN (--epochs 30)
  3. Enable early stopping
  4. Use embedding classifier (fastest)

### Model Not Found
- **Symptom**: Application can't load model
- **Solutions**:
  1. Check model directories exist
  2. Run training to generate models
  3. Verify file permissions
  4. Check logs for errors

---

## References

1. **InsightFace**: https://github.com/deepinsight/insightface
2. **Face Recognition Best Practices**: https://arxiv.org/abs/1804.06655
3. **Metric Learning**: https://arxiv.org/abs/1503.03832
4. **Transfer Learning**: https://cs231n.github.io/transfer-learning/

---

**Last Updated**: 2024-12-26  
**Dataset Version**: 67 users, 1,595 images (240×240)  
**Recommended Model**: Embedding Classifier (InsightFace + Logistic Regression)
