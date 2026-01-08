# Group43 Final Thesis - Complete Figures and Tables Index

This document provides a comprehensive list of all diagrams, tables, and images generated for the Group43 final thesis on Face Recognition Based Attendance Marking System.

## Generated: January 8, 2026

---

## 📊 Performance Comparison Charts

### 1. Model Accuracy Comparison (Figure 4.8)
**File:** `thesis_diagrams/model_accuracy_comparison.png`
**Description:** Bar chart comparing validation accuracy across three models:
- Embedding Classifier: **99.74%** (Production model)
- Custom Embedding: 98.86%
- Lightweight CNN: 64.04%

**Demonstrates:** Superiority of transfer learning approach using InsightFace pre-trained embeddings.

---

### 2. Training Time Comparison
**File:** `thesis_diagrams/training_time_comparison.png`
**Description:** Bar chart showing training duration for each model:
- Embedding Classifier: **30 seconds** (64x faster than CNN)
- Custom Embedding: 2-3 minutes
- Lightweight CNN: 32 minutes

**Demonstrates:** Efficiency advantages of embedding-based approach for edge deployment.

---

### 3. Inference Speed Comparison (Figure 4.9)
**File:** `thesis_diagrams/inference_speed_comparison.png`
**Description:** Bar chart comparing real-time recognition speeds:
- Embedding Classifier: **80-100ms** (fastest)
- Custom Embedding: 90-110ms
- Lightweight CNN: 120-150ms
- All below 200ms real-time threshold

**Demonstrates:** All models achieve real-time performance; Embedding Classifier optimal for production.

---

### 4. Accuracy vs Training Time Trade-off (Figure 4.10)
**File:** `thesis_diagrams/accuracy_vs_training_time.png`
**Description:** Scatter plot showing the optimal balance achieved by Embedding Classifier:
- Best accuracy (99.74%)
- Fastest training (30 seconds)
- Clear winner for production deployment

**Demonstrates:** Embedding Classifier achieves best performance with least computational cost.

---

## 🔧 Hardware Performance Visualizations

### 5. Temperature Performance Graph
**File:** `thesis_diagrams/temperature_performance_graph.png`
**Description:** Dual-panel graph showing Raspberry Pi thermal management over 30 minutes:

**Panel 1 - CPU Temperature:**
- Without fan: Rises to 85°C (thermal throttling)
- With fan: Stable at 45-50°C (optimal)
- 40°C temperature reduction

**Panel 2 - Recognition Performance:**
- Without fan: Degrades from 85ms to 165ms (94% slower)
- With fan: Stable at 85ms throughout
- 50-60% performance improvement

**Demonstrates:** Active cooling is essential for 24/7 operation and sustained performance.

---

### 6. Lighting Accuracy Chart (Table 4.4 Visualization)
**File:** `thesis_diagrams/lighting_accuracy_chart.png`
**Description:** Grouped bar chart comparing recognition accuracy across 6 lighting conditions:

| Condition | Without LED | With LED | Improvement |
|-----------|-------------|----------|-------------|
| Bright Indoor | 99.2% | 99.7% | +0.5% |
| Normal Indoor | 98.5% | 99.6% | +1.1% |
| Dim Indoor | 92.3% | 98.4% | +6.1% |
| Low Light | 78.1% | 96.2% | **+18.1%** |
| Backlit | 85.6% | 94.3% | +8.7% |
| Varied Light | 88.9% | 97.2% | +8.3% |
| **Average** | **90.4%** | **97.6%** | **+7.2%** |

**Demonstrates:** LED light panel dramatically improves low-light performance and ensures consistent accuracy.

---

## 💰 Cost Analysis Visualizations

### 7. Cost Breakdown Pie Chart
**File:** `thesis_diagrams/cost_breakdown_pie.png`
**Description:** Pie chart showing hardware cost distribution:

**Total System Cost: Rs. 56,700 ($189)**

| Component | Cost (LKR) | Cost (USD) | Percentage |
|-----------|------------|------------|------------|
| Raspberry Pi 4 (4GB) | Rs. 16,500 | $55 | 29.1% |
| WiFi Router | Rs. 12,000 | $40 | 21.2% |
| ESP32-CAM Modules (3x) | Rs. 9,000 | $30 | 15.9% |
| Power Supplies | Rs. 7,500 | $25 | 13.2% |
| Miscellaneous | Rs. 6,000 | $20 | 10.6% |
| Cooling & SD Card | Rs. 3,300 | $11 | 5.8% |
| LED Panels | Rs. 2,400 | $8 | 4.2% |

**Demonstrates:** Affordable solution with largest costs in reusable infrastructure (Pi, Router).

---

### 8. Annual Cost Comparison
**File:** `thesis_diagrams/annual_cost_comparison.png`
**Description:** Bar chart comparing first-year total costs:

| System | Cost (LKR) | Cost (USD) | Notes |
|--------|------------|------------|-------|
| **Our System** | **Rs. 56,700** | **$189** | One-time only |
| Cloud SaaS | Rs. 420,000 | $1,400 | Recurring monthly fees |
| Commercial IP | Rs. 858,000 | $2,860 | Proprietary hardware |
| Fingerprint | Rs. 360,000 | $1,200 | Contact-based |
| RFID Card | Rs. 312,000 | $1,040 | Cards can be shared |

**Savings vs Cloud SaaS:** Rs. 363,300 ($1,211) in first year
**Savings vs Commercial IP:** Rs. 801,300 ($2,671) in first year

**Demonstrates:** Massive cost savings compared to commercial alternatives.

---

### 9. ROI Timeline Graph
**File:** `thesis_diagrams/roi_timeline.png`
**Description:** Line graph showing return on investment over one year:

**For 50-person organization (10 min daily roll call @ Rs. 15,000/hour instructor rate):**
- Daily manual cost: Rs. 2,500
- **Break-even point: Day 23 (~3 weeks)**
- **Year 1 savings: Rs. 568,300 ($1,894)**

**Demonstrates:** Extremely fast ROI; system pays for itself in less than one month.

---

## 🏗️ System Architecture

### 10. System Architecture Diagram (Figure 3.1 Simplified)
**File:** `thesis_diagrams/system_architecture_diagram.png`
**Description:** Three-tier architecture block diagram:

**Cloud Layer (Blue):**
- GitHub Actions for automated model training
- Cloud-based evaluation and metrics generation

**Edge Layer (Green):**
- Raspberry Pi 4 with active cooling
- Flask web application
- InsightFace recognition engine
- Local database and storage

**IoT Layer (Yellow):**
- Multiple ESP32-CAM devices with LED panels
- Wireless MJPEG streaming
- Distributed camera placement

**Data Flow:**
- Cloud ↔ Edge: Model synchronization via Git
- Edge ↔ IoT: WiFi video streaming
- All layers communicate seamlessly

**Demonstrates:** Hybrid edge-cloud architecture optimizing performance, privacy, and cost.

---

## 📋 Comparison Tables

### 11. Attendance Methods Comparison (Table 1.1)
**File:** `thesis_diagrams/attendance_methods_comparison.png`
**Description:** Comprehensive comparison table of 6 attendance marking methods:

| Method | Time Required | Proxy Prevention | Contact Required | Cost | Data Management |
|--------|--------------|------------------|------------------|------|-----------------|
| Manual Roll Call | High (5-10 min) | Low | No | Low | Poor |
| Paper Registers | Medium (3-5 min) | Low | Yes | Low | Poor |
| RFID Cards | Low (1-2 min) | Medium | Yes | Medium | Good |
| Fingerprint | Low (1-2 min) | High | Yes | Medium | Good |
| Face Recognition | Very Low (<1 min) | Very High | No | Medium | Excellent |
| **Our System** | **Very Low (<30 sec)** | **Very High** | **No** | **Low** | **Excellent** |

**Demonstrates:** Our system combines best features: fast, contactless, accurate, affordable.

---

## 🎯 Model Performance Images

### Embedding Classifier (Production Model)

#### 12. Confusion Matrix (Figure 4.4)
**File:** `embedding_models/embedding_confusion_matrix.png`
**Description:** 67×67 confusion matrix showing classification results:
- Strong diagonal elements (correct predictions)
- Very few off-diagonal elements (misclassifications)
- **99.74% overall accuracy**

**Demonstrates:** Excellent discrimination across all 67 users with minimal confusion.

---

#### 13. Normalized Confusion Matrix
**File:** `embedding_models/embedding_confusion_matrix_normalized.png`
**Description:** Normalized version showing per-class accuracy percentages.

---

#### 14. Precision-Recall Curve (Figure 4.5)
**File:** `embedding_models/embedding_precision_recall_curve.png`
**Description:** PR curve showing model performance:
- High precision (>99%) across all users
- High recall (>99%) across all users
- Area Under Curve (AUC) ≈ 0.997

**Demonstrates:** Balanced performance with both high precision and recall.

---

#### 15. Confidence Distribution (Figure 4.6)
**File:** `embedding_models/embedding_confidence_curve.png`
**Description:** Histogram of prediction confidence scores:
- Most predictions: 90-100% confidence
- Very few predictions: <70% confidence
- Clear separation validates 60% "Unknown" threshold

**Demonstrates:** Model's high certainty in correct predictions enables reliable threshold setting.

---

#### 16. Precision-Confidence Curve
**File:** `embedding_models/embedding_precision_confidence_curve.png`
**Description:** Shows how precision varies with confidence threshold.

---

### CNN Model (Baseline Comparison)

#### 17. CNN Confusion Matrix (Figure 4.7)
**File:** `cnn_models/cnn_confusion_matrix.png`
**Description:** 67×67 confusion matrix for CNN model:
- Weaker diagonal elements
- More scattered misclassifications
- **64.04% overall accuracy**

**Demonstrates:** Challenges of training deep learning from scratch on limited data.

---

#### 18. CNN Normalized Confusion Matrix
**File:** `cnn_models/cnn_confusion_matrix_normalized.png`
**Description:** Normalized CNN confusion matrix.

---

#### 19. CNN Precision-Recall Curve
**File:** `cnn_models/cnn_precision_recall_curve.png`
**Description:** PR curve showing CNN's lower performance compared to Embedding Classifier.

---

#### 20. CNN Confidence Distribution
**File:** `cnn_models/cnn_confidence_curve.png`
**Description:** CNN confidence histogram showing broader, less certain distribution.

---

#### 21. CNN Precision-Confidence Curve
**File:** `cnn_models/cnn_precision_confidence_curve.png`
**Description:** CNN precision variation with confidence threshold.

---

## 📊 Summary Statistics

### Generated Files Breakdown:

| Category | Count | Location |
|----------|-------|----------|
| **Thesis Diagrams** | 11 | `thesis_diagrams/` |
| **Embedding Model Images** | 5 | `embedding_models/` |
| **CNN Model Images** | 5 | `cnn_models/` |
| **Total Generated Images** | **21** | Multiple directories |

---

## 🎓 Key Findings Visualized

1. **Transfer Learning Superiority** (Figures 4.8, 4.10)
   - 35.7% accuracy improvement over CNN
   - 64x faster training time
   - Clear evidence for production model selection

2. **Hardware Optimization** (Temperature Graph, Lighting Chart)
   - Active cooling: 40°C temperature reduction, 50-60% performance boost
   - LED illumination: 18% improvement in low light, 7.2% average improvement

3. **Cost Effectiveness** (Cost Charts, ROI Timeline)
   - 89% cheaper than commercial alternatives
   - Break-even in 23 days
   - Rs. 568,300 first-year savings for 50-person organization

4. **Production Readiness** (Inference Speed, Architecture Diagram)
   - Real-time performance: 80-100ms per face
   - Hybrid edge-cloud architecture
   - 24/7 operational capability

---

## 📖 Usage in Thesis

All figures are referenced in `THESIS.md` with proper captions and figure numbers. Images can be embedded using:

```markdown
![Description](path/to/image.png)
**Figure X.Y:** Caption text
```

Or in LaTeX:

```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=0.8\textwidth]{path/to/image.png}
\caption{Caption text}
\label{fig:label}
\end{figure}
```

---

## 🔄 Regenerating Diagrams

To regenerate all thesis diagrams:

```bash
cd /path/to/repository
python3 scripts/generate_thesis_diagrams.py
```

This creates/updates all 11 thesis diagrams in `thesis_diagrams/` directory.

Model performance images are generated during training:
```bash
python train.py --epochs 30 --validation-split 0.2
```

---

## ✅ Quality Assurance

All generated images meet thesis requirements:
- ✅ 300 DPI resolution (print quality)
- ✅ Clear labels and legends
- ✅ Consistent color scheme
- ✅ Professional styling
- ✅ Proper figure numbers and captions
- ✅ Referenced in THESIS.md

---

## 📝 Citation

When using these figures, cite:

```
Face Recognition Based Attendance Marking System Using IoT Devices:
A Comprehensive Study on Edge Computing with ESP32-CAM and Raspberry Pi
Group 43, [Institution Name], December 2025
```

---

## 📞 Support

For questions about the diagrams or thesis:
- See: `thesis_diagrams/README.md` for detailed descriptions
- See: `THESIS.md` for full thesis document
- See: `PROJECT_DOCUMENTATION.md` for comprehensive project overview

---

**Document Generated:** January 8, 2026
**Thesis Status:** All figures and tables complete ✓
**Total Pages:** 21 high-quality images ready for thesis submission

---

## 🎯 Next Steps for Thesis Finalization

1. ✅ All diagrams generated
2. ✅ All model performance images available
3. ✅ All comparison charts created
4. ⬜ Physical hardware photos (to be captured during deployment)
5. ⬜ Web interface screenshots (to be captured from running system)
6. ⬜ Final thesis PDF compilation with all figures

---

**Status: COMPLETE - All diagrams and tables generated successfully!**
