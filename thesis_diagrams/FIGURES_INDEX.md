# Thesis Figures and Images Index

This document maps all figures and images mentioned in THESIS.md to their actual file locations.

## Chapter 1: Introduction

| Figure/Image | Description | Location | Status |
|--------------|-------------|----------|--------|
| Figure 1.1 | Traditional vs Automated Attendance System | `thesis_diagrams/attendance_methods_comparison.png` | ✅ Generated |
| Table 1.1 | Comparison of Attendance Marking Methods | `thesis_diagrams/attendance_methods_comparison.png` | ✅ Generated |
| Image | Comparison photo showing traditional vs automated attendance methods | `thesis_diagrams/comparison/` | ⚠️ Physical photo needed |
| Image | Overview of the complete system setup | `thesis_diagrams/hardware/complete_system_deployment.jpg` | ⚠️ Physical photo needed |

## Chapter 3: Methodology

### 3.1 System Architecture

| Figure/Image | Description | Location | Status |
|--------------|-------------|----------|--------|
| Figure 3.1 | System Architecture Diagram | Text-based diagram in THESIS.md | ✅ In document |
| Figure 3.2 | Hardware Integration Flow | Text-based flowchart in THESIS.md | ✅ In document |
| Figure 3.5 | Three-Tier System Architecture | `thesis_diagrams/system_architecture_diagram.png` | ✅ Generated |
| Image | Complete system architecture diagram | `thesis_diagrams/system_architecture_diagram.png` | ✅ Generated |
| Image | Data flow visualization | `thesis_diagrams/system_architecture_diagram.png` | ✅ Generated |
| Image | Physical implementation setup | `thesis_diagrams/hardware/complete_system_deployment.jpg` | ⚠️ Physical photo needed |

### 3.2 Hardware Components

#### ESP32-CAM with LED Light Panel

| Figure/Image | Description | Location | Status |
|--------------|-------------|----------|--------|
| Table 3.1 | ESP32-CAM Technical Specifications | In THESIS.md text | ✅ In document |
| Figure 3.3 | ESP32-CAM with LED Light Panel Setup | Text-based diagram in THESIS.md | ✅ In document |
| Image | ESP32-CAM module with LED light panel ring | `thesis_diagrams/hardware/esp32_cam_led_panel_setup.jpg` | ⚠️ Physical photo needed |
| Image | Close-up of LED panel ring configuration | `thesis_diagrams/hardware/esp32_cam_led_closeup.jpg` | ⚠️ Physical photo needed |
| Image | Wiring diagram for LED panel | `thesis_diagrams/hardware/esp32_led_wiring_diagram.png` | ⚠️ Diagram needed |
| Image | Top view diagram of ESP32-CAM | Text-based in THESIS.md | ✅ In document |
| Image | Side view photo showing illumination | `thesis_diagrams/hardware/esp32_led_side_view.jpg` | ⚠️ Physical photo needed |

#### Raspberry Pi with Cooling Fan

| Figure/Image | Description | Location | Status |
|--------------|-------------|----------|--------|
| Table 3.2 | Raspberry Pi Technical Specifications | In THESIS.md text | ✅ In document |
| Figure 3.4 | Raspberry Pi with Cooling Fan Configuration | Text-based diagram in THESIS.md | ✅ In document |
| Image | Raspberry Pi with cooling fan installed | `thesis_diagrams/hardware/raspberry_pi_with_cooling_fan.jpg` | ⚠️ Physical photo needed |
| Image | Cooling fan mounted showing airflow | `thesis_diagrams/hardware/raspberry_pi_fan_airflow.jpg` | ⚠️ Physical photo needed |
| Image | GPIO pin connection diagram | `thesis_diagrams/hardware/raspberry_pi_fan_wiring_diagram.png` | ⚠️ Diagram needed |
| Image | Side view diagram with fan assembly | Text-based in THESIS.md | ✅ In document |
| Image | Thermal camera comparison | `thesis_diagrams/temperature_performance_graph.png` | ✅ Generated |

### 3.4 Face Recognition Models

| Figure/Image | Description | Location | Status |
|--------------|-------------|----------|--------|
| Table 3.3 | Model Architecture Comparison | In THESIS.md text | ✅ In document |
| Figure 3.6 | Embedding Classifier Architecture | Text-based diagram in THESIS.md | ✅ In document |
| Figure 3.7 | Lightweight CNN Architecture | Text-based diagram in THESIS.md | ✅ In document |
| Figure 3.8 | Custom Embedding Model Architecture | Text-based diagram in THESIS.md | ✅ In document |

### 3.5 Dataset Preparation

| Figure/Image | Description | Location | Status |
|--------------|-------------|----------|--------|
| Table 3.4 | Dataset Statistics | In THESIS.md text | ✅ In document |

## Chapter 4: Results and Discussion

### 4.1 Experimental Setup

| Figure/Image | Description | Location | Status |
|--------------|-------------|----------|--------|
| Figure 4.1 | Web Interface Dashboard Screenshot | Text-based in THESIS.md | ✅ In document |
| Image | Dashboard interface screenshot | `thesis_diagrams/ui_screenshots/web_dashboard_interface.png` | ⚠️ Screenshot needed |
| Figure 4.2 | Add User Page Screenshot | Text-based in THESIS.md | ✅ In document |
| Image | Add user interface screenshot | `thesis_diagrams/ui_screenshots/add_user_interface.png` | ⚠️ Screenshot needed |
| Figure 4.3 | Mark Attendance Page Screenshot | Text-based in THESIS.md | ✅ In document |
| Image | Mark attendance interface screenshot | `thesis_diagrams/ui_screenshots/mark_attendance_interface.png` | ⚠️ Screenshot needed |
| Image | Sample face detection output | `thesis_diagrams/ui_screenshots/sample_face_detection_output.png` | ⚠️ Screenshot needed |

### 4.2 Experimental Results

| Figure/Image | Description | Location | Status |
|--------------|-------------|----------|--------|
| Table 4.1 | Model Performance Metrics | In THESIS.md text | ✅ In document |
| **Figure 4.4** | **Embedding Classifier Confusion Matrix** | `embedding_models/embedding_confusion_matrix.png` | ✅ **Exists** |
| **Figure 4.5** | **Embedding Classifier Precision-Recall Curve** | `embedding_models/embedding_precision_recall_curve.png` | ✅ **Exists** |
| **Figure 4.6** | **Embedding Classifier Confidence Distribution** | `embedding_models/embedding_confidence_curve.png` | ✅ **Exists** |
| **Figure 4.7** | **CNN Model Confusion Matrix** | `cnn_models/cnn_confusion_matrix.png` | ✅ **Exists** |
| Figure 4.8 | Model Performance Comparison Chart | `thesis_diagrams/model_accuracy_comparison.png` | ✅ Generated |
| Figure 4.9 | Real-time Recognition Speed Comparison | `thesis_diagrams/inference_speed_comparison.png` | ✅ Generated |
| Figure 4.10 | Accuracy vs Training Time Trade-off | `thesis_diagrams/accuracy_vs_training_time.png` | ✅ Generated |
| Table 4.2 | Hardware Performance Analysis | In THESIS.md text | ✅ In document |
| Table 4.3 | System Response Time Measurements | In THESIS.md text | ✅ In document |
| Image | Training time comparison chart | `thesis_diagrams/training_time_comparison.png` | ✅ Generated |

### 4.3 Discussion

| Figure/Image | Description | Location | Status |
|--------------|-------------|----------|--------|
| Table 4.4 | Recognition Accuracy Under Different Lighting | In THESIS.md text | ✅ In document |
| Image | Lighting accuracy chart | `thesis_diagrams/lighting_accuracy_chart.png` | ✅ Generated |
| Image | Temperature performance graph | `thesis_diagrams/temperature_performance_graph.png` | ✅ Generated |
| Image | Cost breakdown pie chart | `thesis_diagrams/cost_breakdown_pie.png` | ✅ Generated |
| Image | Annual cost comparison bar chart | `thesis_diagrams/annual_cost_comparison.png` | ✅ Generated |
| Image | ROI timeline graph | `thesis_diagrams/roi_timeline.png` | ✅ Generated |

## Summary Statistics

### ✅ Available Images (Already Exist or Generated)

**Model Performance Images (from training):**
- embedding_models/embedding_confusion_matrix.png
- embedding_models/embedding_precision_recall_curve.png
- embedding_models/embedding_confidence_curve.png
- cnn_models/cnn_confusion_matrix.png
- custom_embedding_models/custom_embedding_confusion_matrix.png
- custom_embedding_models/custom_embedding_precision_recall_curve.png
- custom_embedding_models/custom_embedding_confidence_curve.png

**Generated Diagrams (11 files):**
1. model_accuracy_comparison.png
2. training_time_comparison.png
3. inference_speed_comparison.png
4. accuracy_vs_training_time.png
5. temperature_performance_graph.png
6. cost_breakdown_pie.png
7. annual_cost_comparison.png
8. roi_timeline.png
9. lighting_accuracy_chart.png
10. system_architecture_diagram.png
11. attendance_methods_comparison.png

**Text-based Diagrams (in THESIS.md):**
- System architecture diagrams (ASCII art)
- Hardware flow diagrams
- Model architecture diagrams
- UI mockups (text-based)
- Data tables and specifications

**Total Available:** 18 images + numerous text-based diagrams

### ⚠️ Images Requiring Physical Photos/Screenshots (Optional for Completeness)

**Hardware Photos (7 items):**
1. ESP32-CAM with LED panel (multiple angles)
2. Raspberry Pi with cooling fan
3. Complete system deployment
4. LED wiring diagram
5. Fan wiring diagram
6. Close-up hardware details
7. Thermal camera comparison (optional)

**UI Screenshots (4 items):**
1. Dashboard interface
2. Add user page
3. Mark attendance page
4. Face detection sample output

**Comparison Photos (2 items):**
1. Traditional vs automated methods photo
2. Before/after hardware comparison

**Total Needed:** 13 optional enhancement photos

## Usage Instructions

### For Figures That Exist

```markdown
![Model Accuracy Comparison](thesis_diagrams/model_accuracy_comparison.png)

**Figure 4.8:** Model Performance Comparison Chart showing validation accuracy 
across three architectures: Embedding Classifier (99.74%), Custom Embedding 
(98.86%), and Lightweight CNN (64.04%).
```

### For Model Performance Images

```markdown
![Confusion Matrix](../embedding_models/embedding_confusion_matrix.png)

**Figure 4.4:** Confusion matrix visualization showing the Embedding Classifier's 
performance across 67 users with 99.74% overall accuracy.
```

### For Physical Photos (When Available)

```markdown
![ESP32-CAM Setup](thesis_diagrams/hardware/esp32_cam_led_panel_setup.jpg)

**Figure 3.3:** ESP32-CAM module with integrated LED light panel providing 
ring-light illumination for consistent image quality.
```

## Notes

1. **Priority:** Focus on generated diagrams and existing model images - these provide the core technical content.

2. **Text-Based Diagrams:** Many complex diagrams are already present in THESIS.md as ASCII art, which is acceptable for technical documentation.

3. **Optional Enhancements:** Physical photos and UI screenshots would enhance the thesis but are not critical for conveying the technical contributions.

4. **High Quality:** All generated images are at 300 DPI, suitable for print publication.

5. **Consistency:** Generated images use consistent styling, colors, and fonts for professional presentation.

## Regenerating Diagrams

To regenerate all diagrams:

```bash
python3 scripts/generate_thesis_diagrams.py
```

This will update all generated diagrams in `thesis_diagrams/` directory.

---

**Document Version:** 1.0  
**Last Updated:** January 2026  
**Status:** Core diagrams complete, optional photos can be added as needed
