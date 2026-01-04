# Thesis Diagrams and Visualizations

This directory contains all the diagrams, charts, and visualizations referenced in the thesis document (THESIS.md).

## Generated Diagrams

### Performance Comparison Charts

1. **model_accuracy_comparison.png**
   - Figure 4.8: Model Performance Comparison Chart
   - Compares validation accuracy of three models: Embedding Classifier (99.74%), Custom Embedding (98.86%), and Lightweight CNN (64.04%)

2. **training_time_comparison.png**
   - Training time comparison across models
   - Shows Embedding Classifier (30 sec), Custom Embedding (2-3 min), Lightweight CNN (32 min)

3. **inference_speed_comparison.png**
   - Figure 4.9: Real-time Recognition Speed Comparison
   - Compares inference times: Embedding Classifier (80-100ms), Custom Embedding (90-110ms), CNN (120-150ms)

4. **accuracy_vs_training_time.png**
   - Figure 4.10: Accuracy vs Training Time Trade-off
   - Scatter plot showing the optimal balance achieved by Embedding Classifier

### Hardware Performance

5. **temperature_performance_graph.png**
   - Raspberry Pi thermal performance with and without cooling fan
   - Shows CPU temperature and recognition speed over 30 minutes of continuous operation
   - Demonstrates 40°C temperature reduction and 50-60% performance improvement with fan

6. **lighting_accuracy_chart.png**
   - Table 4.4: Recognition Accuracy Under Different Lighting Conditions
   - Compares accuracy with and without LED light panel across 6 lighting scenarios
   - Shows 18% improvement in low-light conditions

### Cost Analysis

7. **cost_breakdown_pie.png**
   - Hardware cost breakdown pie chart
   - Total system cost: Rs. 56,700 ($189)
   - Shows distribution across components: Raspberry Pi, ESP32-CAM, WiFi Router, etc.

8. **annual_cost_comparison.png**
   - Annual cost comparison with commercial alternatives
   - Compares Our System (Rs. 56,700) vs Cloud SaaS (Rs. 420,000), Commercial IP (Rs. 858,000), etc.

9. **roi_timeline.png**
   - Return on Investment (ROI) timeline graph
   - Shows break-even point at 23 days
   - Demonstrates Rs. 568,300 savings in first year vs manual roll call

### System Architecture

10. **system_architecture_diagram.png**
    - Three-tier system architecture visualization
    - Shows Cloud Layer (GitHub Actions), Edge Layer (Raspberry Pi), and IoT Layer (ESP32-CAM)

11. **attendance_methods_comparison.png**
    - Table 1.1: Comparison of Attendance Marking Methods
    - Compares Manual Roll Call, Paper Registers, RFID, Fingerprint, Face Recognition, and Our System
    - Metrics: Time required, Proxy prevention, Contact requirement, Cost, Data management

## Already Existing Model Performance Images

These images are generated during model training and stored in their respective directories:

### Embedding Classifier (embedding_models/)
- **embedding_confusion_matrix.png** - Figure 4.4
- **embedding_precision_recall_curve.png** - Figure 4.5
- **embedding_confidence_curve.png** - Figure 4.6

### CNN Model (cnn_models/)
- **cnn_confusion_matrix.png** - Figure 4.7

### Custom Embedding Model (custom_embedding_models/)
- **custom_embedding_confusion_matrix.png**
- **custom_embedding_precision_recall_curve.png**
- **custom_embedding_confidence_curve.png**

## Image Placeholders (Physical Photos)

The following items mentioned in the thesis require actual physical photographs:

### Hardware Photos
- ESP32-CAM with LED light panel setup (multiple angles)
- Raspberry Pi with cooling fan installed
- Complete system setup showing integration
- Wiring diagrams for LED panel and cooling fan
- Close-up of LED ring configuration

### UI Screenshots
- Web interface dashboard
- Add user page
- Mark attendance page with live camera feed
- Sample face detection output

### Comparison Photos
- Traditional vs automated attendance methods
- Thermal camera showing temperature reduction

## Generating the Diagrams

To regenerate all diagrams, run:

```bash
python3 scripts/generate_thesis_diagrams.py
```

This will create/update all diagram files in this directory.

## Usage in Thesis

All generated diagrams are referenced in THESIS.md with proper figure numbers and captions. The images can be embedded in the final thesis document using standard markdown or LaTeX image inclusion syntax.

### Example References in Thesis:

```markdown
![Model Accuracy Comparison](thesis_diagrams/model_accuracy_comparison.png)
**Figure 4.8:** Model Performance Comparison Chart

![Temperature Performance](thesis_diagrams/temperature_performance_graph.png)
**Figure:** Raspberry Pi thermal performance showing the impact of active cooling
```

## Resolution and Quality

All diagrams are generated at 300 DPI (print quality) with dimensions optimized for inclusion in academic documents. The images use a consistent color scheme and styling for professional presentation.

## File Formats

- All diagrams are saved as PNG files for high quality and transparency support
- Resolution: 300 DPI (suitable for printing)
- Color space: RGB
- Average file size: 50-500 KB per image

## Citation

When using these diagrams, please cite:

```
Face Recognition Based Attendance Marking System Using IoT Devices
A Comprehensive Study on Edge Computing with ESP32-CAM and Raspberry Pi
[Author Name], [Institution], December 2025
```

## License

These diagrams are part of the thesis project and are provided for educational and research purposes.
