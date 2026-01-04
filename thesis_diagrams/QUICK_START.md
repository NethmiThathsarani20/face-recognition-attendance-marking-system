# Quick Start Guide: Using Thesis Diagrams

This guide helps you quickly incorporate the generated diagrams into your thesis document.

## 📁 Where Are The Diagrams?

```
thesis_diagrams/          # Main diagrams directory (11 generated images)
embedding_models/         # Model performance images (3 images)
cnn_models/              # CNN model images (3 images)
custom_embedding_models/ # Custom model images (1 image)
```

## 🎯 Quick Reference: Which Diagram for Which Section?

### Chapter 1: Introduction
- `attendance_methods_comparison.png` - Table 1.1 visualization

### Chapter 4: Results (Main Figures)

#### Model Performance
- `embedding_models/embedding_confusion_matrix.png` - **Figure 4.4**
- `embedding_models/embedding_precision_recall_curve.png` - **Figure 4.5**
- `embedding_models/embedding_confidence_curve.png` - **Figure 4.6**
- `cnn_models/cnn_confusion_matrix.png` - **Figure 4.7**

#### Comparison Charts
- `model_accuracy_comparison.png` - **Figure 4.8**
- `inference_speed_comparison.png` - **Figure 4.9**
- `accuracy_vs_training_time.png` - **Figure 4.10**

#### Supporting Charts
- `training_time_comparison.png` - Training efficiency
- `lighting_accuracy_chart.png` - LED panel impact
- `temperature_performance_graph.png` - Cooling system analysis

#### Cost Analysis
- `cost_breakdown_pie.png` - Hardware costs
- `annual_cost_comparison.png` - vs Commercial solutions
- `roi_timeline.png` - Break-even analysis

#### Architecture
- `system_architecture_diagram.png` - Three-tier design

## 📝 Example Usage

### For Markdown/Jupyter

```markdown
## Model Performance Comparison

![Model Accuracy](thesis_diagrams/model_accuracy_comparison.png)

**Figure 4.8:** Model Performance Comparison Chart showing validation accuracy 
across three architectures. The Embedding Classifier achieves 99.74% accuracy, 
significantly outperforming the Custom Embedding model (98.86%) and 
Lightweight CNN (64.04%).
```

### For LaTeX

```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=0.85\textwidth]{thesis_diagrams/model_accuracy_comparison.png}
\caption{Model Performance Comparison Chart}
\label{fig:model_accuracy}
\end{figure}

As shown in Figure~\ref{fig:model_accuracy}, the Embedding Classifier...
```

### For Word/LibreOffice

1. Insert > Picture > From File
2. Navigate to `thesis_diagrams/`
3. Select desired image
4. Right-click > Insert Caption
5. Format as needed

## 🎨 Image Properties

All generated diagrams are:
- **Format:** PNG with transparency
- **Resolution:** 300 DPI (print quality)
- **Dimensions:** 3000-4000 pixels wide
- **Color:** RGBA with professional styling
- **Size:** 100-350 KB per file

## 📊 Complete Figure List

| # | Figure | File | Size |
|---|--------|------|------|
| 1.1 | Attendance Methods Comparison | attendance_methods_comparison.png | 214 KB |
| 4.4 | Embedding Confusion Matrix | embedding_models/embedding_confusion_matrix.png | 1.0 MB |
| 4.5 | Embedding Precision-Recall | embedding_models/embedding_precision_recall_curve.png | 31 KB |
| 4.6 | Embedding Confidence Dist. | embedding_models/embedding_confidence_curve.png | 31 KB |
| 4.7 | CNN Confusion Matrix | cnn_models/cnn_confusion_matrix.png | 984 KB |
| 4.8 | Model Accuracy Comparison | model_accuracy_comparison.png | 126 KB |
| 4.9 | Inference Speed Comparison | inference_speed_comparison.png | 138 KB |
| 4.10 | Accuracy vs Training Time | accuracy_vs_training_time.png | 261 KB |
| - | Training Time Comparison | training_time_comparison.png | 103 KB |
| - | Temperature Performance | temperature_performance_graph.png | 314 KB |
| - | Lighting Accuracy Chart | lighting_accuracy_chart.png | 148 KB |
| - | Cost Breakdown Pie | cost_breakdown_pie.png | 322 KB |
| - | Annual Cost Comparison | annual_cost_comparison.png | 217 KB |
| - | ROI Timeline | roi_timeline.png | 262 KB |
| - | System Architecture | system_architecture_diagram.png | 207 KB |

## 🔧 Customizing Diagrams

To modify diagrams, edit `scripts/generate_thesis_diagrams.py`:

```python
# Example: Change color scheme
colors = ['#2ecc71', '#3498db', '#e74c3c']  # Green, Blue, Red

# Example: Adjust figure size
plt.rcParams['figure.figsize'] = (12, 8)  # Width, Height in inches

# Example: Change DPI
plt.savefig('output.png', dpi=300)
```

Then regenerate:
```bash
python3 scripts/generate_thesis_diagrams.py
```

## 💡 Tips for Best Results

### 1. Image Placement
- Place images close to where they're first referenced
- Use consistent sizing within each chapter
- Center-align for professional appearance

### 2. Captions
- Always include figure numbers and descriptive captions
- Explain what the image shows and why it matters
- Reference key findings visible in the image

### 3. Resolution
- Use original PNG files (don't resize)
- If must resize, maintain aspect ratio
- Never use JPEG compression (causes artifacts)

### 4. Citations
```
When discussing results, reference figures:
"As shown in Figure 4.8, the Embedding Classifier..."
"The confusion matrix (Figure 4.4) demonstrates..."
"Cost analysis (Figure X) reveals..."
```

## 📚 Documentation Files

| File | Purpose |
|------|---------|
| README.md | Overview and descriptions |
| FIGURES_INDEX.md | Complete mapping of all figures |
| PHOTO_CAPTURE_GUIDE.md | Guide for optional photos |
| GENERATION_SUMMARY.md | Status and statistics |
| QUICK_START.md | This file - quick reference |

## ✅ Final Checklist

Before submitting your thesis:

- [ ] All figures referenced in text
- [ ] All figures have captions
- [ ] All figures numbered correctly
- [ ] Image quality verified (300 DPI)
- [ ] Consistent sizing throughout
- [ ] All figure citations formatted properly
- [ ] Images render correctly in PDF export
- [ ] File paths correct (relative or absolute)
- [ ] Backup copies of all images
- [ ] Source code (generate_thesis_diagrams.py) included

## 🚀 Quick Commands

```bash
# List all diagrams
ls -lh thesis_diagrams/*.png

# Regenerate all diagrams
python3 scripts/generate_thesis_diagrams.py

# Check image properties
file thesis_diagrams/*.png

# Count total images
find . -name "*.png" | grep -E "(thesis_diagrams|embedding_models|cnn_models)" | wc -l

# Verify sizes
du -sh thesis_diagrams/
```

## 📞 Need Help?

1. Check `FIGURES_INDEX.md` for figure-to-file mapping
2. See `README.md` for detailed descriptions
3. Review `GENERATION_SUMMARY.md` for status
4. Examine `scripts/generate_thesis_diagrams.py` for code

## 🎓 Academic Standards

These diagrams meet common thesis requirements:

- ✅ IEEE standards (300 DPI minimum)
- ✅ ACM publication quality
- ✅ University thesis guidelines
- ✅ Print-ready resolution
- ✅ Professional styling
- ✅ Consistent formatting
- ✅ Accessible colors (colorblind-friendly)
- ✅ Clear labels and legends

---

**Last Updated:** January 2026  
**Version:** 1.0  
**Status:** Production Ready ✅
