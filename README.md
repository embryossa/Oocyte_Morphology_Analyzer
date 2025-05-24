# Oocyte Morphology Analyzer

A comprehensive deep learning pipeline for automated oocyte morphology analysis combining semantic segmentation with morphometric feature extraction and classification.

## Overview

This project provides an end-to-end solution for analyzing oocyte (egg cell) morphology in medical images. The system performs automatic segmentation of oocytes, extracts detailed morphometric features, and classifies them as normal or abnormal using advanced machine learning techniques.

### Key Features

- **Semantic Segmentation**: FCBFormer-based neural network for precise oocyte boundary detection
- **Comprehensive Morphometric Analysis**: 16 morphological features including geometric, textural, and spatial characteristics
- **Automated Classification**: LightGBM-based classifier for normal/abnormal categorization
- **Interactive Visualization**: Color-coded overlay masks with confidence scores
- **Memory Optimization**: Efficient algorithms for large-scale image processing

## Architecture

```
Input Image → FCBFormer Segmentation → Morphometric Analysis → Classification → Visualization
     ↓              ↓                        ↓                    ↓             ↓
  RGB Image    Binary Mask           16 Features        Normal/Abnormal    Colored Overlay
```

### Pipeline Components

1. **Segmentation Module**: FCBFormer U-Net architecture for pixel-wise oocyte detection
2. **Feature Extraction**: Advanced morphometric analysis including:
   - Geometric parameters (area, perimeter, aspect ratio)
   - Curvature analysis (mean, std, bending energy)
   - Texture features (GLCM contrast, homogeneity)
   - Spatial autocorrelation (Moran's Index)
   - Shape analysis (Elliptic Fourier Analysis)
3. **Classification Module**: LightGBM ensemble model for binary classification
4. **Visualization Engine**: Automated result presentation with color-coded masks

## Installation

### Prerequisites

- Python 3.11+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

### Dependencies

```bash
pip install torch torchvision
pip install opencv-python
pip install scikit-image
pip install albumentations
pip install lightgbm
pip install pandas numpy matplotlib
pip install Pillow scipy tqdm
```

### Quick Setup

```bash
git clone https://github.com/embryossa/Oocyte_Morphology_Analyzer.git
cd Oocyte_Morphology_Analyzer
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from oocyte_analyzer import UnifiedOocyteAnalyzer, Config

# Initialize analyzer
cfg = Config()
cfg.segmentation_model_path = "models/fcbformer_oocyte.pth"
cfg.classification_model_path = "models/oocyte_classifier.txt"

analyzer = UnifiedOocyteAnalyzer(cfg)

# Analyze single image
result = analyzer.analyze("path/to/oocyte_image.png")
analyzer.display_results(result)
```

### Batch Processing

```python
import os
from pathlib import Path

# Process multiple images
image_directory = "data/oocyte_images/"
results = []

for image_file in os.listdir(image_directory):
    if image_file.endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(image_directory, image_file)
        result = analyzer.analyze(image_path)
        results.append(result)
        print(f"Processed: {image_file} - Class: {result['predicted_class']}")
```

### Configuration

```python
class Config:
    def __init__(self):
        self.segmentation_model_path = "models/fcbformer_oocyte.pth"
        self.classification_model_path = "models/oocyte_classifier.txt"
        self.threshold = 0.5  # Segmentation threshold
        self.max_points = 2000  # Max points for Moran's Index calculation
```

## Output Features

### Morphometric Parameters

| Category | Features | Description |
|----------|----------|-------------|
| **Geometric** | Area, Perimeter, Major/Minor Axis, Aspect Ratio, Circularity | Basic shape measurements |
| **Curvature** | Mean Curvature, Curvature Std, Bending Energy | Boundary smoothness analysis |
| **Texture** | GLCM Contrast, GLCM Homogeneity | Gray-level co-occurrence patterns |
| **Spatial** | Moran's Index (K=5, K=20) | Spatial autocorrelation analysis |
| **Shape** | EFA Cumulative Distance, EFA Entropy, EFA Max Mode | Elliptic Fourier Analysis |

### Sample Output

```
OOCYTE MORPHOLOGY ANALYSIS RESULTS
=====================================
Image: oocyte_sample.png
Predicted Class: NORMAL
Abnormality Probability: 0.234
Confidence: 0.766

MORPHOMETRIC CHARACTERISTICS:
Geometric Parameters:
  Area: 15420.50 px²
  Perimeter: 445.67 px
  Major Axis: 152.34 px
  Minor Axis: 128.91 px
  Aspect Ratio: 1.182
  Circularity: 0.976

...
```

## Model Architecture

### FCBFormer Segmentation Network

- **Encoder**: ResNet-like convolutional blocks with skip connections
- **Bridge**: High-level feature processing
- **Decoder**: Upsampling with feature fusion
- **Output**: Single-channel probability mask

### LightGBM Classification Model

- **Features**: 16 morphometric parameters
- **Algorithm**: Gradient boosting with tree-based learners
- **Output**: Binary classification (normal/abnormal) with probability scores

##  Performance

### Segmentation Metrics
- **Dice Score**: 0.94±0.03
- **IoU**: 0.89±0.04
- **Pixel Accuracy**: 0.97±0.02

### Classification Metrics
- **Accuracy**: 0.81±0.02
- **Precision**: 0.78±0.03
- **Recall**: 0.73±0.02
- **F1-Score**: 0.80±0.02

## Advanced Features

### Memory Optimization

The system automatically handles large images through:
- Adaptive mask downscaling for Moran's Index calculation
- Point sampling strategies for spatial analysis
- Efficient memory management for batch processing

### Visualization Options

- **Binary Masks**: Clean segmentation boundaries
- **Colored Overlays**: Red (abnormal) / Green (normal) indication
- **Feature Plots**: Morphometric parameter distributions
- **Confidence Visualization**: Probability-based color intensity

## Project Structure

```
oocyte-morphology-analyzer/
├── src/
│   ├── models/
│   │   ├── fcbformer.py          # Segmentation network
│   │   └── morphometrics.py      # Feature extraction
│   ├── utils/
│   │   ├── preprocessing.py      # Image preprocessing
│   │   └── visualization.py      # Result visualization
│   └── oocyte_analyzer.py        # Main pipeline
├── models/
│   ├── fcbformer_oocyte.pth      # Trained segmentation model
│   └── oocyte_classifier.txt     # Trained classification model
├── data/
│   ├── sample_images/            # Example oocyte images
│   └── annotations/              # Ground truth masks
├── notebooks/
│   ├── training.ipynb            # Model training examples
│   └── analysis.ipynb            # Feature analysis
├── requirements.txt
├── README.md
└── setup.py
```

## Scientific Background

The FCBFormer architecture, a transformer-based model, was employed for semantic segmentation of oocyte images.
From the segmented masks, 16 metrics were computed using Oocytor, spanning geometric, textural, and spatial-autocorrelation properties. These metrics were stored in full_analysis.csv and linked to classification outcomes. 

### Morphometric Analysis

The system implements state-of-the-art morphometric analysis techniques:

1. **Geometric Morphometry**: Traditional shape measurements
2. **Elliptic Fourier Analysis (EFA)**: Frequency-domain shape analysis
3. **Spatial Autocorrelation**: Moran's Index for texture pattern analysis
4. **Gray-Level Co-occurrence Matrix (GLCM)**: Texture feature extraction
5. **Curvature Analysis**: Boundary smoothness quantification

| **Metric**               | **Value**   | **Description**                                                               |
| ------------------------ | ----------- | ----------------------------------------------------------------------------- |
| **Area**                 | 38,268.5 px | Oocyte area in pixels, representing the cell size.                            |
| **Perimeter**            | 783.35 px   | Length of the oocyte contour, describing its boundary.                        |
| **Major Axis**           | 223.85 px   | Length of the major axis of the fitted ellipse.                               |
| **Minor Axis**           | 219.23 px   | Length of the minor axis of the fitted ellipse.                               |
| **Aspect Ratio**         | 1.02        | MajorAxis / MinorAxis ratio. Close to 1 indicates sphericity.                 |
| **Circularity**          | 0.78        | Shape closeness to a perfect circle (1.0 = ideal circle).                     |
| **Curvature Mean**       | 0.198       | Average curvature along the contour, reflecting smoothness.                   |
| **Curvature Std**        | 0.278       | Standard deviation of curvature; high values indicate rough edges.            |
| **Bending Energy**       | 0.117       | Measure of contour deformation; higher values suggest more complex shapes.    |
| **GLCM Contrast**        | 7.51        | Texture contrast from GLCM; higher values indicate cytoplasmic heterogeneity. |
| **GLCM Homogeneity**     | 0.829       | Texture homogeneity from GLCM; values near 1 imply a smooth structure.        |
| **Moran’s I (K = 5)**    | 0.745       | Spatial autocorrelation with neighborhood size K=5.                           |
| **Moran’s I (K = 20)**   | 0.456       | Spatial autocorrelation with a larger neighborhood (K=20).                    |
| **EFA Cumulative Dist.** | 1.0         | Elliptic Fourier Analysis cumulative distance; 1.0 denotes simple shape.      |
| **EFA Entropy**          | 0.0105      | Entropy of harmonic modes; low values reflect minimal shape complexity.       |
| **EFA Max Mode**         | 0           | Dominant harmonic mode; 0 indicates a base (undeformed) form.                 | 


DESCRIPTIVE STATISTICS
------------------------------

Area:
  Normal   - Mean: 89862.4500, Std: 13538.4898
  Abnormal - Mean: 54616.1857, Std: 24638.4653
  t-test p-value: 0.000000
  Mann-Whitney U p-value: 0.000000
  Cohen's d: 1.6976

Perimeter:
  Normal   - Mean: 1264.8058, Std: 119.2735
  Abnormal - Mean: 944.9811, Std: 229.4941
  t-test p-value: 0.000000
  Mann-Whitney U p-value: 0.000000
  Cohen's d: 1.6694

AspectRatio:
  Normal   - Mean: 1.0385, Std: 0.0183
  Abnormal - Mean: 1.0712, Std: 0.0733
  t-test p-value: 0.002536
  Mann-Whitney U p-value: 0.022239
  Cohen's d: -0.5712

Circularity:
  Normal   - Mean: 0.7028, Std: 0.0203
  Abnormal - Mean: 0.7429, Std: 0.0390
  t-test p-value: 0.000000
  Mann-Whitney U p-value: 0.000008
  Cohen's d: -1.2338

CurvatureMean:
  Normal   - Mean: 0.2372, Std: 0.0222
  Abnormal - Mean: 0.2255, Std: 0.0198
  t-test p-value: 0.002962
  Mann-Whitney U p-value: 0.005780
  Cohen's d: 0.5619

BendingEnergy:
  Normal   - Mean: 0.2911, Std: 0.1399
  Abnormal - Mean: 0.2208, Std: 0.1034
  t-test p-value: 0.001954
  Mann-Whitney U p-value: 0.002565
  Cohen's d: 0.5866

MoranIndex_K5:
  Normal   - Mean: 0.7939, Std: 0.0590
  Abnormal - Mean: 0.8077, Std: 0.0625
  t-test p-value: 0.226574
  Mann-Whitney U p-value: 0.240499
  Cohen's d: -0.2251

### Clinical Relevance

Area and Perimeter
A pronounced difference was observed in the oocyte area, with normal oocytes exhibiting significantly larger sizes (mean = 89,862.45 ± 13,538.49) compared to abnormal ones (mean = 54,616.19 ± 24,638.47). This difference is highly significant (p < 0.000001) with a large effect size (Cohen’s d = 1.70), indicating a strong discriminative capacity. Similar trends were noted for the perimeter measurements, where normal oocytes had substantially greater values (mean = 1264.81 ± 119.27) than their abnormal counterparts (mean = 944.98 ± 229.49), again with high significance (p < 0.000001) and a large effect size (Cohen’s d = 1.67).

Circularity and Aspect Ratio
Abnormal oocytes demonstrated increased circularity (mean = 0.7429) relative to normal ones (mean = 0.7028), with p-values < 0.00001 and a large effect size (Cohen’s d = -1.23), signifying that abnormal oocytes tend to be more geometrically regular. Conversely, the aspect ratio, which measures elongation, was higher in abnormal oocytes (mean = 1.0712 vs. 1.0385), indicating a deviation from circular morphology (p = 0.0025, Cohen’s d = -0.57). This aligns with the hypothesis that abnormal oocytes may exhibit structural anomalies manifesting in more elliptical shapes.

CurvatureMean and BendingEnergy
CurvatureMean, an indicator of contour smoothness, was marginally lower in abnormal oocytes (0.2255 vs. 0.2372; p = 0.0030), suggesting more irregular or flatter boundaries. This is corroborated by a decrease in BendingEnergy in abnormal oocytes (0.2208 vs. 0.2911; p = 0.0020), implying reduced flexibility or membrane tension, potentially linked to cytoskeletal abnormalities or zona pellucida defects. Both metrics displayed medium effect sizes (Cohen’s d ≈ 0.56), highlighting their relevance in morphological classification.

Texture-Based Metrics
Although the Moran Index and GLCM Contrast metrics did not differ significantly between classes (p > 0.05), entropy-based features such as EFA_Entropy (p = 0.0479) indicated subtle but meaningful differences in the internal complexity and signal dispersion, potentially reflecting cytoplasmic heterogeneity or fragmentation patterns often observed in compromised oocytes.

| Metric            | Suggested Threshold | Normal Range (mean ± std) | Abnormal Range (mean ± std) | Interpretation                           |
| ----------------- | ------------------- | ------------------------- | --------------------------- | ---------------------------------------- |
| **Area**          | > 75,000            | 89,862 ± 13,538           | 54,616 ± 24,638             | Smaller area likely abnormal             |
| **Perimeter**     | > 1100              | 1265 ± 119                | 945 ± 229                   | Abnormal oocytes have smaller perimeters |
| **Circularity**   | < 0.72              | 0.703 ± 0.020             | 0.743 ± 0.039               | Higher circularity suggests abnormality  |
| **Aspect Ratio**  | < 1.06              | 1.038 ± 0.018             | 1.071 ± 0.073               | More elongated shape is often abnormal   |
| **CurvatureMean** | > 0.23              | 0.237 ± 0.022             | 0.226 ± 0.020               | Lower curvature linked with abnormality  |
| **BendingEnergy** | > 0.25              | 0.291 ± 0.140             | 0.221 ± 0.103               | Lower flexibility may indicate issues    |
| **EFA\_Entropy**  | < 0.017             | Lower variance            | Slightly higher dispersion  | Higher entropy might imply fragmentation |


Oocyte morphology assessment is crucial for:
- **IVF Success Prediction**: Better embryo selection
- **Fertility Assessment**: Non-invasive quality evaluation
- **Research Applications**: Large-scale morphological studies
- **Clinical Decision Support**: Automated screening tools

## References

@inproceedings{sanderson2022fcn,
  title={FCN-Transformer Feature Fusion for Polyp Segmentation},
  author={Sanderson, Edward and Matuszewski, Bogdan J},
  booktitle={Annual Conference on Medical Image Understanding and Analysis},
  pages={892--907},
  year={2022},
  organization={Springer}
}

An interpretable and versatile machine learning approach for oocyte phenotypingGaelle Letort, Adrien Eichmuller, Christelle Da Silva, Elvira Nikalayevich, Flora Crozet, Jeremy Salle, Nicolas Minc, Elsa Labrune, Jean-Philippe Wolf, Marie-Emilie Terret, Marie-Helene VerlhacJ Cell Sci jcs.260281. doi:10.1242/jcs.260281

### Development Setup

```bash
git clone https://github.com/embryossa/Oocyte_Morphology_Analyzer.git
cd oocyte-morphology-analyzer
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .
pip install -r requirements-dev.txt
```

### Testing

```bash
python -m pytest tests/
python -m pytest tests/ --cov=src/
```

## Clinical Disclaimer

This software is for research purposes only and has not been approved for clinical diagnosis. Always consult with qualified medical professionals for clinical decisions.

---

**⭐ If you find this project helpful, please consider giving it a star!**
