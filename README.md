<div align="center">

  # AERIS 🛰️
  **Advanced Earth Remote-sensing Intelligence System for Cloud Detection**
  
  <p align="center">
    <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
    <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
    <img src="https://img.shields.io/badge/U--Net-Segmentation-00599C?style=for-the-badge"/>
    <img src="https://img.shields.io/badge/ResNet34-ImageNet-76B900?style=for-the-badge"/>
    <img src="https://img.shields.io/badge/HuggingFace-FFD21F?style=for-the-badge&logo=huggingface&logoColor=black"/>
  </p>
  
  <p align="center">
    <a href="https://aeris-cloud-detection.streamlit.app/">
      <img src="https://img.shields.io/badge/Live_Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
    </a>
    <a href="https://huggingface.co/Eklavya16/aeris-cloud-detection">
      <img src="https://img.shields.io/badge/Model_Weights-HuggingFace-FFD21F?style=for-the-badge&logo=huggingface&logoColor=black"/>
    </a>
    <a href="https://www.kaggle.com/datasets/sorour/38cloud-cloud-segmentation-in-satellite-images">
      <img src="https://img.shields.io/badge/Dataset-Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white"/>
    </a>
  </p>

</div>

---

## 🎯 Overview

AERIS is a deep learning-powered **satellite cloud detection and segmentation system** designed for Landsat 8 multispectral imagery. The system combines **U-Net architecture** with **Monte Carlo Dropout uncertainty quantification** to provide pixel-level cloud probability maps with confidence estimation for remote sensing applications.

### Why Cloud Detection Matters

Cloud masking is critical for:
- 🌍 **Earth Observation** – Removing atmospheric interference from satellite imagery
- 🌾 **Agriculture** – Accurate vegetation index calculation (NDVI, EVI)
- 🌊 **Environmental Monitoring** – Water quality assessment and land cover analysis
- 🏙️ **Urban Planning** – Change detection and infrastructure mapping
- 🔥 **Disaster Response** – Flood mapping and fire monitoring

### Key Features

✅ **Multi-Band Processing** – 4-channel input (Red, Green, Blue, Near-Infrared)  
✅ **Pixel-Level Segmentation** – Binary cloud/clear classification with probability maps  
✅ **Uncertainty Quantification** – Monte Carlo Dropout (30 iterations) for confidence estimation  
✅ **Production-Ready** – Streamlit deployment with HuggingFace model hosting  
✅ **High Performance** – 92.2% IoU, 94.3% Dice, 95.8% Accuracy, 0.70% ECE  
✅ **Calibrated Predictions** – Expected Calibration Error < 1%  

---

## 🏗️ Architecture

### Model Design

```
INPUT (256×256×4) → U-Net Encoder (ResNet34) → Bottleneck → U-Net Decoder → OUTPUT (256×256×1)
                         ↓                                        ↑
                   [Skip Connections]  ────────────────────────────┘
```

| Component | Specification |
|-----------|--------------|
| **Backbone** | ResNet34 (ImageNet pretrained) |
| **Architecture** | U-Net with encoder-decoder structure |
| **Encoder Weights** | ImageNet transfer learning |
| **Input Channels** | 4 (Red, Green, Blue, NIR) |
| **Output Channels** | 1 (Cloud probability: 0-1) |
| **Input Resolution** | 256×256 pixels |
| **Activation** | Sigmoid (output layer) |
| **Uncertainty** | MC Dropout (30 iterations, 20% dropout) |

### Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| **Optimizer** | AdamW (lr=1e-4, weight_decay=1e-4) |
| **Loss Function** | Combined (Dice + BCE, 50/50 weight) |
| **Batch Size** | 32 |
| **Epochs** | 30 |
| **Scheduler** | ReduceLROnPlateau (patience=5, factor=0.5) |
| **Early Stopping** | Based on validation IoU |
| **Dropout Rate** | 0.2 (for MC Dropout inference) |

### Uncertainty Quantification

**Monte Carlo Dropout:**
- 30 forward passes with dropout enabled at inference
- Prediction: Mean of 30 samples
- Uncertainty: Standard deviation of 30 samples

**Benefits:**
- Identifies ambiguous cloud edges and thin cirrus
- Flags low-confidence predictions for manual review
- Enables quality-based filtering (Excellent/Good/Fair/Poor zones)

---

## 📊 Dataset

### 38-Cloud Dataset (Landsat 8)

| Dataset Split | Patches | Images | Resolution |
|--------------|---------|--------|-----------|
| **Training** | 8,400 | 21 scenes | 256×256 px |
| **Validation** | 9,201 | 23 scenes | 256×256 px |
| **Total** | 17,601 | 44 scenes | 256×256 px |

**Data Source:** Landsat 8 Operational Land Imager (OLI)  
**Spectral Bands Used:**  
- Band 4: Red (630-680 nm)
- Band 3: Green (525-600 nm)
- Band 2: Blue (450-515 nm)
- Band 5: Near-Infrared (845-885 nm)

**Ground Truth:** Manually labeled binary cloud masks by remote sensing experts

**Preprocessing:**
- Normalization: ImageNet mean/std for RGB + custom for NIR
- No data augmentation (preserves spectral signatures)
- Patch extraction from full Landsat scenes
- 80/20 train-validation split

**Dataset Citation:**  
Mohajerani, S., & Saeedi, P. (2019). *Cloud-Net: An End-to-End Cloud Detection Algorithm for Landsat 8 Imagery.* IEEE International Geoscience and Remote Sensing Symposium (IGARSS).

---

## 📈 Performance Metrics

### Validation Performance (9,201 patches)

| Metric | Value | Description |
|--------|-------|-------------|
| **IoU (Jaccard)** | **92.2%** | Intersection over Union |
| **Dice Score** | **94.3%** | F1 for segmentation |
| **Accuracy** | **95.8%** | Pixel-level accuracy |
| **Precision** | 93.5% | Cloud pixels correctly identified |
| **Recall** | 95.1% | Fraction of clouds detected |
| **ECE** | **0.0070%** | Expected Calibration Error |

### Calibration Analysis

The model achieves **0.70% Expected Calibration Error (ECE)**, indicating highly reliable confidence estimates:
- Predicted probabilities align closely with actual outcomes
- Critical for uncertainty-based filtering and quality assessment
- Enables trustworthy deployment in operational workflows

### Quality Tiers

| Quality Zone | Uncertainty Range | Coverage (validation) |
|-------------|-------------------|---------------------|
| **Excellent** | < 0.05 | 67.2% of pixels |
| **Good** | 0.05 - 0.10 | 21.8% of pixels |
| **Fair** | 0.10 - 0.20 | 8.3% of pixels |
| **Poor** | > 0.20 | 2.7% of pixels |

---

## 🚀 Deployment

### Live Application

🌐 **Streamlit App:** [https://aeris-cloud-detection.streamlit.app](https://aeris-cloud-detection.streamlit.app)

**Features:**
- Upload single 4-channel TIFF or separate band files
- Real-time MC Dropout inference with progress tracking
- 8-panel comprehensive visualization dashboard
- Downloadable results and quality metrics

### Model Weights

🤗 **HuggingFace Hub:** [https://huggingface.co/Eklavya16/aeris-cloud-detection](https://huggingface.co/Eklavya16/aeris-cloud-detection)

**Files:**
- `Aeris_Model.pth` – Full model checkpoint (state dict + metadata)
- Model architecture: U-Net + ResNet34 encoder
- Input format: 4-channel (R,G,B,NIR), 256×256 patches

---

## 💻 Usage

### Input Requirements

**Option 1: Single 4-Channel File**
```python
# Format: 4-band GeoTIFF or NumPy array
# Shape: (4, 256, 256) or (4, H, W) - will be resized
# Bands: [Red, Green, Blue, NIR]
# Data type: uint8 (0-255) or float32 (0-1)
```

**Option 2: Separate Band Files**
```python
# Four separate single-band TIFF files:
# - red_band.tif
# - green_band.tif
# - blue_band.tif
# - nir_band.tif
```

### Output Interpretations

| Visualization | Description |
|--------------|-------------|
| **Satellite RGB** | True-color composite (original imagery) |
| **Cloud Mask** | Binary segmentation (cloud=1, clear=0) |
| **Cloud Probability** | Continuous 0-1 probability map |
| **Cloud Overlay** | RGB with red cloud overlay |
| **Uncertainty Map** | Pixel-level prediction variance |
| **Confidence Map** | Model certainty (high=reliable) |
| **Cloud Boundaries** | Edge detection on cloud mask |
| **Quality Zones** | 4-tier quality assessment |

### Metrics Provided

**Cloud Coverage Analysis:**
- Cloud coverage percentage
- Category (Clear Sky / Few Clouds / Partly Cloudy / Mostly Cloudy / Overcast)
- Cloud pixel confidence
- Clear pixel confidence

**Uncertainty Metrics:**
- Mean uncertainty across image
- Maximum uncertainty (worst case)
- Quality score (% pixels with uncertainty < 0.1)
- Reliability score (inverse uncertainty)

---

## 🔬 Technical Highlights

### 1. Transfer Learning Strategy
- ResNet34 encoder pretrained on ImageNet (natural images)
- Fine-tuned on Landsat 8 multispectral data
- Adapts low-level features to satellite imagery domain

### 2. Combined Loss Function
```python
Loss = 0.5 × Dice Loss + 0.5 × BCE Loss
```
- **Dice Loss:** Handles class imbalance, optimizes overlap
- **BCE Loss:** Pixel-level cross-entropy, sharp boundaries
- Weighted combination balances regional and pixel-wise objectives

### 3. Monte Carlo Dropout Inference
- Runs the model multiple times with dropout turned on to measure prediction stability  
- Calculates confidence scores to detect uncertain or risky regions in the image

### 4. Calibration
- Expected Calibration Error (ECE) = 0.70%
- Reliability diagrams show near-perfect alignment
- Enables threshold-based filtering for operational use

### 5. Production Deployment
- Model hosted on HuggingFace for version control
- Automatic download and caching on first run
- Streamlit app with custom CSS for professional UI
- Real-time progress bars for MC Dropout iterations

---

## 📊 Visualization Dashboard

The application provides **8 synchronized visualizations** for comprehensive cloud analysis:

1. **Satellite Image (RGB)** – Original scene in true color
2. **Cloud Mask** – Binary segmentation with coverage %
3. **Cloud Probability** – Continuous probability heatmap
4. **Cloud Overlay** – RGB with semi-transparent red cloud mask
5. **Uncertainty Map** – Variance from MC Dropout (hot colormap)
6. **Confidence Map** – Model certainty (high = green/yellow)
7. **Cloud Boundaries** – Sobel edge detection on mask
8. **Quality Zones** – 4-tier quality assessment (Excellent → Poor)

---

## 🛠️ Applications

### Earth Observation Workflows
- **Preprocessing:** Remove cloudy observations from time series
- **Compositing:** Create cloud-free mosaics from multi-temporal data
- **Quality Control:** Flag scenes requiring manual review

### Vegetation Monitoring
- **NDVI Calculation:** Ensure clear-sky NIR measurements
- **Crop Health:** Accurate vegetation index computation
- **Phenology:** Track seasonal changes without cloud contamination

### Water Resource Management
- **Reservoir Monitoring:** Water level estimation from optical imagery
- **Flood Mapping:** Identify inundated areas in disaster response
- **Coastal Analysis:** Shoreline detection and erosion monitoring

---

## 🔍 Model Interpretability

### Uncertainty Interpretation

| Mean Uncertainty | Interpretation | Recommendation |
|-----------------|----------------|----------------|
| **< 0.05** | Very confident | Highly reliable, use directly |
| **0.05 - 0.10** | Confident | Reliable, safe for most applications |
| **0.10 - 0.20** | Moderate | Review high-stakes applications |
| **> 0.20** | Low confidence | Manual verification recommended |

### Common High-Uncertainty Cases
- Thin cirrus clouds (semi-transparent)
- Cloud shadows (can mimic water bodies)
- Bright urban areas (high reflectance like clouds)
- Snow/ice cover (spectral similarity to clouds)
- Haze and atmospheric scattering

---

## ⚠️ Limitations & Future Work

### Current Limitations
- **Fixed Resolution:** Requires 256×256 patches (automated resizing available)
- **Landsat 8 Only:** Trained specifically on Landsat OLI spectral bands
- **Cloud Types:** May struggle with very thin cirrus or mixed cloud/shadow
- **Domain Shift:** Performance may degrade on other satellite sensors

### Future Enhancements
- [ ] Multi-sensor support (Sentinel-2, MODIS, etc.)
- [ ] Cloud shadow detection as separate class
- [ ] Thin cirrus classification refinement
- [ ] Temporal consistency using time-series data
- [ ] Self-supervised pretraining on unlabeled imagery
- [ ] Larger patch sizes (512×512, 1024×1024)
- [ ] Real-time processing pipeline for continuous monitoring

---

## 📧 Contact & Contributions

**Developer:** Eklavya Mohan Agrawal

**Links:**
- 🌐 [Live Demo](https://aeris-cloud-detection.streamlit.app)
- 🤗 [Model Hub](https://huggingface.co/Eklavya16/aeris-cloud-detection)
- 📦 [Dataset](https://www.kaggle.com/datasets/sorour/38cloud-cloud-segmentation-in-satellite-images)

**Contributions Welcome!**  
Issues, bug reports, and feature requests are appreciated. Please open an issue on GitHub for discussion.

---

## 📄 License

This project is licensed under the **MIT License**.  
Model weights are available under **Apache 2.0**.  
38-Cloud dataset follows its original license terms.

---

<p align="center">
  <i>Advancing Earth observation through deep learning and uncertainty-aware cloud detection</i>
</p>

<p align="center">
  ⭐ Star this repository if you found it useful! ⭐
</p>
