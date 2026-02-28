# PBVS 2025 MAVIC-C: StagAI MAVIC-V2 Solution

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org/get-started/locally/)

## 🚀 Overview
A high-performance deep learning solution designed to achieve Rank 1 in the **MAVIC-C Challenge** (PBVS 2025). The architecture leverages **Knowledge Distillation**, **Supervised Contrastive Learning**, and **Automatic Mixed Precision (AMP)** to optimize for both in-distribution accuracy and out-of-distribution (OOD) detection.

## 🛠 Features
- **Multi-Modal Feature Matching**: Distills knowledge from Electro-Optical (EO) teachers to SAR students.
- **SupCon Clustering**: Enhances OOD detection by tightening class-specific latent clusters.
- **Unified Dispatcher (`app.py`)**: Automatic hardware detection. Runs locally on GPU or ships to an A100 cluster if local GPU is unavailable.
- **Dockerized Environment**: Full compatibility across any Linux/Windows/macOS server with one command.

## 📁 Repository Structure
```bash
pbvs_mavic/
├── app.py              # Main Dispatcher (Auto-GPU detection)
├── Dockerfile          # Container environment
├── requirements.txt    # Project dependencies
├── src/                # Core implementation
│   ├── models.py       # Multi-modal backbone (ResNet/EfficientNet)
│   ├── losses.py       # KD, SupCon, and smoothed CE
│   ├── data.py         # SAR-specific transforms & loaders
│   ├── train.py        # Optimized AMP Training loop
│   ├── evaluate.py     # Competition metric suite
│   └── inference.py    # submission.csv generator
└── scripts/            # Validation & testing suite
```

## ⚡ Getting Started

### 1. Simple Clone & Run
```bash
git clone https://github.com/sidharthkumarpradhan/pbvs_mavic.git
cd pbvs_mavic
pip install -r requirements.txt
python app.py
```

### 2. Docker Execution (Recommended for Remote Servers)
```bash
docker build -t pbvs_mavic .
docker run --gpus all pbvs_mavic
```

### 3. Google Colab / Remote GPU
1. Use the [StagAI_MAVIC_V2_Colab.ipynb](https://github.com/sidharthkumarpradhan/pbvs_mavic/blob/master/StagAI_MAVIC_V2_Colab.ipynb) provided.
2. Or use **Modal** for zero-intervention A100 execution:
   ```bash
   pip install modal
   modal setup
   python app.py
   ```

## 🏆 Rank 1 Strategy
The configuration is tuned for:
- `Score = (0.75 * Accuracy) + (0.25 * AUROC)`
- Advanced Logit Normalization (Z-score) is applied during inference to calibrate OOD scores.

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
