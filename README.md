# NHRDNet: A Non-contact Heart Rate Detection Network Based on Convolutional Attention Network

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.12+](https://img.shields.io/badge/pytorch-1.12+-red.svg)](https://pytorch.org/)

This repository is the official implementation of **NHRDNet** (Non-contact Heart Rate Detection Network), a lightweight convolutional attention-based model for non-contact heart rate detection via remote Photoplethysmography (rPPG). The model is proposed in the paper accepted by ICME [可替换为实际会议/期刊名], focusing on balancing accuracy and efficiency for edge-device deployment.

## 📝 Abstract
Remote Photoplethysmography (rPPG) enables non-contact heart rate detection from facial video frames, but existing models suffer from trade-offs between accuracy, computational cost, and robustness to noise (e.g., background interference, head motion). To address this issue, we propose NHRDNet, a lightweight convolutional attention network with three core innovations:
1. **RespConv Module**: Feature reuse via channel splitting and residual connection to expand receptive field without increasing parameters.
2. **LP (Location Pooling) Module**: Spatial attention enhancement for key facial regions (e.g., forehead, cheeks) to suppress background noise.
3. **MSPF (Multi-scale Feature Pyramid Fusion) Module**: Multi-scale feature fusion to retain global context and improve generalization.

NHRDNet achieves state-of-the-art performance on benchmark datasets (UBFC-rPPG, PURE) with only 2.6M parameters, making it suitable for real-time deployment on resource-constrained devices (e.g., smartphones, wearables).

## 🚀 Key Features
- **Lightweight Design**: Only 2.6M parameters (lower than PhysNet/PhysFormer), low memory footprint (≤5MB).
- **High Accuracy**: MAE=0.88 on UBFC-F and MAE=2.63 on PURE-F, outperforming traditional rPPG methods.
- **Efficient Inference**: Real-time inference (≤15ms per frame) on CPU/GPU, supporting 30fps video input.
- **Robust to Noise**: Specialized modules for background/ motion noise suppression, stable performance in real-world scenarios.
- **Full Reproducibility**: Complete training/evaluation code, pre-trained weights, and dataset processing scripts.

