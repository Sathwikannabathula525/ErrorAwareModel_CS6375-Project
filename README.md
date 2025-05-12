# ErrorAwareModel_CS6375-Project
# 🧠 Error-Aware Model Training for Compiler-Induced Perturbations

This repository contains the implementation for our CS 6375 Machine Learning project at The University of Texas at Dallas. The project focuses on making deep learning models more robust to arithmetic errors introduced by compiler optimizations during deployment.

All implementation is done in a **single script**: `project1_v2.py`.

---

## 📌 Overview

Compiler-level transformations such as operator fusion, quantization, and reduced precision can introduce small but harmful arithmetic errors. These may silently propagate through deep learning models, degrading prediction accuracy.
This project proposes a **training-time solution**:
- Inject random numerical perturbations (Gaussian noise) into layer outputs
- Use a custom loss function that penalizes instability under noise
- Train models to maintain stable predictions despite these perturbations

## 🚀 Features

- ✅ Pretrained ResNet18 modified for Tiny ImageNet
- ✅ Gaussian noise injection during forward pass (via hooks)
- ✅ Custom error-aware loss function (`ErrorAwareLoss`)
- ✅ Comparison with standard training baseline
- ✅ Logs clean and noisy accuracy for both models

---

## 📁 Project Structure

ErrorAwareModel.py # Full implementation and entry point

---

## 📦 Requirements

- Python 3.8+
- PyTorch ≥ 1.12
- torchvision
- tqdm
- numpy
- (optional) matplotlib

Install all required packages:
pip install torch torchvision tqdm numpy

🧪 Dataset
Tiny ImageNet (subset of ImageNet with 200 classes, 64x64 images)
Automatically downloaded on first run (~237MB)
Source: http://cs231n.stanford.edu/tiny-imagenet-200.zip

Run the training and evaluation
python ErrorAwareModel.py

🔬 Implementation Highlights
Custom Modules:
CompilerErrorSimulator: injects Gaussian noise into tensors
ErrorAwareTrainingWrapper: uses PyTorch forward hooks on Conv2d/Linear layers
ErrorAwareLoss: penalizes prediction divergence under noisy activations
All training and evaluation logic implemented in ErrorAwareModel.py

📊 Results
Model	Clean Accuracy	With Noise	Accuracy Drop
Error-Aware	55.12%	55.01%	0.11%
Standard	48.10%	47.96%	0.14%
The error-aware model not only achieves higher clean accuracy, but is also more resilient under simulated noise conditions.

📂 Output Files
error_aware_model_weights.pth – saved model after error-aware training
standard_model_weights.pth – baseline model
## Output files were not included due to size constraint

👨‍💻 Authors
Sai Sathwik Annabathula
Sai Charan Palvai
CS 6375 – Machine Learning
The University of Texas at Dallas
