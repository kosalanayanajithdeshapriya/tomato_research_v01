
# Tomato Growth Stage Classification — MLOps CI/CD Pipeline

A deep learning pipeline for classifying tomato crop growth stages using
transfer learning (ResNet50) and semantic segmentation (U-Net).
Built with a full MLOps workflow including CI/CD via GitHub Actions.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-orange)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-green)
![License](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-red)
---

## 📌 Project Overview

This project detects the growth stage of tomato plants using leaf density
estimation and convolutional neural networks. It is part of an undergraduate
research project at the University of Jaffna, Faculty of Engineering,
Department of Computer Engineering.

**Growth Stages Classified:**
- 🌱 Seeding
- 🌿 Developing (Vegetative)
- 🌸 Flowering
- 🍅 Fruiting

---

## 🏗️ Project Structure

```
CI CD/
│
├── .github/
│   └── workflows/
│       └── ml_pipeline.yml        # GitHub Actions CI/CD pipeline
│
├── data/
│   ├── developing/                # Vegetative stage images
│   ├── flowering/
│   ├── fruiting/
│   └── seeding/
│
├── src/
│   ├── train.py                   # ResNet50 training (Phase 1 + Phase 2)
│   ├── evaluate.py                # Model evaluation + confusion matrix
│   └── deploy.py                  # Hugging Face Hub deployment
│
├── tests/
│   └── test_model.py              # PyTorch model unit tests (pytest)
│
├── outputs/
│   ├── models/
│   │   └── cnn_classifier.pth     # Saved model weights
│   ├── plots/
│   │   ├── confusion_matrix.png
│   │   └── metrics_bar.png
│   └── metrics.json               # Training metrics
│
├── requirements.txt               # CPU dependencies (for CI)
├── requirements-gpu.txt           # GPU/CUDA dependencies (for local)
└── README.md
```

---

## 🧠 Model Architecture

### CNN Classifier (ResNet50)
- **Backbone:** ResNet50 pretrained on ImageNet
- **Phase 1:** Train classification head only (frozen backbone)
- **Phase 2:** Fine-tune `layer3`, `layer4`, and `fc` layers
- **Head:** Linear(2048→512) → BN → ReLU → Dropout(0.4) → Linear(512→256) → ReLU → Dropout(0.3) → Linear(256→4)
- **Input size:** 224×224 RGB

### U-Net Segmentation *(in progress)*
- Pixel-level leaf segmentation for leaf density estimation
- Encoder-decoder with skip connections
- Input: 224×224 RGB → Output: 224×224 binary mask

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA support (recommended)

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/tomato_research_v01.git
cd tomato_research_v01
```

### 2. Create Virtual Environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate
```

### 3. Install Dependencies

**For local GPU training (RTX 4060 / CUDA 12.1):**
```bash
pip install -r requirements-gpu.txt
```

**For CPU only:**
```bash
pip install -r requirements.txt
```

### 4. Verify GPU Setup
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

---

## 🚀 Training

```bash
python src/train.py
```

**What it does:**
1. Loads images from `data/` using `ImageFolder`
2. Phase 1 — trains classification head (30 epochs, lr=0.0005)
3. Phase 2 — fine-tunes top ResNet layers (20 epochs, lr=1e-5)
4. Saves best model to `outputs/models/cnn_classifier.pth`
5. Saves metrics to `outputs/metrics.json`

---

## 📊 Evaluation

```bash
python src/evaluate.py
```

Generates:
- Classification report (precision, recall, F1 per class)
- Confusion matrix → `outputs/plots/confusion_matrix.png`
- Metrics bar chart → `outputs/plots/metrics_bar.png`

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

Tests cover:
- Model output shape `(batch, 4)`
- Model loads saved weights correctly
- Softmax probabilities sum to 1.0
- `metrics.json` has all required keys and valid value ranges
- U-Net output shape `(batch, 1, H, W)` *(skipped if not yet trained)*

---

## 🔄 CI/CD Pipeline

Every `git push` to `main` triggers the GitHub Actions pipeline:

```
Checkout → Install deps → Lint (flake8) → Create CI dataset
     → Train model → Run tests → Evaluate → Upload artifacts → Deploy to HuggingFace
```

- CI uses **40 sample images** (random noise) to verify the pipeline runs
- Accuracy threshold is lowered to `0.30` in CI (`CI=true` env var)
- Real accuracy threshold is `0.55` for local/production training
- Model artifacts are uploaded after every successful run

---

## 📈 Results

| Metric | Value |
|---|---|
| Train Accuracy | ~93% |
| Val Accuracy | ~75–93% |
| Model Size | ~97.8 MB (ResNet50) |
| Training Time (RTX 4060) | ~5–10 min |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| PyTorch 2.5.1 + CUDA 12.1 | Model training |
| torchvision | Data loading & transforms |
| scikit-learn | Metrics & classification report |
| GitHub Actions | CI/CD automation |
| Hugging Face Hub | Model deployment |
| pytest | Unit testing |
| flake8 | Code linting |

---

## 👤 Author

**Kosala Deshapriya**
Undergraduate — Computer Engineering
University of Jaffna, Sri Lanka

---

## 📄 License

This project is licensed under the MIT License.
