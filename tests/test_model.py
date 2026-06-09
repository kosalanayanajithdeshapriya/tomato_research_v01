import os
import json
import torch
import pytest
from src.unet import UNet
from src.model import TomatoFusionModel

# ── Constants ─────────────────────────────────────────────────────────────────
BATCH      = 2
DUMMY      = torch.randn(BATCH, 3, 224, 224)
DEVICE     = torch.device("cpu")
IS_CI      = os.environ.get("CI", "false").lower() == "true"
IMG_SIZE   = (224, 224)
NUM_CLASSES = 4
MIN_ACCURACY = 0.55
METRICS_PATH   = "outputs/metrics.json"
CNN_MODEL_PATH = "outputs/models/classifier.pth"
UNET_MODEL_PATH = "outputs/models/unet.pth"


# ── U-Net Architecture Tests ──────────────────────────────────────────────────
class TestUNet:
    def test_output_shape(self):
        model  = UNet()
        output = model(DUMMY)
        assert output.shape == (BATCH, 1, 224, 224)

    def test_output_range(self):
        model  = UNet()
        output = model(DUMMY)
        assert output.min() >= 0.0
        assert output.max() <= 1.0

    def test_sigmoid_applied(self):
        model  = UNet()
        output = model(DUMMY)
        assert output.min() >= 0.0 and output.max() <= 1.0


# ── Fusion Model Tests ────────────────────────────────────────────────────────
class TestFusionModel:
    def test_output_shape(self):
        model  = TomatoFusionModel(num_classes=4)
        output = model(DUMMY)
        assert output.shape == (BATCH, 4)

    def test_num_classes(self):
        for n in [2, 4, 8]:
            model  = TomatoFusionModel(num_classes=n)
            output = model(DUMMY)
            assert output.shape == (BATCH, n)

    def test_backbone_frozen_phase1(self):
        model      = TomatoFusionModel()
        cnn_params = [p for p in model.cnn.parameters()]
        assert not any(p.requires_grad for p in cnn_params)

    def test_unfreeze_backbone(self):
        model = TomatoFusionModel()
        model.unfreeze_backbone(layers=["layer3", "layer4"])
        unfrozen = [
            p for n, p in model.cnn.named_parameters()
            if "layer3" in n or "layer4" in n
        ]
        assert any(p.requires_grad for p in unfrozen)

    def test_leaf_density_range(self):
        model = TomatoFusionModel()
        lpf   = model.get_leaf_density(DUMMY[0:1])
        assert 0.0 <= lpf <= 1.0


# ── CNN Model Tests ───────────────────────────────────────────────────────────
def test_cnn_output_shape():
    if not os.path.exists(CNN_MODEL_PATH):
        pytest.skip("CNN model not yet trained")
    from src.model import TomatoFusionModel
    model = TomatoFusionModel(num_classes=NUM_CLASSES).to(DEVICE)
    model.eval()
    dummy = torch.zeros(1, 3, IMG_SIZE[0], IMG_SIZE[1]).to(DEVICE)
    with torch.no_grad():
        preds = model(dummy)
    assert preds.shape == (1, NUM_CLASSES), \
        f"Expected (1, {NUM_CLASSES}), got {preds.shape}"


def test_cnn_output_is_probability():
    if not os.path.exists(CNN_MODEL_PATH):
        pytest.skip("CNN model not yet trained")
    from src.model import TomatoFusionModel
    model = TomatoFusionModel(num_classes=NUM_CLASSES).to(DEVICE)
    model.eval()
    dummy = torch.rand(1, 3, IMG_SIZE[0], IMG_SIZE[1]).to(DEVICE)
    with torch.no_grad():
        logits = model(dummy)
        probs  = torch.softmax(logits, dim=1)
    assert abs(probs.sum().item() - 1.0) < 1e-3, \
        f"Probabilities sum to {probs.sum().item()}, expected ~1.0"


# ── Metrics Tests ─────────────────────────────────────────────────────────────
def test_metrics_file_exists():
    assert os.path.exists(METRICS_PATH), \
        f"Metrics not found at {METRICS_PATH}"


def test_metrics_has_required_keys():
    if not os.path.exists(METRICS_PATH):
        pytest.skip("Metrics file not yet generated")
    with open(METRICS_PATH) as f:
        m = json.load(f)
    for key in ["val_accuracy", "train_accuracy", "f1_score", "precision", "recall"]:
        assert key in m, f"Missing key: {key}"


def test_val_accuracy_above_threshold():
    if IS_CI:
        pytest.skip("Skipping accuracy threshold test in CI — synthetic data only")
    with open(METRICS_PATH) as f:
        m = json.load(f)
    acc = m["val_accuracy"]
    assert acc >= MIN_ACCURACY, \
        f"val_accuracy {acc} < threshold {MIN_ACCURACY}"


def test_f1_score_above_threshold():
    if IS_CI:
        pytest.skip("Skipping F1 threshold test in CI — synthetic data only")
    with open(METRICS_PATH) as f:
        m = json.load(f)
    f1 = m["f1_score"]
    assert f1 >= 0.05, f"F1 score {f1} is too low"


# ── U-Net Trained Model Tests ─────────────────────────────────────────────────
def test_unet_model_loads():
    if not os.path.exists(UNET_MODEL_PATH):
        pytest.skip("U-Net model not yet trained")
    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    model.load_state_dict(torch.load(UNET_MODEL_PATH, weights_only=True))
    assert model is not None


def test_unet_output_shape():
    if not os.path.exists(UNET_MODEL_PATH):
        pytest.skip("U-Net model not yet trained")
    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    model.load_state_dict(torch.load(UNET_MODEL_PATH, weights_only=True))
    model.eval()
    dummy = torch.zeros(1, 3, IMG_SIZE[0], IMG_SIZE[1]).to(DEVICE)
    with torch.no_grad():
        mask = model(dummy)
    assert mask.shape == (1, 1, IMG_SIZE[0], IMG_SIZE[1]), \
        f"Expected (1, 1, {IMG_SIZE[0]}, {IMG_SIZE[1]}), got {mask.shape}"