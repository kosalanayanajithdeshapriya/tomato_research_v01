import torch
import pytest
from src.unet import UNet
from src.model import TomatoFusionModel


BATCH    = 2
DUMMY    = torch.randn(BATCH, 3, 224, 224)


<<<<<<< Updated upstream
class SimpleUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.enc        = UNetBlock(in_channels, 64)
        self.pool       = nn.MaxPool2d(2)
        self.bottleneck = UNetBlock(64, 128)
        self.up         = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec        = UNetBlock(128, 64)
        self.final      = nn.Conv2d(64, out_channels, 1)
=======
class TestUNet:
    def test_output_shape(self):
        model  = UNet()
        output = model(DUMMY)
        assert output.shape == (BATCH, 1, 224, 224)
>>>>>>> Stashed changes

    def test_output_range(self):
        model  = UNet()
        output = model(DUMMY)
        assert output.min() >= 0.0
        assert output.max() <= 1.0

    def test_sigmoid_applied(self):
        model  = UNet()
        output = model(DUMMY)
        assert output.min() >= 0.0 and output.max() <= 1.0


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

<<<<<<< Updated upstream
def test_cnn_output_shape():
    assert os.path.exists(CNN_MODEL_PATH), "Model file missing"
    model = build_model()
    model.load_state_dict(torch.load(CNN_MODEL_PATH, weights_only=True))
    model.eval()
    dummy = torch.zeros(1, 3, IMG_SIZE[0], IMG_SIZE[1]).to(DEVICE)
    with torch.no_grad():
        preds = model(dummy)
    assert preds.shape == (1, NUM_CLASSES), \
        f"Expected (1, {NUM_CLASSES}), got {preds.shape}"


def test_cnn_output_is_probability():
    assert os.path.exists(CNN_MODEL_PATH), "Model file missing"
    model = build_model()
    model.load_state_dict(torch.load(CNN_MODEL_PATH, weights_only=True))
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
    with open(METRICS_PATH) as f:
        m = json.load(f)
    for key in ["val_accuracy", "train_accuracy", "f1_score", "precision", "recall"]:
        assert key in m, f"Missing key: {key}"


def test_val_accuracy_above_threshold():
    # FIX: skip in CI — synthetic random data makes accuracy meaningless
    if IS_CI:
        pytest.skip("Skipping accuracy threshold test in CI — synthetic data only")
    with open(METRICS_PATH) as f:
        m = json.load(f)
    acc = m["val_accuracy"]
    assert acc >= MIN_ACCURACY, \
        f"val_accuracy {acc} < threshold {MIN_ACCURACY}"


def test_f1_score_above_threshold():
    # FIX: skip in CI — synthetic random data makes F1 meaningless
    if IS_CI:
        pytest.skip("Skipping F1 threshold test in CI — synthetic data only")
    with open(METRICS_PATH) as f:
        m = json.load(f)
    f1 = m["f1_score"]
    assert f1 >= 0.05, f"F1 score {f1} is too low"


# ── U-Net Tests ───────────────────────────────────────────────────────────────
def test_unet_model_loads():
    if not os.path.exists(UNET_MODEL_PATH):
        pytest.skip("U-Net model not yet trained")
    model = SimpleUNet(in_channels=3, out_channels=1).to(DEVICE)
    model.load_state_dict(torch.load(UNET_MODEL_PATH, weights_only=True))
    assert model is not None


def test_unet_output_shape():
    if not os.path.exists(UNET_MODEL_PATH):
        pytest.skip("U-Net model not yet trained")
    model = SimpleUNet(in_channels=3, out_channels=1).to(DEVICE)
    model.load_state_dict(torch.load(UNET_MODEL_PATH, weights_only=True))
    model.eval()
    dummy = torch.zeros(1, 3, IMG_SIZE[0], IMG_SIZE[1]).to(DEVICE)
    with torch.no_grad():
        mask = model(dummy)
    assert mask.shape == (1, 1, IMG_SIZE[0], IMG_SIZE[1]), \
        f"Expected (1, 1, {IMG_SIZE[0]}, {IMG_SIZE[1]}), got {mask.shape}"
=======
    def test_leaf_density_range(self):
        model = TomatoFusionModel()
        lpf   = model.get_leaf_density(DUMMY[0:1])
        assert 0.0 <= lpf <= 1.0
>>>>>>> Stashed changes
