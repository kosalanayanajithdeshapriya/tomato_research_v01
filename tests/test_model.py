import os
import sys
import json
import pytest
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import CNN_MODEL_PATH
from config import UNET_MODEL_PATH
from config import METRICS_PATH
from config import IMG_SIZE
from config import NUM_CLASSES
from config import MIN_ACCURACY


def test_cnn_model_file_exists():
    assert os.path.exists(CNN_MODEL_PATH), f"Model not found at {CNN_MODEL_PATH}"


def test_cnn_model_loads():
    model = tf.keras.models.load_model(CNN_MODEL_PATH)
    assert model is not None


def test_cnn_output_shape():
    model = tf.keras.models.load_model(CNN_MODEL_PATH)
    dummy = np.zeros((1, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32)
    preds = model.predict(dummy)
    assert preds.shape == (1, NUM_CLASSES)


def test_cnn_output_is_probability():
    model = tf.keras.models.load_model(CNN_MODEL_PATH)
    dummy = np.random.rand(1, IMG_SIZE[0], IMG_SIZE[1], 3).astype(np.float32)
    preds = model.predict(dummy)
    assert abs(np.sum(preds) - 1.0) < 1e-3


def test_metrics_file_exists():
    assert os.path.exists(METRICS_PATH), f"Metrics not found at {METRICS_PATH}"


def test_metrics_has_required_keys():
    with open(METRICS_PATH) as f:
        m = json.load(f)
    for key in ["val_accuracy", "train_accuracy", "f1_score", "precision", "recall"]:
        assert key in m, f"Missing key: {key}"


def test_val_accuracy_above_threshold():
    with open(METRICS_PATH) as f:
        m = json.load(f)
    acc = m["val_accuracy"]
    assert acc >= MIN_ACCURACY, f"val_accuracy {acc} < threshold {MIN_ACCURACY}"


def test_f1_score_above_threshold():
    with open(METRICS_PATH) as f:
        m = json.load(f)
    f1 = m["f1_score"]
    assert f1 >= 0.05, f"F1 score {f1} is too low"


def test_unet_model_loads():
    if not os.path.exists(UNET_MODEL_PATH):
        pytest.skip("U-Net model not yet trained")
    model = tf.keras.models.load_model(UNET_MODEL_PATH, compile=False)
    assert model is not None


def test_unet_output_shape():
    if not os.path.exists(UNET_MODEL_PATH):
        pytest.skip("U-Net model not yet trained")
    model = tf.keras.models.load_model(UNET_MODEL_PATH, compile=False)
    dummy = np.zeros((1, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32)
    mask = model.predict(dummy)
    assert mask.shape == (1, IMG_SIZE[0], IMG_SIZE[1], 1)
