import os, json
import numpy as np
import matplotlib.pyplot as plt
from config import OUTPUT_DIR, MODEL_DIR, CLASS_NAMES


def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "plots"), exist_ok=True)


def save_metrics(metrics: dict, path: str):
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"[INFO] Metrics saved → {path}")


def load_metrics(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def plot_training_history(history, model_name: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"], label="Train Acc")
    axes[0].plot(history.history["val_accuracy"], label="Val Acc")
    axes[0].set_title(f"{model_name} — Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title(f"{model_name} — Loss")
    axes[1].legend()

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "plots", f"{model_name}_history.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] Plot saved → {save_path}")


def compute_leaf_pixel_fraction(mask: np.ndarray) -> float:
    """Compute leaf density (Leaf Pixel Fraction) from a binary U-Net mask."""
    leaf_pixels = np.sum(mask > 0.5)
    total_pixels = mask.size
    return float(leaf_pixels / total_pixels)
