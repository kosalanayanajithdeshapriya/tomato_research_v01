import os, sys
sys.path.insert(0, os.path.dirname(__file__))

from config import (EPOCHS, CNN_MODEL_PATH, METRICS_PATH, MIN_ACCURACY)
from utils import ensure_dirs, save_metrics, plot_training_history
from data_loader import get_data_generators
from models.cnn_classifier import build_resnet50

from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


def train():
    ensure_dirs()

    print("[INFO] Loading data generators...")
    train_gen, val_gen = get_data_generators()

    print("[INFO] Building ResNet50 model...")
    model = build_resnet50()
    model.summary()

    print("[INFO] Training model...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=[
            __import__("tensorflow").keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=5, restore_best_weights=True
            ),
            __import__("tensorflow").keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
            ),
        ]
    )

    # Save model
    model.save(CNN_MODEL_PATH)
    print(f"[INFO] Model saved → {CNN_MODEL_PATH}")

    # Evaluate
    val_gen.reset()
    y_pred_probs = model.predict(val_gen)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = val_gen.classes[:len(y_pred)]

    report = classification_report(y_true, y_pred, output_dict=True)
    train_acc = float(history.history["accuracy"][-1])
    val_acc   = float(history.history["val_accuracy"][-1])
    val_loss  = float(history.history["val_loss"][-1])

    metrics = {
        "model": "ResNet50",
        "train_accuracy":  round(train_acc, 4),
        "val_accuracy":    round(val_acc, 4),
        "val_loss":        round(val_loss, 4),
        "precision":       round(report["weighted avg"]["precision"], 4),
        "recall":          round(report["weighted avg"]["recall"], 4),
        "f1_score":        round(report["weighted avg"]["f1-score"], 4),
        "epochs_run":      len(history.history["accuracy"]),
    }

    save_metrics(metrics, METRICS_PATH)
    plot_training_history(history, "ResNet50")

    # CI/CD gate — fail the pipeline if accuracy is too low
    if val_acc < MIN_ACCURACY:
        print(f"[FAIL] Validation accuracy {val_acc:.4f} < threshold {MIN_ACCURACY}")
        sys.exit(1)

    print(f"[PASS] Validation accuracy {val_acc:.4f} ≥ threshold {MIN_ACCURACY}")
    print("[INFO] Training complete.")


if __name__ == "__main__":
    train()
