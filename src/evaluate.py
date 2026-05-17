import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from config import CNN_MODEL_PATH
from config import METRICS_PATH
from config import CLASS_NAMES
from config import MIN_ACCURACY
from data_loader import get_data_generators
from utils import load_metrics


def evaluate():
    print("[INFO] Loading saved model...")
    model = tf.keras.models.load_model(CNN_MODEL_PATH)

    print("[INFO] Loading validation data...")
    _, val_gen = get_data_generators()
    val_gen.reset()

    print("[INFO] Running predictions...")
    y_pred_probs = model.predict(val_gen)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = val_gen.classes[:len(y_pred)]

    print("\n[Classification Report]")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    print("[Confusion Matrix]")
    print(confusion_matrix(y_true, y_pred))

    metrics = load_metrics(METRICS_PATH)
    val_acc = metrics.get("val_accuracy", 0)

    if val_acc < MIN_ACCURACY:
        print(f"[FAIL] val_accuracy {val_acc} < {MIN_ACCURACY}")
        sys.exit(1)

    print(f"[PASS] val_accuracy {val_acc} >= {MIN_ACCURACY} - model accepted.")


if __name__ == "__main__":
    evaluate()
