import os
import sys
import json
import warnings
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR        = "data/"
OUTPUT_DIR      = "outputs/"
MODEL_DIR       = os.path.join(OUTPUT_DIR, "models")
PLOTS_DIR       = os.path.join(OUTPUT_DIR, "plots")
METRICS_PATH    = os.path.join(OUTPUT_DIR, "metrics.json")
CNN_MODEL_PATH  = os.path.join(MODEL_DIR, "cnn_classifier.keras")   # native Keras format

IMG_SIZE        = (224, 224)
BATCH_SIZE      = 16
CHANNELS        = 3
EPOCHS_PHASE1   = 30
EPOCHS_PHASE2   = 20
LEARNING_RATE   = 0.0005        # lowered from 0.001
FINE_TUNE_LR    = 1e-5
VALIDATION_SPLIT = 0.10         # lowered from 0.15 — more data for training
MIN_ACCURACY    = 0.55
FINE_TUNE_LAYERS = 30           # unfreeze last N layers of ResNet50

CLASS_NAMES = ["developing", "flowering", "fruiting", "seeding"]  # ✅ correct
NUM_CLASSES     = len(CLASS_NAMES)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


# ── Data Generators ───────────────────────────────────────────────────────────
def load_data_generators():
    print("[INFO] Loading data generators...")

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=VALIDATION_SPLIT,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True,
        brightness_range=[0.7, 1.3],
        fill_mode="nearest",
    )

    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=VALIDATION_SPLIT,
    )

    train_gen = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        seed=42,
    )

    val_gen = val_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        seed=42,
    )

    return train_gen, val_gen


# ── Class Weights ─────────────────────────────────────────────────────────────
def get_class_weights(train_gen):
    labels = train_gen.classes
    weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
    class_weight_dict = dict(enumerate(weights))
    print(f"[INFO] Class weights: {class_weight_dict}")
    return class_weight_dict


# ── Model Builder ─────────────────────────────────────────────────────────────
def build_model():
    print("[INFO] Building ResNet50 model...")

    inputs = Input(shape=(*IMG_SIZE, CHANNELS))

    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_tensor=inputs,
    )
    base_model.trainable = False   # freeze for Phase 1

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(NUM_CLASSES, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs, name="ResNet50_TomatoStage")
    return model, base_model


# ── Callbacks ─────────────────────────────────────────────────────────────────
def get_callbacks(checkpoint_path, patience=7):
    return [
        EarlyStopping(
            monitor="val_accuracy",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]


# ── Training ──────────────────────────────────────────────────────────────────
def train():
    train_gen, val_gen = load_data_generators()
    class_weights = get_class_weights(train_gen)
    model, base_model = build_model()

    # ── Phase 1: Train head only ───────────────────────────────────────────
    print("[INFO] Phase 1 - Training classification head...")
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    history1 = model.fit(
        train_gen,
        epochs=EPOCHS_PHASE1,
        validation_data=val_gen,
        callbacks=get_callbacks(CNN_MODEL_PATH, patience=7),
        class_weight=class_weights,
    )

    # ── Phase 2: Fine-tune top layers ─────────────────────────────────────
    print("[INFO] Phase 2 - Fine-tuning top layers of ResNet50...")

    # Correctly locate the ResNet50 sub-model by type
    resnet_layer = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            resnet_layer = layer
            break

    if resnet_layer is not None:
        resnet_layer.trainable = True
        for layer in resnet_layer.layers[:-FINE_TUNE_LAYERS]:
            layer.trainable = False
        trainable_count = sum(1 for l in resnet_layer.layers if l.trainable)
        print(f"[INFO] Unfroze last {FINE_TUNE_LAYERS} layers ({trainable_count} trainable layers in base)")
    else:
        print("[WARN] Base model not found, training all layers")

    model.compile(
        optimizer=Adam(learning_rate=FINE_TUNE_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    history2 = model.fit(
        train_gen,
        epochs=EPOCHS_PHASE2,
        validation_data=val_gen,
        callbacks=get_callbacks(CNN_MODEL_PATH, patience=5),
        class_weight=class_weights,
    )

    return model, history1, history2, val_gen


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(model, val_gen):
    print("[INFO] Evaluating model...")
    val_gen.reset()

    val_loss, val_acc = model.evaluate(val_gen, verbose=0)

    val_gen.reset()
    y_pred_probs = model.predict(val_gen, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = val_gen.classes

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        recall    = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1        = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print("\n" + classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        zero_division=0,
    ))

    return val_loss, val_acc, precision, recall, f1


# ── Plot History ──────────────────────────────────────────────────────────────
def save_plot(history1, history2, total_epochs):
    acc  = history1.history["accuracy"]  + history2.history["accuracy"]
    val  = history1.history["val_accuracy"] + history2.history["val_accuracy"]
    loss = history1.history["loss"] + history2.history["loss"]
    vloss= history1.history["val_loss"] + history2.history["val_loss"]
    epochs_range = range(1, len(acc) + 1)
    phase2_start = len(history1.history["accuracy"]) + 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs_range, acc,  label="Train Acc")
    ax1.plot(epochs_range, val,  label="Val Acc")
    ax1.axvline(x=phase2_start, color="gray", linestyle="--", label="Phase 2 start")
    ax1.set_title("Accuracy")
    ax1.legend()

    ax2.plot(epochs_range, loss,  label="Train Loss")
    ax2.plot(epochs_range, vloss, label="Val Loss")
    ax2.axvline(x=phase2_start, color="gray", linestyle="--", label="Phase 2 start")
    ax2.set_title("Loss")
    ax2.legend()

    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, f"{total_epochs}history.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"[INFO] Plot saved - {plot_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    model, history1, history2, val_gen = train()

    total_epochs = (
        len(history1.history["accuracy"]) + len(history2.history["accuracy"])
    )
    train_acc = max(
        max(history1.history["accuracy"]),
        max(history2.history["accuracy"]),
    )

    val_loss, val_acc, precision, recall, f1 = evaluate(model, val_gen)

    metrics = {
        "train_accuracy": round(float(train_acc), 4),
        "val_accuracy":   round(float(val_acc), 4),
        "val_loss":       round(float(val_loss), 4),
        "precision":      round(float(precision), 4),
        "recall":         round(float(recall), 4),
        "f1_score":       round(float(f1), 4),
        "total_epochs":   total_epochs,
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"[INFO] Metrics saved - {METRICS_PATH}")

    model.save(CNN_MODEL_PATH)
    print(f"[INFO] Model saved - {CNN_MODEL_PATH}")

    save_plot(history1, history2, total_epochs)

    print("\n── RESULTS ──────────────────────────────")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    status = "PASS" if val_acc >= MIN_ACCURACY else "FAIL"
    print(f"\n[{status}] val_accuracy {val_acc:.4f} {'≥' if val_acc >= MIN_ACCURACY else '<'} threshold {MIN_ACCURACY}")
    print("[INFO] Training complete.")

    if val_acc < MIN_ACCURACY:
        sys.exit(1)


if __name__ == "__main__":
    main()