import os

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR        = "data/"
OUTPUT_DIR      = "outputs/"
MODEL_DIR       = os.path.join(OUTPUT_DIR, "models")
PLOTS_DIR       = os.path.join(OUTPUT_DIR, "plots")
METRICS_PATH    = os.path.join(OUTPUT_DIR, "metrics.json")
UNET_MODEL_PATH = os.path.join(MODEL_DIR, "unet_segmentation.h5")
CNN_MODEL_PATH  = os.path.join(MODEL_DIR, "cnn_classifier.keras")  # updated: native Keras format

# ── Image & Training Basics ───────────────────────────────────────────────────
IMG_SIZE         = (224, 224)
BATCH_SIZE       = 16
CHANNELS         = 3

# ── Training Phases ───────────────────────────────────────────────────────────
EPOCHS_PHASE1    = 30       # head-only training
EPOCHS_PHASE2    = 20       # fine-tuning

# ── Learning Rates ────────────────────────────────────────────────────────────
LEARNING_RATE    = 0.0005   # lowered from 0.001 for more stable head training
FINE_TUNE_LR     = 1e-5     # kept low to avoid destroying pretrained weights

# ── Regularization ────────────────────────────────────────────────────────────
DROPOUT_1        = 0.4      # after Dense(512)
DROPOUT_2        = 0.3      # after Dense(256)

# ── Fine-Tuning ───────────────────────────────────────────────────────────────
FINE_TUNE_LAYERS = 30       # number of ResNet50 layers to unfreeze in Phase 2

# ── Data Split ────────────────────────────────────────────────────────────────
VALIDATION_SPLIT = 0.10     # lowered from 0.15 — gives more images to training

# ── Augmentation ─────────────────────────────────────────────────────────────
AUGMENT          = True
AUG_ROTATION     = 30
AUG_WIDTH_SHIFT  = 0.2
AUG_HEIGHT_SHIFT = 0.2
AUG_SHEAR        = 0.2
AUG_ZOOM         = 0.3
AUG_HORIZONTAL_FLIP = True
AUG_BRIGHTNESS   = [0.7, 1.3]

# ── Class Info ────────────────────────────────────────────────────────────────
CLASS_NAMES      = ["Seedling", "Vegetative", "Flowering", "Fruiting"]
NUM_CLASSES      = len(CLASS_NAMES)

# ── Evaluation Threshold ──────────────────────────────────────────────────────
MIN_ACCURACY     = 0.55     # minimum val_accuracy to pass CI/CD gate