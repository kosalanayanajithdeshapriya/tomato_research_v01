import os

# Paths
DATA_DIR        = "data/"
OUTPUT_DIR      = "outputs/"
MODEL_DIR       = os.path.join(OUTPUT_DIR, "models")
METRICS_PATH    = os.path.join(OUTPUT_DIR, "metrics.json")
UNET_MODEL_PATH = os.path.join(MODEL_DIR, "unet_segmentation.h5")
CNN_MODEL_PATH  = os.path.join(MODEL_DIR, "cnn_classifier.h5")

# Image settings
IMG_SIZE        = (224, 224)
BATCH_SIZE      = 32
CHANNELS        = 3

# Training
EPOCHS          = 30
LEARNING_RATE   = 0.001
VALIDATION_SPLIT = 0.2
MIN_ACCURACY    = 0.70   # pipeline threshold

# Classes (BBCH growth stages)
CLASS_NAMES = ["Seedling", "Vegetative", "Flowering", "Fruiting"]
NUM_CLASSES = len(CLASS_NAMES)

# Augmentation
AUGMENT = True
