import os, numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import DATA_DIR, IMG_SIZE, BATCH_SIZE, VALIDATION_SPLIT, AUGMENT


def get_data_generators():
    """Return train and validation generators for CNN classification."""
    if AUGMENT:
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            validation_split=VALIDATION_SPLIT,
            rotation_range=20,
            width_shift_range=0.15,
            height_shift_range=0.15,
            shear_range=0.1,
            zoom_range=0.2,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode="nearest",
        )
    else:
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            validation_split=VALIDATION_SPLIT
        )

    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=VALIDATION_SPLIT
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


def get_segmentation_generators(mask_dir: str):
    """Return image/mask pairs for U-Net segmentation training."""
    image_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=VALIDATION_SPLIT)
    mask_datagen  = ImageDataGenerator(rescale=1.0 / 255, validation_split=VALIDATION_SPLIT)

    img_train = image_datagen.flow_from_directory(
        os.path.join(DATA_DIR, "images"),
        target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode=None, subset="training", seed=42
    )
    mask_train = mask_datagen.flow_from_directory(
        mask_dir,
        target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode=None, color_mode="grayscale",
        subset="training", seed=42
    )
    return zip(img_train, mask_train)
