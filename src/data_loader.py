import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import DATA_DIR
from config import IMG_SIZE
from config import BATCH_SIZE
from config import VALIDATION_SPLIT
from config import AUGMENT


def get_data_generators():
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
