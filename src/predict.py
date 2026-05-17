import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
from config import CNN_MODEL_PATH
from config import UNET_MODEL_PATH
from config import IMG_SIZE
from config import CLASS_NAMES
from utils import compute_leaf_pixel_fraction


def load_image(img_path: str) -> np.ndarray:
    img = keras_image.load_img(img_path, target_size=IMG_SIZE)
    return keras_image.img_to_array(img) / 255.0


def predict_growth_stage(img_path: str) -> dict:
    unet = tf.keras.models.load_model(UNET_MODEL_PATH, compile=False)
    img_array = load_image(img_path)
    mask = unet.predict(np.expand_dims(img_array, axis=0))[0, :, :, 0]
    leaf_density = compute_leaf_pixel_fraction(mask)

    cnn = tf.keras.models.load_model(CNN_MODEL_PATH)
    probs = cnn.predict(np.expand_dims(img_array, axis=0))[0]
    pred_class = CLASS_NAMES[np.argmax(probs)]
    confidence = float(np.max(probs))

    result = {
        "predicted_stage": pred_class,
        "confidence": round(confidence, 4),
        "leaf_density_lpf": round(leaf_density, 4),
        "all_probabilities": {
            cls: round(float(prob), 4)
            for cls, prob in zip(CLASS_NAMES, probs)
        }
    }
    print(result)
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
    predict_growth_stage(sys.argv[1])
