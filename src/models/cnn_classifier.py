from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import VGG19
from tensorflow.keras.optimizers import Adam
from config import IMG_SIZE
from config import CHANNELS
from config import NUM_CLASSES
from config import LEARNING_RATE


def build_classification_head(base_model, num_classes):
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return outputs


def build_resnet50(input_shape=(IMG_SIZE[0], IMG_SIZE[1], CHANNELS)):
    base = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    base.trainable = False
    outputs = build_classification_head(base, NUM_CLASSES)
    model = Model(base.input, outputs, name="ResNet50_TomatoStage")
    model.compile(
        optimizer=Adam(LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def build_densenet121(input_shape=(IMG_SIZE[0], IMG_SIZE[1], CHANNELS)):
    base = DenseNet121(weights="imagenet", include_top=False, input_shape=input_shape)
    base.trainable = False
    outputs = build_classification_head(base, NUM_CLASSES)
    model = Model(base.input, outputs, name="DenseNet121_TomatoStage")
    model.compile(
        optimizer=Adam(LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def build_inceptionv3(input_shape=(299, 299, CHANNELS)):
    base = InceptionV3(weights="imagenet", include_top=False, input_shape=input_shape)
    base.trainable = False
    outputs = build_classification_head(base, NUM_CLASSES)
    model = Model(base.input, outputs, name="InceptionV3_TomatoStage")
    model.compile(
        optimizer=Adam(LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def build_vgg19(input_shape=(IMG_SIZE[0], IMG_SIZE[1], CHANNELS)):
    base = VGG19(weights="imagenet", include_top=False, input_shape=input_shape)
    base.trainable = False
    outputs = build_classification_head(base, NUM_CLASSES)
    model = Model(base.input, outputs, name="VGG19_TomatoStage")
    model.compile(
        optimizer=Adam(LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model
