import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from config import IMG_SIZE
from config import CHANNELS


def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    return x


def encoder_block(x, filters):
    skip = conv_block(x, filters)
    pool = layers.MaxPooling2D(2)(skip)
    return skip, pool


def decoder_block(x, skip, filters):
    x = layers.UpSampling2D(2)(x)
    x = layers.Concatenate()([x, skip])
    x = conv_block(x, filters)
    return x


def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )


def build_unet(input_shape=(IMG_SIZE[0], IMG_SIZE[1], CHANNELS)):
    inputs = layers.Input(shape=input_shape)
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)
    b = conv_block(p4, 1024)
    d1 = decoder_block(b, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)
    outputs = layers.Conv2D(1, 1, activation="sigmoid")(d4)
    model = Model(inputs, outputs, name="UNet_LeafSegmentation")
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", dice_coef]
    )
    return model
