import tensorflow as tf
from tensorflow.keras import layers, models

def build_discriminator(input_shape=(224, 224, 3)):
    model = models.Sequential(name="Discriminator")

    model.add(layers.Conv2D(64, kernel_size=4, strides=2, padding="same",
                            input_shape=input_shape))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, kernel_size=4, strides=2, padding="same"))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation="sigmoid"))

    return model
