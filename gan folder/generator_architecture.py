import tensorflow as tf
from tensorflow.keras import layers, models

def build_generator(latent_dim=100):
    model = models.Sequential(name="Generator")

    model.add(layers.Dense(7 * 7 * 256, input_dim=latent_dim))
    model.add(layers.Reshape((7, 7, 256)))

    model.add(layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Conv2D(3, kernel_size=7, activation="tanh", padding="same"))

    return model
