from tensorflow import keras
from tensorflow.keras import layers


def build_classifier_head(embedding_dim, num_classes):
    inputs = keras.Input(shape=(embedding_dim,))

    x = layers.Dense(128, activation="relu")(inputs)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs, name="cry_classifier_head")
