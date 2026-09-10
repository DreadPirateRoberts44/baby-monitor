import os
import tensorflow as tf

import config
from data import get_datasets
from model import build_classifier_head


def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    train_ds, val_ds, test_ds, class_names, input_shape = get_datasets()
    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Input shape: {input_shape}")

    with open(config.LABELS_PATH, "w") as f:
        f.write("\n".join(class_names))

    model = build_classifier_head(config.EMBEDDING_DIM, num_classes=len(class_names))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.LEARNING_RATE),
        # label smoothing keeps the softmax from saturating near 1.0 on every
        # prediction, so max-probability stays a meaningful confidence signal
        # for threshold-based rejection at inference time
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=config.LABEL_SMOOTHING),
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            config.MODEL_PATH, monitor="val_accuracy", save_best_only=True
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=10, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5
        ),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS,
        callbacks=callbacks,
    )

    test_loss, test_acc = model.evaluate(test_ds)
    print(f"Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
