"""
deep_learning.py
------------------
Module 3: 1D-CNN classifier over one-hot encoded DNA sequences, for
comparison against the classical ML models (CO4 + syllabus's "intro to
deep learning libraries").

TensorFlow/Keras is imported lazily so the rest of the pipeline
(Modules 1, 2, 4) still works in environments where TF is not
installed. Install it with:  pip install tensorflow --break-system-packages
"""

import numpy as np


def _import_tf():
    try:
        import tensorflow as tf
        return tf
    except ImportError as e:
        raise ImportError(
            "TensorFlow is not installed. Run:\n"
            "  pip install tensorflow --break-system-packages\n"
            "to enable the deep learning comparison module."
        ) from e


def build_cnn(input_len: int, n_classes: int):
    tf = _import_tf()
    from tensorflow.keras import layers, models

    model = models.Sequential([
        layers.Input(shape=(input_len, 4)),
        layers.Conv1D(32, kernel_size=6, activation="relu"),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(64, kernel_size=6, activation="relu"),
        layers.GlobalMaxPooling1D(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def train_cnn(X_train, y_train, X_test, y_test, n_classes: int, epochs: int = 15, batch_size: int = 16):
    """
    X_train/X_test: one-hot arrays from features.one_hot_encode_sequences()
    y_train/y_test: integer-encoded labels (use the same LabelEncoder as
                    the classical ML module for consistent comparison)
    """
    tf = _import_tf()
    model = build_cnn(X_train.shape[1], n_classes)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    return {
        "model": model,
        "history": history.history,
        "test_accuracy": round(float(test_acc), 4),
        "test_loss": round(float(test_loss), 4),
    }
