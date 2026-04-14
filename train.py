import os
import tensorflow as tf


def load_and_preprocess_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    return (x_train, y_train), (x_test, y_test)


def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),

        tf.keras.layers.Conv2D(32, kernel_size=3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(pool_size=2),

        tf.keras.layers.Conv2D(64, kernel_size=3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(pool_size=2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def main():
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    print(f"  Train samples: {len(x_train)}, Test samples: {len(x_test)}")

    model = build_model()
    model.summary()

    print("\nTraining...")
    model.fit(
        x_train,
        y_train,
        epochs=10,
        batch_size=128,
        validation_split=0.1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
        ],
    )

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest accuracy: {accuracy * 100:.2f}%")

    os.makedirs("model", exist_ok=True)
    save_path = os.path.join("model", "digit_model.keras")
    model.save(save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
