import os
import tensorflow as tf
import tensorflow_datasets as tfds

# EMNIST balanced: 10 digits + 26 uppercase + 11 distinct lowercase = 47 classes
LABEL_NAMES = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt'
NUM_CLASSES = len(LABEL_NAMES)

augment = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
    tf.keras.layers.RandomZoom(0.1),
])


def prepare(image, label, training=False):
    image = tf.cast(image, tf.float32) / 255.0
    # EMNIST images are stored rotated relative to MNIST — fix that
    image = tf.transpose(image, perm=[1, 0, 2])
    if training:
        image = augment(image, training=True)
    return image, label


def load_data():
    print("Loading EMNIST balanced dataset...")
    ds_train, ds_test = tfds.load(
        'emnist/balanced',
        split=['train', 'test'],
        as_supervised=True,
    )

    ds_train = (ds_train
                .map(lambda x, y: prepare(x, y, training=True), num_parallel_calls=tf.data.AUTOTUNE)
                .cache()
                .shuffle(10000)
                .batch(128)
                .prefetch(tf.data.AUTOTUNE))

    ds_test = (ds_test
               .map(lambda x, y: prepare(x, y, training=False), num_parallel_calls=tf.data.AUTOTUNE)
               .cache()
               .batch(128)
               .prefetch(tf.data.AUTOTUNE))

    return ds_train, ds_test


def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),

        tf.keras.layers.Conv2D(32, 3, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(2),

        tf.keras.layers.Conv2D(64, 3, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(2),

        tf.keras.layers.Conv2D(128, 3, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax'),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    return model


def main():
    ds_train, ds_test = load_data()

    model = build_model()
    model.summary()

    print("\nTraining...")
    model.fit(
        ds_train,
        epochs=20,
        validation_data=ds_test,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)
        ],
    )

    _, accuracy = model.evaluate(ds_test, verbose=0)
    print(f"\nTest accuracy: {accuracy * 100:.2f}%")

    os.makedirs("model", exist_ok=True)
    model.save("model/char_model.keras")
    print("Model saved to model/char_model.keras")


if __name__ == "__main__":
    main()
