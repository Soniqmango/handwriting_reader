import gzip
import os
import ssl
import struct
import urllib.request
import zipfile

import certifi

import numpy as np
import tensorflow as tf

EMNIST_URL = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

LABEL_NAMES = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt'
NUM_CLASSES = len(LABEL_NAMES)


def _progress(count, block_size, total):
    pct = min(count * block_size / total * 100, 100)
    print(f"\r  {pct:.1f}%", end="", flush=True)


def _read_idx(path):
    with gzip.open(path, 'rb') as f:
        magic, = struct.unpack('>I', f.read(4))
        count, = struct.unpack('>I', f.read(4))
        if magic == 0x00000803:  # images
            rows, = struct.unpack('>I', f.read(4))
            cols, = struct.unpack('>I', f.read(4))
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(count, rows, cols)
        else:  # labels
            return np.frombuffer(f.read(), dtype=np.uint8)


def load_data():
    zip_path = os.path.join(DATA_DIR, "emnist.zip")
    extract_dir = os.path.join(DATA_DIR, "emnist")

    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(zip_path):
        print("Downloading EMNIST (~535MB)...")
        ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        req = urllib.request.Request(EMNIST_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=ssl_ctx) as response, \
             open(zip_path, 'wb') as out:
            total = int(response.headers.get('Content-Length', 0))
            downloaded = 0
            while chunk := response.read(65536):
                out.write(chunk)
                downloaded += len(chunk)
                if total:
                    _progress(downloaded, 1, total)
        print()

    if not os.path.exists(extract_dir):
        print("Extracting balanced split...")
        with zipfile.ZipFile(zip_path) as zf:
            for name in zf.namelist():
                if 'balanced' in name.lower():
                    zf.extract(name, extract_dir)

    gzip_dir = os.path.join(extract_dir, "gzip")

    def load_split(split):
        images = _read_idx(os.path.join(gzip_dir, f"emnist-balanced-{split}-images-idx3-ubyte.gz"))
        labels = _read_idx(os.path.join(gzip_dir, f"emnist-balanced-{split}-labels-idx1-ubyte.gz"))
        # EMNIST images are stored transposed relative to MNIST
        images = np.transpose(images, (0, 2, 1))
        return images.astype("float32") / 255.0, labels

    print("Loading EMNIST balanced...")
    x_train, y_train = load_split("train")
    x_test, y_test = load_split("test")
    print(f"  Train: {len(x_train)}, Test: {len(x_test)}")

    x_train = x_train[..., np.newaxis]
    x_test = x_test[..., np.newaxis]

    return (x_train, y_train), (x_test, y_test)


def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),

        # augmentation — only active during training
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomTranslation(0.1, 0.1),
        tf.keras.layers.RandomZoom(0.1),

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
    (x_train, y_train), (x_test, y_test) = load_data()

    model = build_model()
    model.summary()

    print("\nTraining...")
    model.fit(
        x_train, y_train,
        epochs=20,
        batch_size=128,
        validation_data=(x_test, y_test),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)
        ],
    )

    _, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest accuracy: {accuracy * 100:.2f}%")

    os.makedirs("model", exist_ok=True)
    model.save("model/char_model.keras")
    print("Model saved to model/char_model.keras")


if __name__ == "__main__":
    main()
