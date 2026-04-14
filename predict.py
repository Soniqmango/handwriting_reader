import os
import sys

import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf


MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "digit_model.keras")


def preprocess_image(image_path: str) -> np.ndarray:
    """
    Loads an image from disk and transforms it to match the model's expected
    input: shape (1, 28, 28, 1), dtype float32, values in [0.0, 1.0].

    Key step — inversion:
      MNIST images have WHITE digits on a BLACK background.
      Photographs of handwritten digits are the opposite (black ink, white paper).
      Inverting the pixel values corrects this mismatch. Step 2 will replace this
      with adaptive thresholding via OpenCV for more robust real-world handling.
    """
    img = Image.open(image_path).convert("L")   # grayscale
    img = ImageOps.invert(img)                  # black-on-white -> white-on-black
    img = img.resize((28, 28), Image.LANCZOS)   # match MNIST resolution

    arr = np.array(img, dtype="float32") / 255.0  # normalize to [0, 1]
    arr = arr.reshape(1, 28, 28, 1)                # batch + channel dims

    return arr


def load_model(model_path: str = MODEL_PATH) -> tf.keras.Model:
    """
    Loads the saved Keras model from disk.
    Raises a clear error if the model has not been trained yet.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No model found at '{model_path}'.\n"
            "Run  python train.py  first to train and save the model."
        )
    return tf.keras.models.load_model(model_path)


def predict(model: tf.keras.Model, image_array: np.ndarray) -> tuple[int, float]:
    """
    Runs the model on a preprocessed image array.
    Returns (predicted_digit, confidence) where confidence is in [0.0, 1.0].
    """
    probabilities = model.predict(image_array, verbose=0)[0]  # shape: (10,)
    digit = int(np.argmax(probabilities))
    confidence = float(probabilities[digit])
    return digit, confidence


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path-to-image>")
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(f"Error: image file '{image_path}' not found.")
        sys.exit(1)

    image_array = preprocess_image(image_path)
    model = load_model()
    digit, confidence = predict(model, image_array)

    print(f"Predicted digit: {digit}  (confidence: {confidence * 100:.1f}%)")


if __name__ == "__main__":
    main()
