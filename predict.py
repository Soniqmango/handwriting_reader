import os
import sys

import numpy as np
import tensorflow as tf

from utils.preprocessing import preprocess_image


MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "digit_model.keras")


def load_model(model_path=MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No model found at '{model_path}'.\n"
            "Run  python train.py  first."
        )
    return tf.keras.models.load_model(model_path)


def predict(model, image_array):
    probabilities = model.predict(image_array, verbose=0)[0]
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
