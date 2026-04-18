import cv2
import numpy as np


def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    # blur to reduce noise, then threshold to get white digit on black background
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    # find the digit and crop to its bounding box
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        x, y, w, h = cv2.boundingRect(np.vstack(contours))

        # add padding so the digit isn't right at the edge
        pad = 20
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(img.shape[1] - x, w + 2 * pad)
        h = min(img.shape[0] - y, h + 2 * pad)

        img = img[y:y + h, x:x + w]

    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    img = img.astype("float32") / 255.0

    return img.reshape(1, 28, 28, 1)


def segment_characters(pil_image):
    img = np.array(pil_image)

    # find contours on the white-on-black canvas image
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # filter out tiny specks
    contours = [c for c in contours if cv2.contourArea(c) > 50]

    if not contours:
        return []

    # get bounding boxes sorted left to right
    boxes = sorted([cv2.boundingRect(c) for c in contours], key=lambda b: b[0])

    # merge boxes that are close horizontally — they belong to the same character
    # (e.g. the dot on 'i', or two strokes of the same letter drawn separately)
    merged = [list(boxes[0])]
    for x, y, w, h in boxes[1:]:
        px, py, pw, ph = merged[-1]
        if x <= px + pw + 20:
            merged[-1] = [
                min(px, x),
                min(py, y),
                max(px + pw, x + w) - min(px, x),
                max(py + ph, y + h) - min(py, y),
            ]
        else:
            merged.append([x, y, w, h])

    # crop each character, pad, and resize to 28x28
    chars = []
    for x, y, w, h in merged:
        pad = 8
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img.shape[1], x + w + pad)
        y2 = min(img.shape[0], y + h + pad)

        char_img = img[y1:y2, x1:x2]
        char_img = cv2.resize(char_img, (28, 28), interpolation=cv2.INTER_AREA)
        char_img = char_img.astype("float32") / 255.0
        chars.append(char_img.reshape(1, 28, 28, 1))

    return chars
