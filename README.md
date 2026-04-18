# Handwriting Reader

A Python machine learning project that recognises handwritten digits and uppercase letters. Draw directly on the canvas or feed in an image file.

## Features

- Recognises digits (0–9) and uppercase letters (A–Z)
- Interactive drawing canvas — write a full word and it predicts each character
- CLI prediction from any image file
- CNN trained on the EMNIST dataset with data augmentation and batch normalisation

## Project Structure

```
handwriting_reader/
├── train.py              # Download EMNIST, train the model, save it
├── predict.py            # CLI: predict a character from an image file
├── app.py                # Tkinter drawing canvas GUI
├── utils/
│   └── preprocessing.py  # Image preprocessing and character segmentation
└── requirements.txt
```

> `data/` and `model/` are excluded from git. Run `train.py` to generate both locally.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### 1. Train the model

```bash
python train.py
```

Downloads EMNIST (~535MB, cached after first run), trains the CNN, and saves the model to `model/char_model.keras`.

### 2. Predict from an image

```bash
python predict.py path/to/image.png
```

Outputs the predicted character and confidence score.

### 3. Drawing canvas

```bash
python app.py
```

Opens a canvas where you can write a word with your mouse. Press **Predict** to recognise each character, **Clear** to reset.

![Canvas demo placeholder](https://placehold.co/560x280/000000/FFFFFF?text=Draw+here)

## Model

| Property | Value |
|---|---|
| Dataset | EMNIST balanced (digits + uppercase, 36 classes) |
| Architecture | 3× Conv2D + BatchNorm + MaxPool → Dense(256) → Dropout |
| Augmentation | Random rotation, translation, zoom |
| Optimiser | Adam with ReduceLROnPlateau |

## Tech Stack

- [TensorFlow / Keras](https://www.tensorflow.org/)
- [OpenCV](https://opencv.org/)
- [Pillow](https://python-pillow.org/)
- [Tkinter](https://docs.python.org/3/library/tkinter.html) (built into Python)
- [NumPy](https://numpy.org/)
