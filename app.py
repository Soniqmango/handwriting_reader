import tkinter as tk
from PIL import Image, ImageDraw
from predict import load_model, predict
from utils.preprocessing import segment_characters

CANVAS_W = 560
CANVAS_H = 280
BRUSH_SIZE = 16


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Character Recognizer")
        self.root.resizable(False, False)

        self.model = None

        self.pil_image = Image.new("L", (CANVAS_W, CANVAS_H), 0)
        self.draw = ImageDraw.Draw(self.pil_image)

        self.canvas = tk.Canvas(
            root, width=CANVAS_W, height=CANVAS_H,
            bg="black", cursor="cross"
        )
        self.canvas.grid(row=0, column=0, columnspan=2, padx=10, pady=10)
        self.canvas.bind("<B1-Motion>", self.paint)

        predict_btn = tk.Button(root, text="Predict", command=self.do_predict, width=12)
        predict_btn.grid(row=1, column=0, pady=10)

        clear_btn = tk.Button(root, text="Clear", command=self.clear, width=12)
        clear_btn.grid(row=1, column=1, pady=10)

        self.result = tk.Label(root, text="Write a word then press Predict", font=("Helvetica", 22))
        self.result.grid(row=2, column=0, columnspan=2, pady=10)

    def paint(self, event):
        x, y = event.x, event.y
        r = BRUSH_SIZE
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="white")
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill=255)

    def do_predict(self):
        if self.model is None:
            self.result.config(text="Loading model...")
            self.root.update()
            self.model = load_model()

        chars = segment_characters(self.pil_image)

        if not chars:
            self.result.config(text="Nothing detected")
            return

        word = ""
        for char_arr in chars:
            label, _ = predict(self.model, char_arr)
            word += label

        self.result.config(text=word)

    def clear(self):
        self.canvas.delete("all")
        self.pil_image = Image.new("L", (CANVAS_W, CANVAS_H), 0)
        self.draw = ImageDraw.Draw(self.pil_image)
        self.result.config(text="Write a word then press Predict")


def main():
    root = tk.Tk()
    root.lift()
    root.focus_force()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
