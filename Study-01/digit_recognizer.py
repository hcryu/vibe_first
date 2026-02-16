"""
Handwritten Digit Recognition Application
Uses a CNN trained on MNIST dataset with a Tkinter drawing canvas.
"""

import os
import numpy as np
from PIL import Image, ImageDraw

# Suppress TensorFlow info logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from keras import layers
import tkinter as tk

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mnist_model.keras")


def build_and_train_model():
    """Train a CNN on the MNIST dataset and save it."""
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normalize and reshape: (samples, 28, 28, 1)
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, kernel_size=3, activation="relu"),
        layers.MaxPooling2D(pool_size=2),
        layers.Conv2D(64, kernel_size=3, activation="relu"),
        layers.MaxPooling2D(pool_size=2),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("Training model...")
    model.fit(x_train, y_train, epochs=3, batch_size=128, validation_split=0.1, verbose=1)

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {accuracy:.4f}")

    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    return model


def load_model():
    """Load existing model or train a new one."""
    if os.path.exists(MODEL_PATH):
        print("Loading saved model...")
        return keras.models.load_model(MODEL_PATH)
    else:
        return build_and_train_model()


class DigitRecognizerApp:
    """Tkinter GUI for drawing digits and recognizing them."""

    CANVAS_SIZE = 280  # 280x280 pixels (10x scale of 28x28)
    BRUSH_RADIUS = 8

    def __init__(self, model):
        self.model = model
        self.root = tk.Tk()
        self.root.title("Handwritten Digit Recognizer")
        self.root.resizable(False, False)

        # Drawing state tracked via PIL (Tkinter canvas has no pixel export)
        self.pil_image = Image.new("L", (self.CANVAS_SIZE, self.CANVAS_SIZE), 0)
        self.pil_draw = ImageDraw.Draw(self.pil_image)

        self._build_ui()

    def _build_ui(self):
        # Canvas for drawing
        self.canvas = tk.Canvas(
            self.root,
            width=self.CANVAS_SIZE,
            height=self.CANVAS_SIZE,
            bg="black",
            cursor="cross",
        )
        self.canvas.pack(padx=10, pady=10)

        # Mouse events
        self.canvas.bind("<B1-Motion>", self._paint)
        self.canvas.bind("<Button-1>", self._paint)

        # Buttons
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=5)

        recognize_btn = tk.Button(
            btn_frame, text="Recognize", command=self._recognize, width=12
        )
        recognize_btn.pack(side=tk.LEFT, padx=5)

        clear_btn = tk.Button(
            btn_frame, text="Clear", command=self._clear, width=12
        )
        clear_btn.pack(side=tk.LEFT, padx=5)

        # Result label
        self.result_label = tk.Label(
            self.root, text="Draw a digit and click Recognize", font=("Arial", 16)
        )
        self.result_label.pack(pady=10)

    def _paint(self, event):
        r = self.BRUSH_RADIUS
        x, y = event.x, event.y
        # Draw on Tkinter canvas (visual)
        self.canvas.create_oval(
            x - r, y - r, x + r, y + r, fill="white", outline="white"
        )
        # Draw on PIL image (for prediction)
        self.pil_draw.ellipse([x - r, y - r, x + r, y + r], fill=255)

    def _clear(self):
        self.canvas.delete("all")
        self.pil_image = Image.new("L", (self.CANVAS_SIZE, self.CANVAS_SIZE), 0)
        self.pil_draw = ImageDraw.Draw(self.pil_image)
        self.result_label.config(text="Draw a digit and click Recognize")

    def _recognize(self):
        # Resize to 28x28 with high-quality downsampling
        img = self.pil_image.resize((28, 28), Image.LANCZOS)
        img_array = np.array(img).astype("float32") / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        prediction = self.model.predict(img_array, verbose=0)
        digit = np.argmax(prediction)
        confidence = prediction[0][digit] * 100

        self.result_label.config(text=f"Prediction: {digit}   (Confidence: {confidence:.1f}%)")

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    model = load_model()
    app = DigitRecognizerApp(model)
    app.run()
