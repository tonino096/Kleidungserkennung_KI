from __future__ import annotations

from PIL import ImageDraw
from PIL.Image import Image

from src.detector.base import Prediction


def draw_predictions(image: Image, predictions: list[Prediction]) -> Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)

    for pred in predictions:
        x1, y1, x2, y2 = pred["bbox"]
        label = pred["label"]
        confidence = pred["confidence"]
        text = f"{label} ({confidence:.2f})"

        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, max(0, y1 - 14)), text, fill="red")

    return output
