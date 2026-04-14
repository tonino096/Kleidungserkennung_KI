from __future__ import annotations

from pathlib import Path

from PIL.Image import Image

from src.detector.base import BaseDetector, Prediction


class YoloDetector(BaseDetector):
    def __init__(self, weights_path: str | None = None) -> None:
        self.weights_path = Path(weights_path) if weights_path else None
        # TODO: Später ultralytics.YOLO laden und Modell hier initialisieren.

    def predict(self, image: Image) -> list[Prediction]:
        # TODO: Bild durch YOLOv8n laufen lassen und ins Prediction-Format mappen.
        if self.weights_path and not self.weights_path.exists():
            raise FileNotFoundError(f"YOLO-Gewichte nicht gefunden: {self.weights_path}")
        return []
