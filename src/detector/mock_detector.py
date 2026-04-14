from __future__ import annotations

from PIL.Image import Image

from src.detector.base import BaseDetector, Prediction


class MockDetector(BaseDetector):
    def predict(self, image: Image) -> list[Prediction]:
        width, height = image.size
        x1 = int(width * 0.2)
        y1 = int(height * 0.2)
        x2 = int(width * 0.8)
        y2 = int(height * 0.8)

        return [
            {
                "label": "top",
                "confidence": 0.93,
                "bbox": [x1, y1, x2, y2],
            }
        ]
