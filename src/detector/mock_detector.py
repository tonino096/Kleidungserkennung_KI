from __future__ import annotations

from PIL.Image import Image

from src.detector.base import BaseDetector, Prediction


class MockDetector(BaseDetector):
    def predict(self, image: Image) -> list[Prediction]:
        # Der Mock-Detector liefert absichtlich immer eine einfache,
        # reproduzierbare Beispielbox, damit die Oberfläche auch ohne
        # trainiertes Modell demonstrierbar bleibt.
        width, height = image.size
        return [
            {
                "label": "top",
                "confidence": 0.93,
                "bbox": [
                    int(width * 0.2),
                    int(height * 0.2),
                    int(width * 0.8),
                    int(height * 0.8),
                ],
            }
        ]
