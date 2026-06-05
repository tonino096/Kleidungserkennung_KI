from __future__ import annotations

from PIL.Image import Image

from src.config.settings import Settings
from src.detector.base import BaseDetector, Prediction
from src.detector.mock_detector import MockDetector
from src.detector.yolo_detector import YoloDetector

# Die Pipeline ist die kleine Vermittlerin zwischen App und Detector. Die App
# fragt nur die Pipeline, und die Pipeline entscheidet, welcher Detector läuft.


class InferencePipeline:
    def __init__(
        self,
        settings: Settings,
        detector_type: str | None = None,
        detector: BaseDetector | None = None,
    ) -> None:
        self.settings = settings
        # Falls von außen bereits ein Detector übergeben wird, nutzt die Pipeline
        # diesen direkt. Das ist besonders für Tests und Sonderfälle praktisch.
        self.detector = detector or self._build_detector(detector_type or settings.detector_type)

    def _build_detector(self, detector_type: str) -> BaseDetector:
        # Die Pipeline ist bewusst einfach gehalten: ein String entscheidet,
        # welche konkrete Detector-Implementierung verwendet wird.
        if detector_type == "mock":
            return MockDetector()
        if detector_type == "yolo":
            return YoloDetector(weights_path=self.settings.yolo_weights_path)
        raise ValueError(f"Unbekannter detector_type: {detector_type}")

    def predict(self, image: Image) -> list[Prediction]:
        # Der eigentliche Inferenzaufruf wird komplett an den ausgewählten
        # Detector delegiert, damit die Pipeline schlank bleibt.
        return self.detector.predict(image)
