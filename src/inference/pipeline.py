from __future__ import annotations

from PIL.Image import Image

from src.config.settings import Settings
from src.detector.base import BaseDetector, Prediction
from src.detector.mock_detector import MockDetector
from src.detector.yolo_detector import YoloDetector


class InferencePipeline:
    def __init__(
        self,
        settings: Settings,
        detector_type: str | None = None,
        detector: BaseDetector | None = None,
    ) -> None:
        self.settings = settings
        self.detector = detector or self._build_detector(detector_type or settings.detector_type)

    def _build_detector(self, detector_type: str) -> BaseDetector:
        if detector_type == "mock":
            return MockDetector()
        if detector_type == "yolo":
            return YoloDetector(weights_path=self.settings.yolo_weights_path)
        raise ValueError(f"Unbekannter detector_type: {detector_type}")

    def predict(self, image: Image) -> list[Prediction]:
        return self.detector.predict(image)
