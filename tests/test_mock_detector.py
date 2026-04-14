from __future__ import annotations

from PIL import Image

from src.detector.mock_detector import MockDetector


def test_mock_detector_returns_expected_format() -> None:
    detector = MockDetector()
    image = Image.new("RGB", (200, 100), color="white")

    predictions = detector.predict(image)

    assert len(predictions) == 1
    pred = predictions[0]
    assert set(pred.keys()) == {"label", "confidence", "bbox"}
    assert pred["label"] == "top"
    assert isinstance(pred["confidence"], float)
    assert len(pred["bbox"]) == 4
    assert all(isinstance(value, int) for value in pred["bbox"])
