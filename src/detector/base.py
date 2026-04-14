from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypedDict

from PIL.Image import Image


class Prediction(TypedDict):
    label: str
    confidence: float
    bbox: list[int]


class BaseDetector(ABC):
    @abstractmethod
    def predict(self, image: Image) -> list[Prediction]:
        """Return a list of predictions in a shared format."""
