from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypedDict

from PIL.Image import Image


class Prediction(TypedDict):
    # Dieses gemeinsame Vorhersageformat sorgt dafür, dass Mock-, YOLO-,
    # Streamlit- und Matchtest-Code dieselbe Datenstruktur erwarten.
    label: str
    confidence: float
    bbox: list[int]


class BaseDetector(ABC):
    @abstractmethod
    def predict(self, image: Image) -> list[Prediction]:
        # Jeder Detector muss ein PIL-Bild entgegennehmen und eine Liste von
        # Vorhersagen im gemeinsamen Prediction-Format zurückgeben.
        """Return a list of predictions in a shared format."""
