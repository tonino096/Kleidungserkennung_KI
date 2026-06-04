from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image as PilImage
from PIL.Image import Image

from src.config.settings import resolve_project_path
from src.detector.base import BaseDetector, Prediction


class YoloDetector(BaseDetector):
    def __init__(self, weights_path: str | None = None, model: Any | None = None) -> None:
        # Der Gewichtspfad wird direkt beim Erzeugen in einen absoluten Pfad
        # umgewandelt, damit spätere Aufrufe unabhängig vom Startordner sind.
        self.weights_path = resolve_project_path(weights_path) if weights_path else None
        self.model = model

        # In Tests kann ein vorbereitetes Fake-Modell übergeben werden.
        # Im Normalbetrieb wird das Ultralytics-Modell hier einmalig geladen.
        if self.model is None:
            if self.weights_path is None:
                raise ValueError("Kein YOLO-Gewichtspfad angegeben.")
            if not self.weights_path.exists():
                raise FileNotFoundError(f"YOLO-Gewichte nicht gefunden: {self.weights_path}")
            self.model = self._load_model()

    def predict(self, image: Image) -> list[Prediction]:
        # Für den normalen Projektfluss reicht die reine Vorhersageliste.
        results = self._predict_results(image)
        return self._map_predictions(results)

    def predict_and_render(self, image: Image) -> tuple[list[Prediction], Image]:
        # Diese Variante wird genutzt, wenn zusätzlich exakt das von
        # Ultralytics gezeichnete Ergebnisbild benötigt wird.
        results = self._predict_results(image)
        return self._map_predictions(results), self._render_results(results)

    def _predict_results(self, image: Image) -> list[Any]:
        # Alle YOLO-Inferenzaufrufe laufen durch diese Hilfsfunktion, damit
        # Mapping und Rendering auf derselben Ergebnisbasis arbeiten.
        return self.model.predict(image, verbose=False)

    def _map_predictions(self, results: list[Any]) -> list[Prediction]:
        # Ultralytics liefert ein komplexes Ergebnisobjekt; hier reduzieren wir
        # es auf das einfache Projektformat mit Label, Konfidenz und Bounding Box.
        predictions: list[Prediction] = []

        for result in results:
            names = result.names
            for box in result.boxes:
                class_id = int(box.cls.item())
                confidence = float(box.conf.item())
                x1, y1, x2, y2 = [int(round(value)) for value in box.xyxy[0].tolist()]
                predictions.append(
                    {
                        "label": self._resolve_label(names, class_id),
                        "confidence": confidence,
                        "bbox": [x1, y1, x2, y2],
                    }
                )

        return predictions

    @staticmethod
    def _render_results(results: list[Any]) -> Image:
        # result.plot() erzeugt ein OpenCV-/NumPy-Bild im BGR-Format.
        # Für den Rest des Projekts wird es wieder in ein PIL-RGB-Bild gewandelt.
        if not results:
            raise ValueError("YOLO hat keine Ergebnisse zum Rendern geliefert.")

        plotted = results[0].plot()
        return PilImage.fromarray(plotted[..., ::-1])

    def _load_model(self) -> Any:
        # Die Bibliothek wird erst hier importiert, damit reine Struktur- oder
        # Mock-Tests nicht unnötig von einer installierten GPU-Umgebung abhängen.
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise ImportError(
                "Ultralytics ist nicht installiert. Bitte 'pip install -r requirements.txt' ausführen."
            ) from error

        return YOLO(str(self.weights_path))

    @staticmethod
    def _resolve_label(names: Any, class_id: int) -> str:
        # Je nach Ultralytics-Version liegen Klassennamen als Dict oder Liste vor.
        # Diese Hilfsfunktion kapselt beide Fälle.
        if isinstance(names, dict):
            return str(names.get(class_id, class_id))
        if isinstance(names, list):
            return str(names[class_id])
        return str(class_id)
