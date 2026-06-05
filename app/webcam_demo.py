from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# Auch dieses Hilfsskript kann direkt aus dem app-Ordner gestartet werden.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import Settings
from src.inference.pipeline import InferencePipeline
from src.visualization.draw import draw_predictions

# Diese Datei ist die Live-Version der Demo. Statt ein Bild hochzuladen, liest
# sie ständig neue Frames aus Webcam oder Video und zeigt sofort die Boxen an.


def parse_args() -> argparse.Namespace:
    # Alle wichtigen Live-Demo-Einstellungen werden über Argumente steuerbar,
    # damit Webcam, Videodatei und alternative Gewichte leicht testbar sind.
    parser = argparse.ArgumentParser(description="Live-Kleidungs-Erkennung per Webcam.")
    parser.add_argument("--source", default="0", help="Webcam-Index oder Video-Pfad, Standard: 0")
    parser.add_argument("--weights", default=None, help="Optionaler Pfad zu YOLO-Gewichten")
    parser.add_argument("--model-name", default=None, help="Name aus der 3er-Modellliste, z. B. modell_1")
    parser.add_argument("--window", default="Kleidungs-Erkennung Live", help="Fenstertitel")
    return parser.parse_args()


def resolve_source(raw_source: str) -> int | str:
    # "0" soll als Webcam-Index 0 verstanden werden, nicht als Dateiname.
    return int(raw_source) if raw_source.isdigit() else raw_source


def attempt_open_capture(source: int | str) -> tuple[cv2.VideoCapture, str]:
    # OpenCV verhält sich auf Windows je nach Kamera und Backend unterschiedlich.
    # Deshalb werden mehrere sinnvolle Kombinationen automatisch durchprobiert.
    attempts: list[tuple[int | str, int | None, str]] = []

    if isinstance(source, int):
        for candidate_source in range(source, source + 4):
            attempts.extend(
                [
                    (candidate_source, cv2.CAP_MSMF, f"index={candidate_source}, backend=MSMF"),
                    (candidate_source, cv2.CAP_DSHOW, f"index={candidate_source}, backend=DSHOW"),
                    (candidate_source, None, f"index={candidate_source}, backend=default"),
                ]
            )
    else:
        attempts.extend(
            [
                (source, None, f"path={source}, backend=default"),
                (source, cv2.CAP_FFMPEG, f"path={source}, backend=FFMPEG"),
            ]
        )

    for candidate_source, backend, description in attempts:
        capture = cv2.VideoCapture(candidate_source, backend) if backend is not None else cv2.VideoCapture(candidate_source)
        if capture.isOpened():
            return capture, description
        capture.release()

    # Die Fehlermeldung listet bewusst alle getesteten Varianten auf, damit
    # bei Kamera-Problemen schneller klar ist, was bereits versucht wurde.
    tried = ", ".join(description for _, _, description in attempts)
    raise RuntimeError(
        "Videoquelle konnte nicht geöffnet werden. "
        f"Getestet wurden: {tried}. "
        "Bitte prüfe Kamera-Zugriff in Windows oder starte mit --source 1 bzw. --source 2."
    )


def build_pipeline(weights_override: str | None, model_name: str | None) -> InferencePipeline:
    # Die Live-Demo läuft immer mit YOLO; optional kann dafür ein anderer
    # Gewichtspfad als der Standard gesetzt werden.
    settings = Settings()
    if model_name:
        settings.yolo_weights_path = settings.get_yolo_weights_path(model_name)
    if weights_override:
        settings.yolo_weights_path = weights_override
    return InferencePipeline(settings=settings, detector_type="yolo")


def main() -> None:
    # Der Hauptloop liest fortlaufend Frames ein, wandelt sie in PIL um,
    # lässt Vorhersagen erzeugen und zeigt das Ergebnis in OpenCV an.
    args = parse_args()
    pipeline = build_pipeline(args.weights, args.model_name)
    source = resolve_source(args.source)

    capture, opened_with = attempt_open_capture(source)

    cv2.namedWindow(args.window, cv2.WINDOW_NORMAL)
    previous_time = time.perf_counter()

    try:
        while True:
            success, frame_bgr = capture.read()
            if not success:
                # Bei Videos ist das normalerweise einfach das Dateiende; bei
                # Webcams bedeutet es meistens, dass die Kamera nicht mehr liefert.
                break

            # Das Projekt zeichnet auf PIL-Bildern, OpenCV liefert aber BGR-Frames.
            # Deshalb wird jedes Frame für die Inferenz einmal konvertiert.
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            predictions = pipeline.predict(image)
            annotated_image = draw_predictions(image, predictions)
            annotated_bgr = cv2.cvtColor(
                np.array(annotated_image.convert("RGB")),
                cv2.COLOR_RGB2BGR,
            )

            current_time = time.perf_counter()
            fps = 1.0 / max(current_time - previous_time, 1e-6)
            previous_time = current_time

            # Die FPS-Anzeige hilft direkt zu sehen, wie flüssig das Modell
            # auf der aktuellen Hardware läuft.
            cv2.putText(
                annotated_bgr,
                f"FPS: {fps:.1f} | {opened_with} | q = beenden",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(args.window, annotated_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
