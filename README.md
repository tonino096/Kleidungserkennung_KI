# KI-Web-App zur Erkennung von Kleidungsstücken

Wir haben eine KI trainiert, die eine Streamlit-App und eine OpenCV-Live-Demo als Ausgabe hat. Sie erkennt die Kleidungsstücke in Bildern, Videos und Webcam-Aufnahmen mit YOLO, wertet sie aus und visualisiert die Ergebnisse verständlich. 

## Features

- Streamlit-Weboberfläche für Einzelbildtests
- Live-Webcam- oder Video-Demo im OpenCV-Fenster
- DeepFashion2 zu YOLO-Label Konvertierung ohne Bildkopien
- Einzel- und Batch-Matchtests mit Report-Bild und HTML-Übersicht

## Projektstruktur

```text
app/streamlit_app.py              Streamlit-UI für Bildtests und Matchtests
app/webcam_demo.py                Live-Demo für Webcam oder Videodatei
src/config/settings.py            zentrale Einstellungen und Projektpfade
src/detector/base.py              gemeinsames Vorhersageformat
src/detector/mock_detector.py     einfacher Demo-Detector ohne KI-Modell
src/detector/yolo_detector.py     YOLO-Detector mit Ultralytics
src/inference/pipeline.py         verbindet Settings und Detector
src/testing/matchtest.py          Matchtest- und Batch-Report-Logik
src/training/deepfashion2_yolo.py DeepFashion2-zu-YOLO-Labels
src/visualization/draw.py         Bounding-Boxes und Report-Visualisierung
main.py                           optionale CLI für Konvertierung und Matchtests
requirements.txt                  Python-Abhängigkeiten
```
> [!IMPORTANT]
> `main.py` muss nicht zwingend ausgeführt werden. Für die normale Nutzung reicht es, der Anleitung im README zu folgen, also Abhängigkeiten installieren und dann die Streamlit-App oder die Webcam-Demo starten. `main.py` ist nur für Zusatzaufgaben wie bspw. DeepFashion2-Konvertierung gedacht.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## App starten

```bash
streamlit run app/streamlit_app.py
```

## Live-Webcam-Demo

```bash
set YOLO_WEIGHTS_PATH=models/best.pt
python app/webcam_demo.py
```

Optional:

```bash
python app/webcam_demo.py --source 1
python app/webcam_demo.py --weights "R:/Schulprojekt KI/Kleidungserkennung_KI/models/best.pt"
python app/webcam_demo.py --source "video.mp4"
```

> [!NOTE]
> - `q` oder `Esc` beendet das Fenster.
> - Falls die Standard-Webcam nicht gefunden wird, teste `--source 1` oder `--source 2`.


## DeepFashion2 zu YOLO konvertieren
```
python main.py convert-deepfashion2-to-yolo --dataset-root "R:\Schulprojekt KI\DeepFashion2" --output-root data/deepfashion2_yolo_labels
```

## YOLO trainieren

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
model.train(data="data/deepfashion2_yolo_labels/data.yaml", epochs=30, imgsz=640)
```

## Matchtest

Einzelbild:

```bash
python main.py matchtest-deepfashion2 --dataset-root "R:\Schulprojekt KI\DeepFashion2" --weights-path "models/best.pt" --split validation --annotation-stem 000001 --output-root outputs/matchtests
```

Batch:

```bash
python main.py batch-matchtest-deepfashion2 --dataset-root "R:\Schulprojekt KI\DeepFashion2" --weights-path "models/best.pt" --split validation --limit 12 --output-root outputs/matchtests
```

## Konfiguration

- `DETECTOR_TYPE` Standard: `mock`
- `YOLO_WEIGHTS_PATH` Standard: `models/best.pt`
- `DEEPFASHION2_ROOT` Standard: `data/deepfashion2`
