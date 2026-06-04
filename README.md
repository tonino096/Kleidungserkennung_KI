# KI-Web-App zur Erkennung von Kleidungsstücken

Dieses Repository ist bewusst kompakt gehalten: eine Streamlit-App, eine OpenCV-Live-Demo und vier `src`-Dateien für YOLO-Inferenz, DeepFashion2-Konvertierung, Matchtests und Visualisierung.

## Features

- Streamlit-Weboberfläche für Einzelbildtests
- Live-Webcam- oder Video-Demo im OpenCV-Fenster
- YOLO- oder Mock-Inferenz über dieselbe Pipeline
- DeepFashion2 -> YOLO Label-Konvertierung ohne Bildkopien
- Einzel- und Batch-Matchtests mit Report-Bild und HTML-Übersicht

## Projektstruktur

```text
app/streamlit_app.py     Streamlit UI
app/webcam_demo.py       Live-Demo für Webcam oder Videodatei
src/core.py              Settings, Detektoren und Inferenz-Pipeline
src/draw.py              Bounding-Boxes und Report-Visualisierung
src/deepfashion2_yolo.py DeepFashion2 -> YOLO Labels
src/matchtest.py         Matchtest- und Batch-Report-Logik
main.py                  CLI für Konvertierung und Matchtests
requirements.txt         Python-Abhängigkeiten
```

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

Hinweise:

- `q` oder `Esc` beendet das Fenster.
- Falls die Standard-Webcam nicht gefunden wird, teste `--source 1` oder `--source 2`.

## DeepFashion2 -> YOLO konvertieren

Schneller Testlauf:

```bash
python main.py convert-deepfashion2-to-yolo --dataset-root "R:\Schulprojekt KI\DeepFashion2" --output-root data/deepfashion2_yolo_labels --limit-per-split 200
```

Voller Lauf:

```bash
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
