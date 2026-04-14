# KI-Web-App zur Erkennung von Kleidungsstücken

Dieses Repository enthaelt ein Schulprojekt-Grundgeruest fuer eine spaetere Kleidungserkennung mit YOLOv8.
Aktuell laeuft die App im `mock`-Modus mit einer simulierten Vorhersage inklusive Bounding Box.

## Features

- Streamlit-Weboberflaeche zum Hochladen von Bildern
- Inferenz-Pipeline mit austauschbarem Detector (`mock` oder `yolo`)
- Visualisierung von Vorhersagen direkt im Bild
- Erste Tests fuer den Mock-Detector

## Projektstruktur

```text
app/streamlit_app.py          Streamlit UI
src/config/settings.py        Konfiguration ueber Umgebungsvariablen
src/detector/base.py          Detector-Interface + Prediction-Typ
src/detector/mock_detector.py Mock-Inferenz
src/detector/yolo_detector.py YOLO-Platzhalter
src/inference/pipeline.py     Detector-Auswahl und Inferenz-Pipeline
src/visualization/draw.py     Bounding-Box-Visualisierung
tests/test_mock_detector.py   Unit-Test fuer Mock-Detector
```

## Voraussetzungen

- Python 3.10+ (empfohlen)

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

Danach ist die App lokal im Browser verfuegbar (typisch unter `http://localhost:8501`).

## Tests ausfuehren

```bash
pytest
```

## Konfiguration

Die App liest folgende Umgebungsvariablen aus:

- `DETECTOR_TYPE` (Standard: `mock`)
- `YOLO_WEIGHTS_PATH` (Standard: `models/yolov8n-clothes.pt`)

Beispiel in PowerShell:

```powershell
$env:DETECTOR_TYPE="mock"
$env:YOLO_WEIGHTS_PATH="models/yolov8n-clothes.pt"
streamlit run app/streamlit_app.py
```

## Aktueller Stand

- `mock`: funktionsfaehig, erzeugt eine Beispielvorhersage
- `yolo`: als Platzhalter vorbereitet, Modellinferenz muss noch implementiert werden
