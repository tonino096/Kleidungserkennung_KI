# KI-Web-App zur Erkennung von Kleidungsstücken

Dieses Repository enthaelt ein Schulprojekt-Grundgeruest fuer eine spaetere Kleidungserkennung mit YOLOv8.
Aktuell laeuft die App im `mock`-Modus mit einer simulierten Vorhersage inklusive Bounding Box.

## Features

- Streamlit-Weboberflaeche zum Hochladen von Bildern
- Inferenz-Pipeline mit austauschbarem Detector (`mock` oder `yolo`)
- Visualisierung von Vorhersagen direkt im Bild
- Erste Tests fuer den Mock-Detector
- SQLite-Datenbank fuer Trainingsbilder und Detection-Annotationen
- DeepFashion2-Importer fuer die offiziellen `train/annos` und `validation/annos`

## Projektstruktur

```text
app/streamlit_app.py          Streamlit UI
src/config/settings.py        Konfiguration ueber Umgebungsvariablen
src/detector/base.py          Detector-Interface + Prediction-Typ
src/detector/mock_detector.py Mock-Inferenz
src/detector/yolo_detector.py YOLO-Platzhalter
src/inference/pipeline.py     Detector-Auswahl und Inferenz-Pipeline
src/training/database.py      SQLite-Schema fuer Bilder, Labels und Bounding Boxes
src/training/deepfashion2_importer.py Import von DeepFashion2-JSON nach SQLite
src/visualization/draw.py     Bounding-Box-Visualisierung
tests/test_mock_detector.py   Unit-Test fuer Mock-Detector
tests/test_deepfashion2_importer.py Import-Test fuer DeepFashion2
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
- `DATABASE_PATH` (Standard: `data/clothing.db`)
- `DEEPFASHION2_ROOT` (Standard: `data/deepfashion2`)

Beispiel in PowerShell:

```powershell
$env:DETECTOR_TYPE="mock"
$env:YOLO_WEIGHTS_PATH="models/yolov8n-clothes.pt"
streamlit run app/streamlit_app.py
```

## DeepFashion2 importieren

Erwartete Ordnerstruktur des offiziellen Datensatzes:

```text
data/deepfashion2/
├─ train/
│  ├─ image/
│  └─ annos/
└─ validation/
   ├─ image/
   └─ annos/
```

SQLite-Schema anlegen:

```bash
python main.py init-db
```

DeepFashion2 in die lokale Datenbank importieren:

```bash
python main.py import-deepfashion2 --dataset-root data/deepfashion2
```

Optional mit expliziten Splits:

```bash
python main.py import-deepfashion2 --dataset-root data/deepfashion2 --splits train validation
```

Der Import legt die 13 offiziellen DeepFashion2-Kategorien an und speichert pro Bild:
- Bildpfad, Split, Quelle und `pair_id`
- Bounding Boxes pro `item`
- Landmarks und Segmentierungen als JSON-Text
- DeepFashion2-Metadaten wie `style`, `scale`, `occlusion`, `zoom_in`, `viewpoint`

## Aktueller Stand

- `mock`: funktionsfaehig, erzeugt eine Beispielvorhersage
- `yolo`: als Platzhalter vorbereitet, Modellinferenz muss noch implementiert werden
