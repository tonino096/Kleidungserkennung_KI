from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from src.detector.base import Prediction
from src.detector.yolo_detector import YoloDetector
from src.training.deepfashion2_yolo import DEEPFASHION2_CATEGORIES
from src.visualization.draw import draw_batch_report, draw_match_comparison, get_category_color

# Diese Datei beantwortet die Frage: "Wie gut trifft das Modell die echten
# DeepFashion2-Boxen?" Dafür werden Vorhersagen und Ground Truth verglichen.


@dataclass
class MatchRecord:
    # Ein MatchRecord beschreibt das Ergebnis genau einer Ground-Truth-Box:
    # ob sie gefunden wurde, mit welchem Label und welcher IoU.
    ground_truth_label: str
    predicted_label: str | None
    iou: float
    matched: bool


@dataclass
class MatchEvaluation:
    records: list[MatchRecord]
    unmatched_predictions: list[Prediction]

    @property
    def matched_count(self) -> int:
        # Die Zahl der Treffer wird dynamisch aus den Einzel-Records berechnet,
        # damit keine doppelte Zustandsverwaltung nötig ist.
        return len([record for record in self.records if record.matched])


class DeepFashion2MatchTester:
    def __init__(
        self,
        dataset_root: str | Path,
        weights_path: str | Path,
        output_root: str | Path,
    ) -> None:
        # Der Matchtester kapselt Datensatzpfad, Modell und Ausgabeordner,
        # damit Einzel- und Batchläufe dieselbe Logik wiederverwenden.
        self.dataset_root = Path(dataset_root)
        self.weights_path = Path(weights_path)
        self.output_root = Path(output_root)
        self.detector = YoloDetector(weights_path=str(self.weights_path))

    def run(
        self,
        split: str = "validation",
        annotation_stem: str | None = None,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.5,
    ) -> dict[str, Any]:
        # Der Einzeltest verarbeitet genau eine Annotation und erzeugt eine
        # direkt lesbare Zusammenfassung für App, CLI und JSON-Export.
        split_root = self._resolve_split_root(split)
        annotation_path = self._resolve_annotation_path(split_root, annotation_stem)
        image_summary = self._process_annotation(
            split_root=split_root,
            split=split,
            annotation_path=annotation_path,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            image_output_dir=self.output_root,
        )

        summary = {
            "image_path": image_summary["image_path"],
            "annotation_path": image_summary["annotation_path"],
            "output_image_path": image_summary["output_image_path"],
            "ground_truth_count": image_summary["ground_truth_count"],
            "prediction_count": image_summary["prediction_count"],
            "matches": image_summary["match_count"],
            "match_records": image_summary["match_records"],
            "false_positives": image_summary["false_positives"],
            "precision": image_summary["precision"],
            "recall": image_summary["recall"],
        }
        self.output_root.mkdir(parents=True, exist_ok=True)
        summary_path = self.output_root / f"{split}_{annotation_path.stem}_matchtest.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summary["summary_path"] = str(summary_path)
        return summary

    def run_batch(
        self,
        *,
        split: str = "validation",
        limit: int = 12,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.5,
    ) -> dict[str, Any]:
        # Im Batchlauf werden mehrere Bilder nacheinander ausgewertet und
        # anschließend zu Gesamtmetriken und Berichten zusammengefasst.
        split_root = self._resolve_split_root(split)
        annotation_paths = sorted((split_root / "annos").glob("*.json"))
        if limit > 0:
            annotation_paths = annotation_paths[:limit]
        if not annotation_paths:
            raise FileNotFoundError(f"No annotation files found for split: {split}")

        image_output_dir = self.output_root / "images"
        image_output_dir.mkdir(parents=True, exist_ok=True)

        image_summaries: list[dict[str, Any]] = []
        for annotation_path in annotation_paths:
            # Jedes Bild wird mit exakt derselben Logik ausgewertet wie beim
            # Einzeltest, damit beide Modi vergleichbar bleiben.
            image_summary = self._process_annotation(
                split_root=split_root,
                split=split,
                annotation_path=annotation_path,
                confidence_threshold=confidence_threshold,
                iou_threshold=iou_threshold,
                image_output_dir=image_output_dir,
            )
            image_summaries.append(image_summary)

        totals = summarize_image_summaries(image_summaries)
        category_rows = build_category_rows(image_summaries)
        report_image = draw_batch_report(
            totals=totals,
            category_rows=category_rows,
            sample_image_paths=[Path(summary["output_image_path"]) for summary in image_summaries[:4]],
        )

        self.output_root.mkdir(parents=True, exist_ok=True)
        report_image_path = self.output_root / f"{split}_batch_report.png"
        report_image.save(report_image_path, quality=95)

        summary_json_path = self.output_root / f"{split}_batch_summary.json"
        summary_payload = {
            "split": split,
            "limit": limit,
            "confidence_threshold": confidence_threshold,
            "iou_threshold": iou_threshold,
            "totals": totals,
            "categories": category_rows,
            "images": image_summaries,
            "report_image_path": str(report_image_path),
        }
        summary_json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

        report_html_path = self.output_root / f"{split}_batch_report.html"
        report_html_path.write_text(
            build_html_report(
                split=split,
                totals=totals,
                category_rows=category_rows,
                image_summaries=image_summaries,
                output_root=self.output_root,
            ),
            encoding="utf-8",
        )

        return {
            "split": split,
            "processed_images": totals["images_processed"],
            "ground_truth_count": totals["ground_truth_count"],
            "prediction_count": totals["prediction_count"],
            "match_count": totals["match_count"],
            "false_positives": totals["false_positives"],
            "precision": totals["precision"],
            "recall": totals["recall"],
            "categories": category_rows,
            "images": image_summaries,
            "report_image_path": str(report_image_path),
            "report_html_path": str(report_html_path),
            "summary_json_path": str(summary_json_path),
        }

    def run_prediction_only(
        self,
        *,
        split: str = "test",
        image_stem: str | None = None,
        confidence_threshold: float = 0.25,
    ) -> dict[str, Any]:
        # Für den Test-Split ohne Ground Truth werden nur Vorhersagen
        # berechnet, gerendert und als JSON zusammengefasst.
        split_root = self._resolve_image_only_split_root(split)
        image_path = self._resolve_prediction_image_path(split_root / "image", image_stem)
        image_summary = self._process_prediction_only_image(
            split=split,
            image_path=image_path,
            confidence_threshold=confidence_threshold,
            image_output_dir=self.output_root,
        )

        summary = {
            "split": split,
            "image_path": image_summary["image_path"],
            "output_image_path": image_summary["output_image_path"],
            "prediction_count": image_summary["prediction_count"],
            "predictions": image_summary["predictions"],
        }
        self.output_root.mkdir(parents=True, exist_ok=True)
        summary_path = self.output_root / f"{split}_{image_path.stem}_prediction_only.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summary["summary_path"] = str(summary_path)
        return summary

    def run_prediction_only_batch(
        self,
        *,
        split: str = "test",
        limit: int = 12,
        confidence_threshold: float = 0.25,
    ) -> dict[str, Any]:
        # Im Test-Split ohne Annotationen entstehen keine Matchmetriken,
        # sondern nur gerenderte Vorhersagebilder plus JSON-Übersicht.
        split_root = self._resolve_image_only_split_root(split)
        image_paths = sorted(self._iter_image_paths(split_root / "image"))
        if limit > 0:
            image_paths = image_paths[:limit]
        if not image_paths:
            raise FileNotFoundError(f"No image files found for split: {split}")

        image_output_dir = self.output_root / "images"
        image_output_dir.mkdir(parents=True, exist_ok=True)

        image_summaries: list[dict[str, Any]] = []
        for image_path in image_paths:
            image_summary = self._process_prediction_only_image(
                split=split,
                image_path=image_path,
                confidence_threshold=confidence_threshold,
                image_output_dir=image_output_dir,
            )
            image_summaries.append(image_summary)

        prediction_count = sum(summary["prediction_count"] for summary in image_summaries)
        summary_payload = {
            "split": split,
            "limit": limit,
            "confidence_threshold": confidence_threshold,
            "processed_images": len(image_summaries),
            "prediction_count": prediction_count,
            "images": image_summaries,
        }

        self.output_root.mkdir(parents=True, exist_ok=True)
        summary_json_path = self.output_root / f"{split}_prediction_only_batch.json"
        summary_json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

        return {
            "split": split,
            "processed_images": len(image_summaries),
            "prediction_count": prediction_count,
            "images": image_summaries,
            "summary_json_path": str(summary_json_path),
        }

    def _process_annotation(
        self,
        *,
        split_root: Path,
        split: str,
        annotation_path: Path,
        confidence_threshold: float,
        iou_threshold: float,
        image_output_dir: Path,
    ) -> dict[str, Any]:
        # Zuerst werden Annotation, Ground Truth und Bild geladen. Danach
        # folgen Vorhersage, Filterung, Matching und Visualisierung.
        image_path = self._resolve_image_path(split_root / "image", annotation_path.stem)

        with annotation_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        ground_truth = self._build_ground_truth_predictions(payload)
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
            raw_predictions = self.detector.predict(image)

        predictions = [
            prediction for prediction in raw_predictions if prediction["confidence"] >= confidence_threshold
        ]
        # Nur Vorhersagen oberhalb des gewählten Confidence-Schwellenwerts
        # fließen in Bewertung und Visualisierung ein.
        evaluation = evaluate_predictions(ground_truth, predictions, iou_threshold=iou_threshold)

        rendered = draw_match_comparison(image, ground_truth, predictions)
        image_output_dir.mkdir(parents=True, exist_ok=True)
        output_image_path = image_output_dir / f"{split}_{annotation_path.stem}_matchtest.jpg"
        rendered.save(output_image_path, quality=95)

        matched_labels = [
            record.ground_truth_label
            for record in evaluation.records
            if record.matched
        ]
        return {
            "annotation_stem": annotation_path.stem,
            "image_path": str(image_path),
            "annotation_path": str(annotation_path),
            "output_image_path": str(output_image_path),
            "ground_truth_count": len(ground_truth),
            "prediction_count": len(predictions),
            "match_count": evaluation.matched_count,
            "false_positives": len(evaluation.unmatched_predictions),
            "precision": safe_ratio(evaluation.matched_count, len(predictions)),
            "recall": safe_ratio(evaluation.matched_count, len(ground_truth)),
            "ground_truth_by_label": count_labels(ground_truth),
            "prediction_by_label": count_labels(predictions),
            "matched_by_label": count_strings(matched_labels),
            "match_records": [asdict(record) for record in evaluation.records],
        }

    def _process_prediction_only_image(
        self,
        *,
        split: str,
        image_path: Path,
        confidence_threshold: float,
        image_output_dir: Path,
    ) -> dict[str, Any]:
        # Diese Variante verarbeitet ein Bild ohne Ground Truth und zeichnet
        # ausschließlich die Modellvorhersagen für den Test-Split.
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
            raw_predictions = self.detector.predict(image)

        predictions = [
            prediction for prediction in raw_predictions if prediction["confidence"] >= confidence_threshold
        ]
        rendered = draw_match_comparison(image, [], predictions)
        image_output_dir.mkdir(parents=True, exist_ok=True)
        output_image_path = image_output_dir / f"{split}_{image_path.stem}_prediction_only.jpg"
        rendered.save(output_image_path, quality=95)

        return {
            "image_stem": image_path.stem,
            "image_path": str(image_path),
            "output_image_path": str(output_image_path),
            "prediction_count": len(predictions),
            "prediction_by_label": count_labels(predictions),
            "predictions": predictions,
        }

    def _resolve_split_root(self, split: str) -> Path:
        # Unterstützt werden sowohl direkte Splitordner als auch die bei
        # DeepFashion2 häufige doppelte Verschachtelung.
        if _is_split_root(self.dataset_root):
            return self.dataset_root

        direct_root = self.dataset_root / split
        if (direct_root / "image").exists() and (direct_root / "annos").exists():
            return direct_root

        nested_root = direct_root / split
        if (nested_root / "image").exists() and (nested_root / "annos").exists():
            return nested_root

        available_splits = discover_available_splits(self.dataset_root)
        available_text = ", ".join(available_splits) if available_splits else "keine erkannt"
        raise FileNotFoundError(
            "DeepFashion2 split not found: "
            f"{split}. Checked: '{direct_root}' and '{nested_root}'. "
            f"Available splits: {available_text}. "
            "Expected folders like 'train/image + train/annos' or "
            "'train/train/image + train/train/annos'."
        )

    def _resolve_image_only_split_root(self, split: str) -> Path:
        # Für den Vorhersagemodus reicht ein Bildordner, auch wenn keine
        # Annotationsdateien vorhanden sind.
        if _is_image_only_split_root(self.dataset_root):
            return self.dataset_root

        direct_root = self.dataset_root / split
        if _is_image_only_split_root(direct_root):
            return direct_root

        nested_root = direct_root / split
        if _is_image_only_split_root(nested_root):
            return nested_root

        raise FileNotFoundError(
            "DeepFashion2 image-only split not found: "
            f"{split}. Checked: '{direct_root}' and '{nested_root}'. "
            "Expected folders like 'test/image' or 'test/test/image'."
        )

    @staticmethod
    def _resolve_annotation_path(split_root: Path, annotation_stem: str | None) -> Path:
        # Ohne expliziten Stem wird einfach die erste vorhandene Annotation
        # verwendet, damit der Einzeltest auch ohne Zusatzangaben lauffähig ist.
        annotation_dir = split_root / "annos"
        if annotation_stem is not None:
            annotation_path = annotation_dir / f"{annotation_stem}.json"
            if not annotation_path.exists():
                raise FileNotFoundError(f"Annotation not found: {annotation_path}")
            return annotation_path

        try:
            return sorted(annotation_dir.glob("*.json"))[0]
        except IndexError as error:
            raise FileNotFoundError(f"No annotation files found in: {annotation_dir}") from error

    @staticmethod
    def _resolve_image_path(image_dir: Path, stem: str) -> Path:
        # Annotation und Bild werden über denselben Dateistamm verknüpft.
        for suffix in (".jpg", ".jpeg", ".png"):
            candidate = image_dir / f"{stem}{suffix}"
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"No matching image found for annotation stem: {stem}")

    @staticmethod
    def _resolve_prediction_image_path(image_dir: Path, image_stem: str | None) -> Path:
        # Im Test-Split wird direkt per Bildname oder notfalls über das erste
        # vorhandene Bild gearbeitet.
        if image_stem is not None:
            return DeepFashion2MatchTester._resolve_image_path(image_dir, image_stem)

        try:
            return sorted(DeepFashion2MatchTester._iter_image_paths(image_dir))[0]
        except IndexError as error:
            raise FileNotFoundError(f"No image files found in: {image_dir}") from error

    @staticmethod
    def _iter_image_paths(image_dir: Path) -> list[Path]:
        # Alle unterstützten Bildendungen werden gesammelt, damit Batch- und
        # Einzelmodus dieselbe Dateilogik nutzen.
        image_paths: list[Path] = []
        for suffix in ("*.jpg", "*.jpeg", "*.png"):
            image_paths.extend(image_dir.glob(suffix))
        return image_paths

    @staticmethod
    def _build_ground_truth_predictions(payload: dict[str, Any]) -> list[Prediction]:
        # Die echten Boxen aus DeepFashion2 werden in dasselbe Format gebracht
        # wie die Modellvorhersagen, damit beide direkt vergleichbar sind.
        predictions: list[Prediction] = []

        for key, value in payload.items():
            if not key.startswith("item") or not isinstance(value, dict):
                continue

            category_id = value.get("category_id")
            bbox = value.get("bounding_box")
            if category_id not in DEEPFASHION2_CATEGORIES:
                continue
            if not _is_valid_bbox(bbox):
                continue

            x1, y1, x2, y2 = [int(round(number)) for number in bbox]
            predictions.append(
                {
                    "label": DEEPFASHION2_CATEGORIES[int(category_id)],
                    "confidence": 1.0,
                    "bbox": [x1, y1, x2, y2],
                }
            )

        return predictions


def evaluate_predictions(
    ground_truth: list[Prediction],
    predictions: list[Prediction],
    iou_threshold: float = 0.5,
) -> MatchEvaluation:
    # Jede Ground-Truth-Box sucht sich die beste noch freie Vorhersage
    # derselben Klasse. So vermeiden wir Mehrfachbelegungen.
    remaining_predictions = predictions.copy()
    records: list[MatchRecord] = []

    for target in ground_truth:
        best_index = -1
        best_iou = 0.0

        for index, prediction in enumerate(remaining_predictions):
            if prediction["label"] != target["label"]:
                continue

            # Die IoU sagt, wie stark echte Box und Vorhersage überlappen.
            # Nur die beste Vorhersage darf am Ende als Treffer zählen.
            iou = calculate_iou(target["bbox"], prediction["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_index = index

        if best_index >= 0 and best_iou >= iou_threshold:
            matched_prediction = remaining_predictions.pop(best_index)
            records.append(
                MatchRecord(
                    ground_truth_label=target["label"],
                    predicted_label=matched_prediction["label"],
                    iou=best_iou,
                    matched=True,
                )
            )
            continue

        records.append(
            MatchRecord(
                ground_truth_label=target["label"],
                predicted_label=None,
                iou=best_iou,
                matched=False,
            )
        )

    return MatchEvaluation(records=records, unmatched_predictions=remaining_predictions)


def summarize_image_summaries(image_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    # Aus den Einzelbild-Ergebnissen werden Gesamtzahlen für den Batchreport.
    ground_truth_count = sum(summary["ground_truth_count"] for summary in image_summaries)
    prediction_count = sum(summary["prediction_count"] for summary in image_summaries)
    match_count = sum(summary["match_count"] for summary in image_summaries)
    false_positives = sum(summary["false_positives"] for summary in image_summaries)

    return {
        "images_processed": len(image_summaries),
        "ground_truth_count": ground_truth_count,
        "prediction_count": prediction_count,
        "match_count": match_count,
        "false_positives": false_positives,
        "precision": safe_ratio(match_count, prediction_count),
        "recall": safe_ratio(match_count, ground_truth_count),
    }


def build_category_rows(image_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Diese Tabelle fasst zusammen, wie gut jede einzelne Kleidungs-Kategorie
    # im Batch abgeschnitten hat.
    categories = [name for _, name in sorted(DEEPFASHION2_CATEGORIES.items())]
    rows: list[dict[str, Any]] = []

    for label in categories:
        ground_truth = sum(summary["ground_truth_by_label"].get(label, 0) for summary in image_summaries)
        predicted = sum(summary["prediction_by_label"].get(label, 0) for summary in image_summaries)
        matched = sum(summary["matched_by_label"].get(label, 0) for summary in image_summaries)
        if ground_truth == 0 and predicted == 0:
            continue

        rows.append(
            {
                "label": label,
                "ground_truth": ground_truth,
                "predicted": predicted,
                "matched": matched,
                "precision": safe_ratio(matched, predicted),
                "recall": safe_ratio(matched, ground_truth),
                "color": rgb_to_hex(get_category_color(label)),
            }
        )

    rows.sort(key=lambda row: (-row["recall"], row["label"]))
    return rows


def build_html_report(
    *,
    split: str,
    totals: dict[str, Any],
    category_rows: list[dict[str, Any]],
    image_summaries: list[dict[str, Any]],
    output_root: Path,
) -> str:
    # Der HTML-Report ist eine leicht teilbare Alternative zum PNG-Report und
    # zeigt dieselben Kennzahlen zusätzlich in einer Galerieansicht.
    category_cards = "\n".join(
        f"""
        <div class="category-row">
          <div class="category-name">{row['label']}</div>
          <div class="category-bar">
            <div class="category-fill" style="width: {row['recall'] * 100:.1f}%; background: {row['color']};"></div>
          </div>
          <div class="category-metric">{row['matched']}/{row['ground_truth']} ({row['recall'] * 100:.1f}%)</div>
        </div>
        """
        for row in category_rows
    )

    gallery_cards = "\n".join(
        f"""
        <article class="image-card">
          <img src="{Path(summary['output_image_path']).relative_to(output_root).as_posix()}" alt="{summary['annotation_stem']}">
          <div class="image-meta">
            <h3>{summary['annotation_stem']}</h3>
            <p>Recall: {summary['recall'] * 100:.1f}% | Precision: {summary['precision'] * 100:.1f}%</p>
            <p>GT: {summary['ground_truth_count']} | Predictions: {summary['prediction_count']} | Matches: {summary['match_count']}</p>
          </div>
        </article>
        """
        for summary in image_summaries
    )

    return f"""<!DOCTYPE html>
<html lang="de">
<head>
  <meta charset="utf-8">
  <title>DeepFashion2 Batch Report</title>
  <style>
    body {{
      margin: 0;
      font-family: "Segoe UI", Arial, sans-serif;
      background: #f7f3ec;
      color: #2d2d2d;
    }}
    .wrap {{
      max-width: 1400px;
      margin: 0 auto;
      padding: 32px;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(6, 1fr);
      gap: 16px;
      margin-bottom: 28px;
    }}
    .stat-card {{
      background: #fffaf3;
      border: 2px solid #e2d8ca;
      border-radius: 18px;
      padding: 18px;
    }}
    .stat-card .label {{
      color: #6c6c6c;
      font-size: 14px;
      margin-bottom: 10px;
    }}
    .stat-card .value {{
      color: #245079;
      font-size: 30px;
      font-weight: 700;
    }}
    .panel {{
      background: #fffaf3;
      border: 2px solid #e2d8ca;
      border-radius: 20px;
      padding: 24px;
      margin-bottom: 28px;
    }}
    .category-row {{
      display: grid;
      grid-template-columns: 260px 1fr 220px;
      gap: 16px;
      align-items: center;
      margin-bottom: 12px;
    }}
    .category-bar {{
      height: 20px;
      background: #e7e1d8;
      border-radius: 999px;
      overflow: hidden;
    }}
    .category-fill {{
      height: 100%;
      border-radius: 999px;
    }}
    .gallery {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 20px;
    }}
    .image-card {{
      background: #ffffff;
      border-radius: 18px;
      overflow: hidden;
      border: 2px solid #e2d8ca;
    }}
    .image-card img {{
      width: 100%;
      display: block;
    }}
    .image-meta {{
      padding: 16px 18px 20px;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>DeepFashion2 Batch Report</h1>
    <p>Split: {split}</p>
    <section class="stats">
      <div class="stat-card"><div class="label">Bilder</div><div class="value">{totals['images_processed']}</div></div>
      <div class="stat-card"><div class="label">GT Boxen</div><div class="value">{totals['ground_truth_count']}</div></div>
      <div class="stat-card"><div class="label">Vorhersagen</div><div class="value">{totals['prediction_count']}</div></div>
      <div class="stat-card"><div class="label">Matches</div><div class="value">{totals['match_count']}</div></div>
      <div class="stat-card"><div class="label">Recall</div><div class="value">{totals['recall'] * 100:.1f}%</div></div>
      <div class="stat-card"><div class="label">Precision</div><div class="value">{totals['precision'] * 100:.1f}%</div></div>
    </section>
    <section class="panel">
      <h2>Trefferquote pro Kategorie</h2>
      {category_cards}
    </section>
    <section class="panel">
      <h2>Beispielbilder</h2>
      <div class="gallery">
        {gallery_cards}
      </div>
    </section>
  </div>
</body>
</html>
"""


def calculate_iou(first_box: list[int], second_box: list[int]) -> float:
    # Die Intersection over Union misst, wie stark sich zwei Boxen überlappen.
    # Sie ist die zentrale Grundlage für das Matching.
    first_x1, first_y1, first_x2, first_y2 = first_box
    second_x1, second_y1, second_x2, second_y2 = second_box

    intersection_x1 = max(first_x1, second_x1)
    intersection_y1 = max(first_y1, second_y1)
    intersection_x2 = min(first_x2, second_x2)
    intersection_y2 = min(first_y2, second_y2)

    intersection_width = max(0, intersection_x2 - intersection_x1)
    intersection_height = max(0, intersection_y2 - intersection_y1)
    intersection_area = intersection_width * intersection_height
    if intersection_area == 0:
        # Ohne Überschneidung gibt es auch keinen Treffer.
        return 0.0

    first_area = max(0, first_x2 - first_x1) * max(0, first_y2 - first_y1)
    second_area = max(0, second_x2 - second_x1) * max(0, second_y2 - second_y1)
    union_area = first_area + second_area - intersection_area
    if union_area <= 0:
        return 0.0

    return intersection_area / union_area


def count_labels(predictions: list[Prediction]) -> dict[str, int]:
    # Hilfsfunktion für Statistik pro Kategorie.
    return count_strings([prediction["label"] for prediction in predictions])


def count_strings(values: list[str]) -> dict[str, int]:
    # Zählt beliebige Strings, damit keine zusätzliche Bibliothek nötig ist.
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def safe_ratio(numerator: int, denominator: int) -> float:
    # Verhindert Division durch Null bei leeren Vorhersage- oder GT-Mengen.
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def rgb_to_hex(color: tuple[int, int, int]) -> str:
    # Für den HTML-Report werden RGB-Werte als CSS-Hexfarbe gebraucht.
    return "#{:02x}{:02x}{:02x}".format(*color)


def discover_available_splits(dataset_root: str | Path) -> list[str]:
    # Diese Erkennung hilft sowohl bei Fehlermeldungen als auch bei der UI,
    # um vorhandene Datensatz-Splits sichtbar zu machen.
    root = Path(dataset_root)
    if _is_split_root(root):
        return [root.name]

    available: list[str] = []
    for split in ("validation", "train", "test"):
        direct_root = root / split
        nested_root = direct_root / split
        if (
            _is_split_root(direct_root)
            or _is_split_root(nested_root)
            or _is_image_only_split_root(direct_root)
            or _is_image_only_split_root(nested_root)
        ):
            available.append(split)
    return available


def _is_split_root(path: Path) -> bool:
    # Ein Split gilt genau dann als gültig, wenn Bild- und Annotationsordner
    # vorhanden sind.
    return (path / "image").exists() and (path / "annos").exists()


def _is_image_only_split_root(path: Path) -> bool:
    # Fuer den reinen Vorhersagemodus reicht beim Test-Split ein Bildordner.
    return (path / "image").exists()


def _is_valid_bbox(bbox: object) -> bool:
    # Ungültige oder invertierte Bounding Boxes werden konsequent verworfen.
    if not isinstance(bbox, list) or len(bbox) != 4:
        return False

    x1, y1, x2, y2 = bbox
    return all(isinstance(value, (int, float)) for value in (x1, y1, x2, y2)) and x2 > x1 and y2 > y1
