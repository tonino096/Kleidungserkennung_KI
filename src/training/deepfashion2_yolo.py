from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

# DeepFashion2 nutzt numerische Kategorien. Für das Projekt wird einmalig
# festgelegt, welcher Zahlenwert welchem Kleidungsnamen entspricht.
DEEPFASHION2_CATEGORIES: dict[int, str] = {
    1: "short sleeve top",
    2: "long sleeve top",
    3: "short sleeve outwear",
    4: "long sleeve outwear",
    5: "vest",
    6: "sling",
    7: "shorts",
    8: "trousers",
    9: "skirt",
    10: "short sleeve dress",
    11: "long sleeve dress",
    12: "vest dress",
    13: "sling dress",
}


class DeepFashion2ToYoloConverter:
    def __init__(self, dataset_root: str | Path, output_root: str | Path) -> None:
        # Beide Pfade werden als Path-Objekte gespeichert, weil später sehr
        # viel mit Unterordnern und Dateinamen gearbeitet wird.
        self.dataset_root = Path(dataset_root)
        self.output_root = Path(output_root)

    def convert(
        self,
        splits: Iterable[str] = ("train", "validation"),
        limit_per_split: int | None = None,
    ) -> dict[str, int]:
        # Die Zusammenfassung hilft später dabei, schnell zu sehen, wie viele
        # Bilder und Labels erfolgreich übernommen wurden.
        summary = {
            "images": 0,
            "labels": 0,
            "skipped_images": 0,
            "skipped_annotations": 0,
        }
        self._prepare_output_dirs()

        # Jeder Split wird getrennt verarbeitet, damit train und validation
        # unabhängig voneinander begrenzt oder geprüft werden können.
        for split in splits:
            split_summary = self._convert_split(split, limit_per_split)
            for key, value in split_summary.items():
                summary[key] += value

        self._write_data_yaml(splits)
        return summary

    def _convert_split(self, split: str, limit_per_split: int | None) -> dict[str, int]:
        # Zuerst wird die tatsächliche Split-Struktur aufgelöst, da DeepFashion2
        # je nach Download direkt oder verschachtelt vorliegen kann.
        split_root = self._resolve_split_root(split)
        image_dir = split_root / "image"
        anno_dir = split_root / "annos"
        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        if not anno_dir.exists():
            raise FileNotFoundError(f"Annotation directory not found: {anno_dir}")

        yolo_split = self._map_split_name(split)
        out_label_dir = self.output_root / "labels" / yolo_split

        summary = {
            "images": 0,
            "labels": 0,
            "skipped_images": 0,
            "skipped_annotations": 0,
        }

        annotation_paths = sorted(anno_dir.glob("*.json"))
        if limit_per_split is not None:
            # Das Limit ist vor allem für schnelle Probeläufe hilfreich.
            annotation_paths = annotation_paths[:limit_per_split]

        for annotation_path in annotation_paths:
            # Es werden nur Annotationen übernommen, zu denen auch wirklich
            # eine passende Bilddatei vorhanden ist.
            image_path = self._find_image_path(image_dir, annotation_path.stem)
            if image_path is None:
                summary["skipped_images"] += 1
                continue

            with annotation_path.open("r", encoding="utf-8") as file:
                payload = json.load(file)

            lines = self._build_yolo_lines(payload, image_path)
            if not lines:
                summary["skipped_annotations"] += 1
                continue

            target_label_path = out_label_dir / f"{annotation_path.stem}.txt"
            target_label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

            summary["images"] += 1
            summary["labels"] += len(lines)

        return summary

    def _build_yolo_lines(self, payload: dict, image_path: Path) -> list[str]:
        # YOLO benötigt normalisierte Mittelpunkt-/Breiten-/Höhenwerte.
        # Deshalb werden die DeepFashion2-Boxen hier umgerechnet.
        width, height = self._read_image_size(payload, image_path)
        lines: list[str] = []

        for key, value in payload.items():
            # Relevante Kleidungsstücke liegen in DeepFashion2 als item1, item2, ...
            if not key.startswith("item") or not isinstance(value, dict):
                continue

            category_id = value.get("category_id")
            bbox = value.get("bounding_box")
            if not isinstance(category_id, int) or category_id not in DEEPFASHION2_CATEGORIES:
                continue
            if not self._is_valid_bbox(bbox):
                continue

            class_id = category_id - 1
            x1, y1, x2, y2 = [float(number) for number in bbox]
            x_center = ((x1 + x2) / 2.0) / width
            y_center = ((y1 + y2) / 2.0) / height
            box_width = (x2 - x1) / width
            box_height = (y2 - y1) / height

            values = [class_id, x_center, y_center, box_width, box_height]
            lines.append(f"{values[0]} {values[1]:.6f} {values[2]:.6f} {values[3]:.6f} {values[4]:.6f}")

        return lines

    def _prepare_output_dirs(self) -> None:
        # YOLO erwartet getrennte Labelordner für Train und Validation.
        for split in ("train", "val"):
            (self.output_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    def _write_data_yaml(self, splits: Iterable[str]) -> None:
        # Die data.yaml ist die zentrale YOLO-Konfigurationsdatei, über die
        # Training und Validierung später auf Bilder und Klassennamen zugreifen.
        split_roots = {
            self._map_split_name(split): self._resolve_split_root(split) / "image"
            for split in splits
        }
        lines = [
            f"path: {self.output_root.as_posix()}",
            f"train: {split_roots['train'].as_posix()}",
        ]
        if "val" in split_roots:
            lines.append(f"val: {split_roots['val'].as_posix()}")
        lines.extend(
            [
                "names:",
                *[
                    f"  {category_id - 1}: {category_name}"
                    for category_id, category_name in sorted(DEEPFASHION2_CATEGORIES.items())
                ],
            ]
        )
        (self.output_root / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _resolve_split_root(self, split: str) -> Path:
        # Auch hier werden beide DeepFashion2-Strukturen unterstützt:
        # split/image oder split/split/image.
        direct_root = self.dataset_root / split
        if (direct_root / "image").exists() and (direct_root / "annos").exists():
            return direct_root

        nested_root = direct_root / split
        if (nested_root / "image").exists() and (nested_root / "annos").exists():
            return nested_root

        return direct_root

    @staticmethod
    def _map_split_name(split: str) -> str:
        # Ultralytics erwartet standardmäßig "val" statt "validation".
        return "val" if split == "validation" else split

    @staticmethod
    def _find_image_path(image_dir: Path, stem: str) -> Path | None:
        # Je nach Datensatz oder Export können unterschiedliche Dateiendungen
        # vorkommen; deshalb werden die gängigen Bildformate nacheinander geprüft.
        for suffix in (".jpg", ".jpeg", ".png"):
            candidate = image_dir / f"{stem}{suffix}"
            if candidate.exists():
                return candidate
        return None

    @staticmethod
    def _is_valid_bbox(bbox: object) -> bool:
        if not isinstance(bbox, list) or len(bbox) != 4:
            return False
        x1, y1, x2, y2 = bbox
        return all(isinstance(value, (int, float)) for value in (x1, y1, x2, y2)) and x2 > x1 and y2 > y1

    @staticmethod
    def _read_image_size(payload: dict, image_path: Path) -> tuple[int, int]:
        # Wenn die Annotation schon Bildbreite und -höhe enthält, spart das
        # ein zusätzliches Öffnen der Bilddatei.
        width = payload.get("width")
        height = payload.get("height")
        if isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0:
            return width, height

        from PIL import Image

        with Image.open(image_path) as image:
            return image.size
