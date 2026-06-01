from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from PIL import Image

from src.training.database import ClothingDatabase


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


class DeepFashion2Importer:
    """
    Imports DeepFashion2 training/validation annotations into the local SQLite DB.

    Expected directory layout from the official dataset:
    - train/image
    - train/annos
    - validation/image
    - validation/annos
    """

    def __init__(self, dataset_root: str | Path, database: ClothingDatabase) -> None:
        self.dataset_root = Path(dataset_root)
        self.database = database

    def import_all(self, splits: Iterable[str] = ("train", "validation")) -> dict[str, int]:
        summary = {
            "images": 0,
            "annotations": 0,
            "skipped_images": 0,
            "skipped_annotations": 0,
        }
        self.database.create_tables()
        self._seed_categories()

        for split in splits:
            split_summary = self.import_split(split)
            for key, value in split_summary.items():
                summary[key] += value
        return summary

    def import_split(self, split: str) -> dict[str, int]:
        image_dir = self.dataset_root / split / "image"
        anno_dir = self.dataset_root / split / "annos"
        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        if not anno_dir.exists():
            raise FileNotFoundError(f"Annotation directory not found: {anno_dir}")

        summary = {
            "images": 0,
            "annotations": 0,
            "skipped_images": 0,
            "skipped_annotations": 0,
        }

        for annotation_path in sorted(anno_dir.glob("*.json")):
            image_path = image_dir / f"{annotation_path.stem}.jpg"
            if not image_path.exists():
                image_path = image_dir / f"{annotation_path.stem}.png"
            if not image_path.exists():
                summary["skipped_images"] += 1
                continue

            with annotation_path.open("r", encoding="utf-8") as file:
                payload = json.load(file)

            width, height = self._read_image_size(image_path)
            image_id = self.database.upsert_image(
                str(image_path),
                split=split,
                source=payload.get("source"),
                dataset_name="deepfashion2",
                pair_id=self._to_int(payload.get("pair_id")),
                width=width,
                height=height,
                is_training=(split == "train"),
                is_test=(split == "test"),
            )
            summary["images"] += 1

            for item_key, item_payload in self._iter_items(payload):
                category_id = self._to_int(item_payload.get("category_id"))
                category_name = DEEPFASHION2_CATEGORIES.get(category_id)
                bbox = item_payload.get("bounding_box")

                if category_name is None or not self._is_valid_bbox(bbox):
                    summary["skipped_annotations"] += 1
                    continue

                label_id = self.database.get_or_create_clothing_type(
                    category_name,
                    dataset_category_id=category_id,
                )
                self.database.add_annotation(
                    image_id=image_id,
                    label_id=label_id,
                    item_key=item_key,
                    bbox=[float(value) for value in bbox],
                    segmentation=item_payload.get("segmentation"),
                    landmarks=item_payload.get("landmarks"),
                    style=self._to_int(item_payload.get("style")),
                    scale=self._to_int(item_payload.get("scale")),
                    occlusion=self._to_int(item_payload.get("occlusion")),
                    zoom_in=self._to_int(item_payload.get("zoom_in")),
                    viewpoint=self._to_int(item_payload.get("viewpoint")),
                )
                summary["annotations"] += 1

        return summary

    def _seed_categories(self) -> None:
        for category_id, category_name in DEEPFASHION2_CATEGORIES.items():
            self.database.get_or_create_clothing_type(
                category_name,
                dataset_category_id=category_id,
            )

    @staticmethod
    def _iter_items(payload: dict) -> Iterable[tuple[str, dict]]:
        for key, value in payload.items():
            if key.startswith("item") and isinstance(value, dict):
                yield key, value

    @staticmethod
    def _is_valid_bbox(bbox: object) -> bool:
        if not isinstance(bbox, list) or len(bbox) != 4:
            return False
        x1, y1, x2, y2 = bbox
        return all(isinstance(value, (int, float)) for value in (x1, y1, x2, y2)) and x2 > x1 and y2 > y1

    @staticmethod
    def _to_int(value: object) -> int | None:
        if value is None:
            return None
        return int(value)

    @staticmethod
    def _read_image_size(image_path: Path) -> tuple[int, int]:
        with Image.open(image_path) as image:
            return image.size
