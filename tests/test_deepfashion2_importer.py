from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from src.training.database import ClothingDatabase
from src.training.deepfashion2_importer import DeepFashion2Importer


def _create_sample_dataset(root: Path) -> None:
    image_dir = root / "train" / "image"
    anno_dir = root / "train" / "annos"
    image_dir.mkdir(parents=True)
    anno_dir.mkdir(parents=True)

    Image.new("RGB", (120, 80), "white").save(image_dir / "000001.jpg")
    payload = {
        "source": "user",
        "pair_id": 7,
        "item1": {
            "category_name": "short sleeve top",
            "category_id": 1,
            "style": 2,
            "bounding_box": [10, 12, 50, 70],
            "landmarks": [10, 20, 2],
            "segmentation": [[10, 12, 50, 12, 50, 70, 10, 70]],
            "scale": 2,
            "occlusion": 1,
            "zoom_in": 1,
            "viewpoint": 2,
        },
    }
    (anno_dir / "000001.json").write_text(json.dumps(payload), encoding="utf-8")


def test_deepfashion2_importer_imports_train_split(tmp_path: Path) -> None:
    dataset_root = tmp_path / "deepfashion2"
    _create_sample_dataset(dataset_root)
    db_path = tmp_path / "clothing.db"

    with ClothingDatabase(str(db_path)) as database:
        importer = DeepFashion2Importer(dataset_root, database)
        summary = importer.import_all(["train"])

        assert summary["images"] == 1
        assert summary["annotations"] == 1

        stats = database.get_database_stats()
        assert stats["clothing_types"] == 13
        assert stats["training_images"] == 1
        assert stats["total_annotations"] == 1

        training_images = database.get_training_images()
        assert training_images[0]["annotation_count"] == 1

        annotations = database.get_image_annotations(training_images[0]["id"])
        assert len(annotations) == 1
        assert annotations[0]["label"] == "short sleeve top"
        assert annotations[0]["bbox"] == [10.0, 12.0, 50.0, 70.0]
