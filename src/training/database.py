from __future__ import annotations

import json
import os
import sqlite3
from typing import Any


class ClothingDatabase:
    """
    SQLite storage for clothing datasets and detection annotations.

    The schema supports both the previous simple image/label flow and
    object-detection datasets such as DeepFashion2 with per-item bounding boxes.
    """

    def __init__(self, db_path: str = "data/clothing.db") -> None:
        self.db_path = db_path
        self._ensure_db_directory()
        self.connection = sqlite3.connect(self.db_path)
        self.connection.row_factory = sqlite3.Row
        self.cursor = self.connection.cursor()
        self.cursor.execute("PRAGMA foreign_keys = ON")

    def _ensure_db_directory(self) -> None:
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

    def create_tables(self) -> None:
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS clothing_types (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                description TEXT,
                dataset_category_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL UNIQUE,
                file_name TEXT NOT NULL,
                width INTEGER,
                height INTEGER,
                split TEXT,
                source TEXT,
                dataset_name TEXT,
                pair_id INTEGER,
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_training BOOLEAN DEFAULT 1,
                is_test BOOLEAN DEFAULT 0
            )
            """
        )

        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS image_labels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id INTEGER NOT NULL,
                label_id INTEGER NOT NULL,
                confidence REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE,
                FOREIGN KEY (label_id) REFERENCES clothing_types(id) ON DELETE CASCADE,
                UNIQUE(image_id, label_id)
            )
            """
        )

        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS annotations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id INTEGER NOT NULL,
                label_id INTEGER NOT NULL,
                item_key TEXT NOT NULL,
                bbox_x1 REAL NOT NULL,
                bbox_y1 REAL NOT NULL,
                bbox_x2 REAL NOT NULL,
                bbox_y2 REAL NOT NULL,
                area REAL,
                style INTEGER,
                scale INTEGER,
                occlusion INTEGER,
                zoom_in INTEGER,
                viewpoint INTEGER,
                segmentation_json TEXT,
                landmarks_json TEXT,
                confidence REAL DEFAULT 1.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE,
                FOREIGN KEY (label_id) REFERENCES clothing_types(id) ON DELETE CASCADE,
                UNIQUE(image_id, item_key)
            )
            """
        )

        self.cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_images_dataset_split
            ON images(dataset_name, split)
            """
        )
        self.cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_annotations_image_id
            ON annotations(image_id)
            """
        )
        self.cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_annotations_label_id
            ON annotations(label_id)
            """
        )

        self.connection.commit()

    def get_or_create_clothing_type(
        self,
        name: str,
        description: str | None = None,
        dataset_category_id: int | None = None,
    ) -> int:
        self.cursor.execute("SELECT id FROM clothing_types WHERE name = ?", (name,))
        existing = self.cursor.fetchone()
        if existing:
            return int(existing["id"])

        self.cursor.execute(
            """
            INSERT INTO clothing_types (name, description, dataset_category_id)
            VALUES (?, ?, ?)
            """,
            (name, description, dataset_category_id),
        )
        self.connection.commit()
        return int(self.cursor.lastrowid)

    def add_clothing_type(
        self,
        name: str,
        description: str | None = None,
        dataset_category_id: int | None = None,
    ) -> int:
        return self.get_or_create_clothing_type(name, description, dataset_category_id)

    def upsert_image(
        self,
        file_path: str,
        *,
        split: str | None = None,
        source: str | None = None,
        dataset_name: str | None = None,
        pair_id: int | None = None,
        width: int | None = None,
        height: int | None = None,
        is_training: bool = True,
        is_test: bool = False,
    ) -> int:
        file_name = os.path.basename(file_path)
        self.cursor.execute("SELECT id FROM images WHERE file_path = ?", (file_path,))
        existing = self.cursor.fetchone()
        if existing:
            self.cursor.execute(
                """
                UPDATE images
                SET file_name = ?, width = ?, height = ?, split = ?, source = ?,
                    dataset_name = ?, pair_id = ?, is_training = ?, is_test = ?
                WHERE id = ?
                """,
                (
                    file_name,
                    width,
                    height,
                    split,
                    source,
                    dataset_name,
                    pair_id,
                    int(is_training),
                    int(is_test),
                    int(existing["id"]),
                ),
            )
            self.connection.commit()
            return int(existing["id"])

        self.cursor.execute(
            """
            INSERT INTO images (
                file_path, file_name, width, height, split, source,
                dataset_name, pair_id, is_training, is_test
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                file_path,
                file_name,
                width,
                height,
                split,
                source,
                dataset_name,
                pair_id,
                int(is_training),
                int(is_test),
            ),
        )
        self.connection.commit()
        return int(self.cursor.lastrowid)

    def add_image(
        self,
        file_path: str,
        is_training: bool = True,
        is_test: bool = False,
        source: str | None = None,
    ) -> int:
        return self.upsert_image(
            file_path,
            is_training=is_training,
            is_test=is_test,
            source=source,
        )

    def add_label_to_image(
        self,
        image_id: int,
        label_id: int,
        confidence: float = 1.0,
    ) -> None:
        self.cursor.execute(
            """
            INSERT OR REPLACE INTO image_labels (image_id, label_id, confidence)
            VALUES (?, ?, ?)
            """,
            (image_id, label_id, confidence),
        )
        self.connection.commit()

    def add_annotation(
        self,
        *,
        image_id: int,
        label_id: int,
        item_key: str,
        bbox: list[float],
        segmentation: list[list[float]] | None = None,
        landmarks: list[float] | None = None,
        style: int | None = None,
        scale: int | None = None,
        occlusion: int | None = None,
        zoom_in: int | None = None,
        viewpoint: int | None = None,
        confidence: float = 1.0,
    ) -> int:
        if len(bbox) != 4:
            raise ValueError("bbox must contain [x1, y1, x2, y2]")

        x1, y1, x2, y2 = [float(value) for value in bbox]
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)

        self.cursor.execute(
            """
            INSERT OR REPLACE INTO annotations (
                image_id, label_id, item_key, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                area, style, scale, occlusion, zoom_in, viewpoint,
                segmentation_json, landmarks_json, confidence
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                image_id,
                label_id,
                item_key,
                x1,
                y1,
                x2,
                y2,
                area,
                style,
                scale,
                occlusion,
                zoom_in,
                viewpoint,
                json.dumps(segmentation or []),
                json.dumps(landmarks or []),
                confidence,
            ),
        )
        self.connection.commit()
        return int(self.cursor.lastrowid)

    def get_all_clothing_types(self) -> list[tuple[int, str, str | None]]:
        self.cursor.execute(
            "SELECT id, name, description FROM clothing_types ORDER BY id ASC"
        )
        rows = self.cursor.fetchall()
        return [(int(row["id"]), row["name"], row["description"]) for row in rows]

    def get_training_images(self) -> list[dict[str, Any]]:
        self.cursor.execute(
            """
            SELECT
                i.id,
                i.file_path,
                i.file_name,
                COALESCE(GROUP_CONCAT(DISTINCT ct.name), 'Unlabeled') AS labels,
                COUNT(DISTINCT a.id) AS annotation_count
            FROM images i
            LEFT JOIN image_labels il ON i.id = il.image_id
            LEFT JOIN annotations a ON i.id = a.image_id
            LEFT JOIN clothing_types ct
                ON ct.id = COALESCE(a.label_id, il.label_id)
            WHERE i.is_training = 1
            GROUP BY i.id
            ORDER BY i.upload_date DESC
            """
        )
        rows = self.cursor.fetchall()
        return [
            {
                "id": int(row["id"]),
                "file_path": row["file_path"],
                "file_name": row["file_name"],
                "labels": row["labels"],
                "annotation_count": int(row["annotation_count"]),
            }
            for row in rows
        ]

    def get_test_images(self) -> list[dict[str, Any]]:
        self.cursor.execute(
            """
            SELECT
                i.id,
                i.file_path,
                i.file_name,
                COALESCE(GROUP_CONCAT(DISTINCT ct.name), 'Unlabeled') AS labels,
                COUNT(DISTINCT a.id) AS annotation_count
            FROM images i
            LEFT JOIN image_labels il ON i.id = il.image_id
            LEFT JOIN annotations a ON i.id = a.image_id
            LEFT JOIN clothing_types ct
                ON ct.id = COALESCE(a.label_id, il.label_id)
            WHERE i.is_test = 1
            GROUP BY i.id
            ORDER BY i.upload_date DESC
            """
        )
        rows = self.cursor.fetchall()
        return [
            {
                "id": int(row["id"]),
                "file_path": row["file_path"],
                "file_name": row["file_name"],
                "labels": row["labels"],
                "annotation_count": int(row["annotation_count"]),
            }
            for row in rows
        ]

    def get_image_annotations(self, image_id: int) -> list[dict[str, Any]]:
        self.cursor.execute(
            """
            SELECT
                a.id,
                ct.name AS label,
                a.item_key,
                a.bbox_x1,
                a.bbox_y1,
                a.bbox_x2,
                a.bbox_y2,
                a.style,
                a.scale,
                a.occlusion,
                a.zoom_in,
                a.viewpoint,
                a.confidence
            FROM annotations a
            JOIN clothing_types ct ON ct.id = a.label_id
            WHERE a.image_id = ?
            ORDER BY a.id ASC
            """,
            (image_id,),
        )
        rows = self.cursor.fetchall()
        return [
            {
                "id": int(row["id"]),
                "label": row["label"],
                "item_key": row["item_key"],
                "bbox": [
                    float(row["bbox_x1"]),
                    float(row["bbox_y1"]),
                    float(row["bbox_x2"]),
                    float(row["bbox_y2"]),
                ],
                "style": row["style"],
                "scale": row["scale"],
                "occlusion": row["occlusion"],
                "zoom_in": row["zoom_in"],
                "viewpoint": row["viewpoint"],
                "confidence": float(row["confidence"]),
            }
            for row in rows
        ]

    def get_database_stats(self) -> dict[str, int]:
        self.cursor.execute("SELECT COUNT(*) FROM clothing_types")
        num_types = int(self.cursor.fetchone()[0])

        self.cursor.execute("SELECT COUNT(*) FROM images WHERE is_training = 1")
        num_training = int(self.cursor.fetchone()[0])

        self.cursor.execute("SELECT COUNT(*) FROM images WHERE is_test = 1")
        num_test = int(self.cursor.fetchone()[0])

        self.cursor.execute("SELECT COUNT(*) FROM image_labels")
        num_labels = int(self.cursor.fetchone()[0])

        self.cursor.execute("SELECT COUNT(*) FROM annotations")
        num_annotations = int(self.cursor.fetchone()[0])

        return {
            "clothing_types": num_types,
            "training_images": num_training,
            "test_images": num_test,
            "total_labels": num_labels,
            "total_annotations": num_annotations,
        }

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "ClothingDatabase":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
