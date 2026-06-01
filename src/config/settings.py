from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass
class Settings:
    detector_type: str = os.getenv("DETECTOR_TYPE", "mock")
    yolo_weights_path: str = os.getenv("YOLO_WEIGHTS_PATH", "models/yolov8n-clothes.pt")
    database_path: str = os.getenv("DATABASE_PATH", "data/clothing.db")
    deepfashion2_root: str = os.getenv("DEEPFASHION2_ROOT", "data/deepfashion2")
