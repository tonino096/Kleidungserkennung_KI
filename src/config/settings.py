from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass
class Settings:
    detector_type: str = os.getenv("DETECTOR_TYPE", "mock")
    yolo_weights_path: str = os.getenv("YOLO_WEIGHTS_PATH", "models/yolov8n-clothes.pt")
