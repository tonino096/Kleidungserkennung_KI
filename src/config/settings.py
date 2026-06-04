from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path

# PROJECT_ROOT dient als feste Basis für alle relativen Projektpfade,
# egal von welchem aktuellen Arbeitsverzeichnis das Projekt gestartet wird.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(path_value: str | Path) -> Path:
    # Relative Pfade aus .env, Streamlit oder CLI werden hier zentral
    # in absolute Projektpfade umgerechnet.
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def detect_deepfashion2_root() -> str:
    # Falls der Nutzer den Datensatzpfad explizit vorgibt, hat dieser Wert
    # immer Vorrang vor jeder automatischen Erkennung.
    explicit_root = os.getenv("DEEPFASHION2_ROOT")
    if explicit_root:
        return str(resolve_project_path(explicit_root))

    # Diese typischen Orte decken die häufigsten Projekt-Setups ab:
    # im Repository, neben dem Repository oder relativ zum Startordner.
    candidates = [
        PROJECT_ROOT / "data" / "deepfashion2",
        PROJECT_ROOT.parent / "DeepFashion2",
        Path.cwd() / "data" / "deepfashion2",
        Path.cwd().parent / "DeepFashion2",
    ]

    for candidate in candidates:
        if _looks_like_deepfashion2_root(candidate):
            return str(candidate)

    # Wenn nichts gefunden wird, bleibt ein sinnvoller Default stehen.
    # Die Oberfläche kann dann immer noch einen abweichenden Pfad annehmen.
    return str(PROJECT_ROOT / "data" / "deepfashion2")


def _looks_like_deepfashion2_root(path: Path) -> bool:
    # DeepFashion2 existiert je nach Download/Entpackung entweder direkt als
    # split/image + split/annos oder verschachtelt als split/split/image + annos.
    for split in ("train", "validation", "test"):
        direct_root = path / split
        nested_root = direct_root / split
        if (direct_root / "image").exists() and (direct_root / "annos").exists():
            return True
        if (nested_root / "image").exists() and (nested_root / "annos").exists():
            return True
    return False


def resolve_default_best_model_path() -> str:
    # Für das dritte Modell wird zuerst ein expliziter Pfad respektiert.
    # Ohne Vorgabe suchen wir nach einer vorhandenen Datei, damit die App
    # nicht an einem fehlenden Standardnamen hängen bleibt.
    explicit_path = os.getenv("YOLO_MODEL_3_PATH")
    if explicit_path:
        return str(resolve_project_path(explicit_path))

    # Die Reihenfolge ist bewusst priorisiert: erst ein später mögliches
    # finales best.pt, danach das vorhandene YOLO11-Modell und zuletzt das
    # YOLOv8-Modell als sicherer Rückfall.
    candidates = [
        resolve_project_path("models/best.pt"),
        resolve_project_path("models/bestv11n_15ep.pt"),
        resolve_project_path("models/bestv8n_15ep.pt"),
    ]

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    # Falls noch keine Datei vorhanden ist, bleibt ein stabiler Fallback.
    return str(candidates[1])


def build_model_registry() -> dict[str, str]:
    # Die drei Modellpfade können bei Bedarf per Umgebungsvariable überschrieben
    # werden, funktionieren aber auch mit den hier hinterlegten Standardnamen.
    # Die Schlüssel sind absichtlich schon die später sichtbaren UI-Namen,
    # damit alle Projektteile dieselben Bezeichnungen wiederverwenden können.
    return {
        "YOLO v8n 15 Epochen": str(resolve_project_path(os.getenv("YOLO_MODEL_1_PATH", "models/bestv8n_15ep.pt"))),
        "YOLO v11n 15 Epochen": str(resolve_project_path(os.getenv("YOLO_MODEL_2_PATH", "models/bestv11n_15ep.pt"))),
        "YOLO best": resolve_default_best_model_path(),
    }


@dataclass
class Settings:
    # Alle Laufzeitoptionen werden an einer Stelle gebündelt, damit CLI,
    # Streamlit-App und Hilfsskripte dieselben Vorgaben verwenden.
    detector_type: str = os.getenv("DETECTOR_TYPE", "mock")
    yolo_weights_path: str = str(resolve_project_path(os.getenv("YOLO_WEIGHTS_PATH", "models/bestv8n_15ep.pt")))
    deepfashion2_root: str = detect_deepfashion2_root()
    model_registry: dict[str, str] = field(default_factory=build_model_registry)
    active_yolo_model: str = os.getenv("ACTIVE_YOLO_MODEL", "YOLO v8n 15 Epochen")

    def get_available_model_names(self) -> list[str]:
        # Die Reihenfolge der Registry-Einträge bestimmt direkt die Reihenfolge
        # in den Auswahlfeldern der Oberfläche.
        return list(self.model_registry.keys())

    def get_yolo_weights_path(self, model_name: str | None = None) -> str:
        # Ist kein Modellname angegeben, wird das aktuell aktive Modell genutzt.
        chosen_model = model_name or self.active_yolo_model
        if chosen_model in self.model_registry:
            return self.model_registry[chosen_model]
        return self.yolo_weights_path
