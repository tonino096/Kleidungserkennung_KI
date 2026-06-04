from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Das Projekt wird beim direkten Aufruf von main.py einmalig in den Importpfad
# eingehängt, damit alle Unterpakete unter src gefunden werden.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import Settings
from src.testing.matchtest import DeepFashion2MatchTester
from src.training.deepfashion2_yolo import DeepFashion2ToYoloConverter


def build_parser() -> argparse.ArgumentParser:
    # Alle Projektwerkzeuge hängen unter einem gemeinsamen CLI-Einstieg.
    # So kann die Gruppe Konvertierung, Einzeltest und Batch-Test mit
    # demselben Startpunkt bedienen.
    parser = argparse.ArgumentParser(description="Project utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Dieser Befehl liest die originalen DeepFashion2-Annotationen ein und
    # schreibt daraus YOLO-Labeldateien, ohne die Bilder zu kopieren.
    convert_df2 = subparsers.add_parser(
        "convert-deepfashion2-to-yolo",
        help="Convert DeepFashion2 directly into a YOLO dataset",
    )
    convert_df2.add_argument("--dataset-root", default=Settings().deepfashion2_root)
    convert_df2.add_argument("--output-root", default="data/deepfashion2_yolo")
    convert_df2.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation"],
        help="DeepFashion2 splits to convert",
    )
    convert_df2.add_argument(
        "--limit-per-split",
        type=int,
        default=None,
        help="Optional limit for faster first runs",
    )

    # Der Einzel-Matchtest prüft ein konkretes Bild gegen die Ground Truth
    # und speichert ein Vergleichsbild sowie eine JSON-Zusammenfassung.
    matchtest = subparsers.add_parser(
        "matchtest-deepfashion2",
        help="Run a single DeepFashion2 matchtest and render color-coded bounding boxes",
    )
    matchtest.add_argument("--dataset-root", default=Settings().deepfashion2_root)
    matchtest.add_argument("--weights-path", default=Settings().yolo_weights_path)
    matchtest.add_argument("--model-name", default=Settings().active_yolo_model)
    matchtest.add_argument("--output-root", default="outputs/matchtests")
    matchtest.add_argument("--split", default="validation")
    matchtest.add_argument("--annotation-stem", default=None)
    matchtest.add_argument("--confidence-threshold", type=float, default=0.25)
    matchtest.add_argument("--iou-threshold", type=float, default=0.5)

    # Der Batch-Matchtest wiederholt denselben Ablauf für mehrere Bilder und
    # erzeugt zusätzlich Sammelberichte als PNG, HTML und JSON.
    batch_matchtest = subparsers.add_parser(
        "batch-matchtest-deepfashion2",
        help="Run a batch DeepFashion2 matchtest and generate visual summary reports",
    )
    batch_matchtest.add_argument("--dataset-root", default=Settings().deepfashion2_root)
    batch_matchtest.add_argument("--weights-path", default=Settings().yolo_weights_path)
    batch_matchtest.add_argument("--model-name", default=Settings().active_yolo_model)
    batch_matchtest.add_argument("--output-root", default="outputs/matchtests")
    batch_matchtest.add_argument("--split", default="validation")
    batch_matchtest.add_argument("--limit", type=int, default=12)
    batch_matchtest.add_argument("--confidence-threshold", type=float, default=0.25)
    batch_matchtest.add_argument("--iou-threshold", type=float, default=0.5)
    return parser


def main() -> None:
    # Der Parser liefert ein einziges Args-Objekt zurück, dessen Felder
    # je nach Unterbefehl unterschiedlich belegt sind.
    parser = build_parser()
    args = parser.parse_args()
    settings = Settings()

    if args.command == "convert-deepfashion2-to-yolo":
        converter = DeepFashion2ToYoloConverter(args.dataset_root, args.output_root)
        summary = converter.convert(args.splits, args.limit_per_split)
        print("DeepFashion2 YOLO conversion finished")
        for key, value in summary.items():
            print(f"{key}: {value}")
        print(f"YOLO dataset written to: {args.output_root}")
        return

    if args.command == "matchtest-deepfashion2":
        selected_weights_path = settings.get_yolo_weights_path(args.model_name) if getattr(args, "model_name", None) else args.weights_path
        tester = DeepFashion2MatchTester(
            dataset_root=args.dataset_root,
            weights_path=selected_weights_path,
            output_root=args.output_root,
        )
        summary = tester.run(
            split=args.split,
            annotation_stem=args.annotation_stem,
            confidence_threshold=args.confidence_threshold,
            iou_threshold=args.iou_threshold,
        )
        print("DeepFashion2 matchtest finished")
        for key, value in summary.items():
            print(f"{key}: {value}")
        return

    if args.command == "batch-matchtest-deepfashion2":
        selected_weights_path = settings.get_yolo_weights_path(args.model_name) if getattr(args, "model_name", None) else args.weights_path
        tester = DeepFashion2MatchTester(
            dataset_root=args.dataset_root,
            weights_path=selected_weights_path,
            output_root=args.output_root,
        )
        summary = tester.run_batch(
            split=args.split,
            limit=args.limit,
            confidence_threshold=args.confidence_threshold,
            iou_threshold=args.iou_threshold,
        )
        print("DeepFashion2 batch matchtest finished")
        for key, value in summary.items():
            print(f"{key}: {value}")
        return

    # Diese Zeile sollte im Normalfall nie erreicht werden, fängt aber
    # fehlerhafte oder später ergänzte Kommandonamen sauber ab.
    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
