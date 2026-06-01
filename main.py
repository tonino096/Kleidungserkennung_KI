from __future__ import annotations

import argparse

from src.config.settings import Settings
from src.training.database import ClothingDatabase
from src.training.deepfashion2_importer import DeepFashion2Importer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Project utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_db = subparsers.add_parser("init-db", help="Create the local SQLite schema")
    init_db.add_argument("--db-path", default=Settings().database_path)

    import_df2 = subparsers.add_parser(
        "import-deepfashion2",
        help="Import DeepFashion2 train/validation annotations into SQLite",
    )
    import_df2.add_argument("--db-path", default=Settings().database_path)
    import_df2.add_argument("--dataset-root", default=Settings().deepfashion2_root)
    import_df2.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation"],
        help="DeepFashion2 splits to import",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "init-db":
        with ClothingDatabase(args.db_path) as database:
            database.create_tables()
        print(f"Database initialized at {args.db_path}")
        return

    if args.command == "import-deepfashion2":
        with ClothingDatabase(args.db_path) as database:
            importer = DeepFashion2Importer(args.dataset_root, database)
            summary = importer.import_all(args.splits)
        print("DeepFashion2 import finished")
        for key, value in summary.items():
            print(f"{key}: {value}")
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
