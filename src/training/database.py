"""
Datenbank-Module für Kleidungs-Erkennungs-Training
Verwendet SQLite für einfache lokale Speicherung
"""

from __future__ import annotations

import sqlite3
import os
from pathlib import Path
from datetime import datetime


class ClothingDatabase:
    """
    Verwaltet die SQLite-Datenbank für Trainings- und Testdaten.
    
    Schema:
    - clothing_types: Kleidungsarten (z.B. T-Shirt, Hose, etc.)
    - images: Trainingsdaten (Bildpfade und Metadaten)
    - image_labels: Relationen zwischen Bildern und Labels (Many-to-Many)
    """

    def __init__(self, db_path: str = "data/clothing.db"):
        """
        Initialisiert die Datenbankverbindung.
        
        Args:
            db_path: Pfad zur SQLite-Datenbankdatei
        """
        self.db_path = db_path
        self._ensure_db_directory()
        self.connection = sqlite3.connect(self.db_path)
        self.cursor = self.connection.cursor()

    def _ensure_db_directory(self) -> None:
        """Erstellt das Verzeichnis für die DB, falls es nicht existiert."""
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

    def create_tables(self) -> None:
        """
        Erstellt alle notwendigen Tabellen im Blueprint.
        
        SQL-Statements:
        - CREATE TABLE clothing_types: Kleidungsarten (Label-Typen)
        - CREATE TABLE images: Bilder und ihre Metadaten
        - CREATE TABLE image_labels: Viele-zu-Viele Relation
        """
        # Tabelle 1: Kleidungstypen/Labels
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS clothing_types (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Tabelle 2: Trainingsbilder
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL UNIQUE,
                file_name TEXT NOT NULL,
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_training BOOLEAN DEFAULT 1,
                is_test BOOLEAN DEFAULT 0,
                source TEXT
            )
        """)

        # Tabelle 3: Viele-zu-Viele Relation (1 Bild kann mehrere Labels haben)
        self.cursor.execute("""
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
        """)

        self.connection.commit()
        print("✓ Tabellen erstellt!")

    def add_clothing_type(self, name: str, description: str = None) -> int:
        """
        Fügt einen neuen Kleidungstyp zur Datenbank hinzu.
        
        Args:
            name: Name des Kleidungstyps (z.B. "T-Shirt", "Jeans")
            description: Optionale Beschreibung
            
        Returns:
            ID des eingefügten Datensatzes
        """
        self.cursor.execute(
            "INSERT INTO clothing_types (name, description) VALUES (?, ?)",
            (name, description)
        )
        self.connection.commit()
        return self.cursor.lastrowid

    def add_image(self, file_path: str, is_training: bool = True, is_test: bool = False, source: str = None) -> int:
        """
        Fügt ein Trainingsbild zur Datenbank hinzu.
        
        Args:
            file_path: Vollständiger Pfad zum Bild
            is_training: True, wenn für Training verwendet
            is_test: True, wenn für Testen verwendet
            source: Optionale Quelle (z.B. "kaggle", "manual")
            
        Returns:
            ID des eingefügten Bildes
        """
        file_name = os.path.basename(file_path)
        self.cursor.execute(
            """INSERT INTO images (file_path, file_name, is_training, is_test, source) 
               VALUES (?, ?, ?, ?, ?)""",
            (file_path, file_name, is_training, is_test, source)
        )
        self.connection.commit()
        return self.cursor.lastrowid

    def add_label_to_image(self, image_id: int, label_id: int, confidence: float = 1.0) -> None:
        """
        Fügt ein Label zu einem Bild hinzu (Many-to-Many Relation).
        
        Args:
            image_id: ID des Bildes
            label_id: ID des Kleidungstyps
            confidence: Konfidenzwert (0.0 - 1.0), Standard: 1.0 (100%)
        """
        self.cursor.execute(
            """INSERT OR REPLACE INTO image_labels (image_id, label_id, confidence) 
               VALUES (?, ?, ?)""",
            (image_id, label_id, confidence)
        )
        self.connection.commit()

    def get_all_clothing_types(self) -> list[tuple]:
        """
        Holt alle Kleidungstypen aus der Datenbank.
        
        Returns:
            Liste von (id, name, description) Tupeln
        """
        self.cursor.execute("SELECT id, name, description FROM clothing_types")
        return self.cursor.fetchall()

    def get_training_images(self) -> list[dict]:
        """
        Holt alle Trainingsbilder mit ihren Labels.
        
        Returns:
            Liste von Wörterbüchern mit Bild- und Label-Informationen
        """
        self.cursor.execute("""
            SELECT i.id, i.file_path, i.file_name, GROUP_CONCAT(ct.name, ', ') as labels
            FROM images i
            LEFT JOIN image_labels il ON i.id = il.image_id
            LEFT JOIN clothing_types ct ON il.label_id = ct.id
            WHERE i.is_training = 1
            GROUP BY i.id
            ORDER BY i.upload_date DESC
        """)
        
        rows = self.cursor.fetchall()
        return [
            {
                "id": row[0],
                "file_path": row[1],
                "file_name": row[2],
                "labels": row[3] if row[3] else "Unlabeled"
            }
            for row in rows
        ]

    def get_test_images(self) -> list[dict]:
        """
        Holt alle Test-Bilder mit ihren Labels.
        
        Returns:
            Liste von Wörterbüchern mit Bild- und Label-Informationen
        """
        self.cursor.execute("""
            SELECT i.id, i.file_path, i.file_name, GROUP_CONCAT(ct.name, ', ') as labels
            FROM images i
            LEFT JOIN image_labels il ON i.id = il.image_id
            LEFT JOIN clothing_types ct ON il.label_id = ct.id
            WHERE i.is_test = 1
            GROUP BY i.id
            ORDER BY i.upload_date DESC
        """)
        
        rows = self.cursor.fetchall()
        return [
            {
                "id": row[0],
                "file_path": row[1],
                "file_name": row[2],
                "labels": row[3] if row[3] else "Unlabeled"
            }
            for row in rows
        ]

    def get_database_stats(self) -> dict:
        """
        Gibt statistische Informationen über die Datenbank aus.
        
        Returns:
            Wörterbuch mit Anzahl der Kleidungstypen, Bilder, etc.
        """
        self.cursor.execute("SELECT COUNT(*) FROM clothing_types")
        num_types = self.cursor.fetchone()[0]

        self.cursor.execute("SELECT COUNT(*) FROM images WHERE is_training = 1")
        num_training = self.cursor.fetchone()[0]

        self.cursor.execute("SELECT COUNT(*) FROM images WHERE is_test = 1")
        num_test = self.cursor.fetchone()[0]

        self.cursor.execute("SELECT COUNT(*) FROM image_labels")
        num_labels = self.cursor.fetchone()[0]

        return {
            "clothing_types": num_types,
            "training_images": num_training,
            "test_images": num_test,
            "total_labels": num_labels
        }

    def close(self) -> None:
        """Schließt die Datenbankverbindung."""
        self.connection.close()

    def __enter__(self):
        """Context Manager Eintritt."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context Manager Austritt."""
        self.close()


if __name__ == "__main__":
    # Beispiel: Datenbank initialisieren und testen
    db = ClothingDatabase(db_path="data/clothing.db")
    db.create_tables()

    # Kleidungstypen hinzufügen
    print("\n📝 Füge Kleidungstypen hinzu...")
    shirt_id = db.add_clothing_type("T-Shirt", "Kurzärmliges Oberteil")
    jeans_id = db.add_clothing_type("Jeans", "Lange Hose aus Denim")
    dress_id = db.add_clothing_type("Kleid", "Einteiliges Kleidungsstück")
    
    print("✓ Kleidungstypen hinzugefügt!")

    # Beispielbilder hinzufügen (später mit echten Daten ersetzen)
    print("\n📸 Füge Beispielbilder hinzu...")
    img1_id = db.add_image("data/images/shirt_001.jpg", is_training=True, source="manual")
    img2_id = db.add_image("data/images/jeans_001.jpg", is_training=True, source="manual")
    img3_id = db.add_image("data/images/dress_001.jpg", is_test=True, source="manual")
    
    # Labels zu Bildern hinzufügen
    print("\n🏷️  Füge Labels zu Bildern hinzu...")
    db.add_label_to_image(img1_id, shirt_id, confidence=0.95)
    db.add_label_to_image(img2_id, jeans_id, confidence=0.98)
    db.add_label_to_image(img3_id, dress_id, confidence=0.92)
    
    # Statistiken anzeigen
    print("\n📊 Datenbank-Statistiken:")
    stats = db.get_database_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Trainingsbilder anzeigen
    print("\n📚 Trainingsbilder:")
    for img in db.get_training_images():
        print(f"  {img['file_name']}: {img['labels']}")

    db.close()
    print("\n✓ Fertig!")
