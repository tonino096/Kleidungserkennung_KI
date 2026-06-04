from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from PIL import Image as PilImage
from PIL import ImageDraw, ImageFont, ImageOps
from PIL.Image import Image

from src.detector.base import Prediction

# Feste Farben pro Kategorie machen die Ausgaben leichter lesbar, weil
# dieselbe Kleidungsart in App, Matchtest und Report immer gleich aussieht.
_CATEGORY_COLORS: dict[str, tuple[int, int, int]] = {
    "short sleeve top": (231, 76, 60),
    "long sleeve top": (192, 57, 43),
    "short sleeve outwear": (230, 126, 34),
    "long sleeve outwear": (211, 84, 0),
    "vest": (241, 196, 15),
    "sling": (243, 156, 18),
    "shorts": (46, 204, 113),
    "trousers": (39, 174, 96),
    "skirt": (26, 188, 156),
    "short sleeve dress": (52, 152, 219),
    "long sleeve dress": (41, 128, 185),
    "vest dress": (155, 89, 182),
    "sling dress": (142, 68, 173),
}
_FALLBACK_PALETTE: list[tuple[int, int, int]] = [
    (231, 76, 60),
    (46, 204, 113),
    (52, 152, 219),
    (241, 196, 15),
    (155, 89, 182),
    (26, 188, 156),
    (230, 126, 34),
    (149, 165, 166),
]
# Alle Vorhersagebilder werden auf dieselbe Anzeige-Leinwand gebracht,
# damit Boxen und Texte zwischen verschiedenen Bildgrößen vergleichbar bleiben.
_ANNOTATION_CANVAS_SIZE = (1280, 960)
_ANNOTATION_FONT_SIZE = 18


def get_category_color(label: str) -> tuple[int, int, int]:
    # Bekannte Klassen bekommen ihre definierte Farbe, unbekannte Klassen
    # erhalten eine stabile Fallback-Farbe aus der Palette.
    normalized = label.strip().lower()
    if normalized in _CATEGORY_COLORS:
        return _CATEGORY_COLORS[normalized]

    index = int(hashlib.md5(normalized.encode("utf-8")).hexdigest(), 16) % len(_FALLBACK_PALETTE)
    return _FALLBACK_PALETTE[index]


def _shift_color(color: tuple[int, int, int], offset: int) -> tuple[int, int, int]:
    # Für Ground Truth und Prediction derselben Klasse wird bewusst nur eine
    # leicht hellere oder dunklere Variante derselben Basisfarbe verwendet.
    return tuple(max(0, min(255, channel + offset)) for channel in color)


def _format_image_for_annotation(image: Image) -> tuple[Image, float, float, int, int]:
    # Das Originalbild wird proportional in eine feste Anzeige-Leinwand
    # eingepasst und mittig platziert. Die Rückgabewerte enthalten zusätzlich
    # die Skalierungs- und Verschiebungswerte für die Boxen.
    source = image.convert("RGB")
    canvas_width, canvas_height = _ANNOTATION_CANVAS_SIZE
    scale = min(canvas_width / source.width, canvas_height / source.height)
    resized_width = max(1, int(round(source.width * scale)))
    resized_height = max(1, int(round(source.height * scale)))
    resized = source.resize((resized_width, resized_height), resample=PilImage.Resampling.LANCZOS)

    offset_x = (canvas_width - resized_width) // 2
    offset_y = (canvas_height - resized_height) // 2
    canvas = PilImage.new("RGB", _ANNOTATION_CANVAS_SIZE, (18, 18, 18))
    canvas.paste(resized, (offset_x, offset_y))
    return canvas, scale, scale, offset_x, offset_y


def _scale_predictions(
    predictions: list[Prediction],
    *,
    scale_x: float,
    scale_y: float,
    offset_x: int,
    offset_y: int,
) -> list[Prediction]:
    # Die Bounding Boxes werden von der Originalbildgröße auf die formatierte
    # Anzeige-Leinwand umgerechnet.
    scaled_predictions: list[Prediction] = []
    for prediction in predictions:
        x1, y1, x2, y2 = prediction["bbox"]
        scaled_predictions.append(
            {
                "label": prediction["label"],
                "confidence": prediction["confidence"],
                "bbox": [
                    int(round(x1 * scale_x)) + offset_x,
                    int(round(y1 * scale_y)) + offset_y,
                    int(round(x2 * scale_x)) + offset_x,
                    int(round(y2 * scale_y)) + offset_y,
                ],
            }
        )
    return scaled_predictions


def _load_annotation_font() -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    # Es werden zwei gängige Fonts versucht; falls beide fehlen, wird auf
    # den PIL-Standardfont zurückgefallen.
    for font_name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(font_name, _ANNOTATION_FONT_SIZE)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_box_overlay(
    draw: ImageDraw.ImageDraw,
    prediction: Prediction,
    *,
    label_prefix: str,
    color_offset: int,
    line_width: int,
    text_offset: int,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
) -> None:
    # Diese Hilfsfunktion zeichnet eine einzelne Box plus Textlabel.
    # Sie wird von allen Anzeigevarianten wiederverwendet.
    x1, y1, x2, y2 = prediction["bbox"]
    label = prediction["label"]
    confidence = prediction["confidence"]
    base_color = get_category_color(label)
    color = _shift_color(base_color, color_offset)
    text = f"{label_prefix}{label} ({confidence:.2f})"

    draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
    draw.text((x1, max(0, y1 + text_offset)), text, fill=color, font=font)


def draw_predictions(image: Image, predictions: list[Prediction]) -> Image:
    # Standardansicht für normale Vorhersagen ohne Ground Truth.
    output, scale_x, scale_y, offset_x, offset_y = _format_image_for_annotation(image)
    scaled_predictions = _scale_predictions(
        predictions,
        scale_x=scale_x,
        scale_y=scale_y,
        offset_x=offset_x,
        offset_y=offset_y,
    )
    draw = ImageDraw.Draw(output)
    font = _load_annotation_font()

    for pred in scaled_predictions:
        draw_box_overlay(
            draw,
            pred,
            label_prefix="",
            color_offset=0,
            line_width=6,
            text_offset=-22,
            font=font,
        )

    return output


def draw_match_comparison(
    image: Image,
    ground_truth: list[Prediction],
    predictions: list[Prediction],
) -> Image:
    # Für Matchtests werden echte Boxen und Modellvorhersagen gemeinsam
    # auf derselben Bildleinwand visualisiert.
    output, scale_x, scale_y, offset_x, offset_y = _format_image_for_annotation(image)
    scaled_ground_truth = _scale_predictions(
        ground_truth,
        scale_x=scale_x,
        scale_y=scale_y,
        offset_x=offset_x,
        offset_y=offset_y,
    )
    scaled_predictions = _scale_predictions(
        predictions,
        scale_x=scale_x,
        scale_y=scale_y,
        offset_x=offset_x,
        offset_y=offset_y,
    )
    draw = ImageDraw.Draw(output)
    font = _load_annotation_font()

    for target in scaled_ground_truth:
        draw_box_overlay(
            draw,
            target,
            label_prefix="GT: ",
            color_offset=-20,
            line_width=7,
            text_offset=-24,
            font=font,
        )

    for prediction in scaled_predictions:
        draw_box_overlay(
            draw,
            prediction,
            label_prefix="P: ",
            color_offset=30,
            line_width=5,
            text_offset=6,
            font=font,
        )

    return output


def draw_batch_report(
    totals: dict[str, Any],
    category_rows: list[dict[str, Any]],
    sample_image_paths: list[Path],
) -> Image:
    # Der Batch-Report wird als eigenständiges Bild erzeugt, damit er einfach
    # geteilt, eingebettet oder zusätzlich als Screenshot abgegeben werden kann.
    width = 1600
    top_height = 250
    row_height = 38
    gallery_height = 360 if sample_image_paths else 0
    height = top_height + 90 + len(category_rows) * row_height + gallery_height + 80

    background = (247, 243, 236)
    panel = (255, 252, 248)
    text = (45, 45, 45)
    muted = (108, 108, 108)
    accent = (36, 80, 121)

    report = PilImage.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(report)

    draw.text((50, 28), "DeepFashion2 Matchtest Report", fill=text)
    draw.text((50, 58), "Ground Truth und YOLO-Vorhersagen im direkten Vergleich", fill=muted)

    stat_cards = [
        ("Bilder", str(totals["images_processed"])),
        ("GT Boxen", str(totals["ground_truth_count"])),
        ("Vorhersagen", str(totals["prediction_count"])),
        ("Matches", str(totals["match_count"])),
        ("Recall", _format_percentage(totals["recall"])),
        ("Precision", _format_percentage(totals["precision"])),
    ]
    card_width = 230
    card_height = 92
    card_top = 100
    card_left = 50
    card_gap = 18

    for index, (title, value) in enumerate(stat_cards):
        # Die Kennzahlen werden in wiederholbaren "Karten" dargestellt, damit
        # die wichtigsten Gesamtwerte sofort erkennbar sind.
        x = card_left + index * (card_width + card_gap)
        draw.rounded_rectangle(
            [x, card_top, x + card_width, card_top + card_height],
            radius=16,
            fill=panel,
            outline=(226, 216, 202),
            width=2,
        )
        draw.text((x + 18, card_top + 18), title, fill=muted)
        draw.text((x + 18, card_top + 48), value, fill=accent)

    legend_top = 210
    draw.text((50, legend_top), "Legende:", fill=text)
    draw.text((145, legend_top), "GT = Ground Truth, P = Prediction", fill=muted)

    section_top = 260
    draw.text((50, section_top), "Trefferquote pro Kategorie", fill=text)
    draw.text((50, section_top + 26), "Balken zeigen Match-Anteil bezogen auf die echten Boxen.", fill=muted)

    bar_left = 430
    bar_width = 520
    row_top = section_top + 70

    for index, row in enumerate(category_rows):
        # Pro Kategorie wird ein Balken gezeichnet, dessen Länge direkt die
        # Recall-Qualität der Kategorie repräsentiert.
        y = row_top + index * row_height
        category = row["label"]
        gt_count = row["ground_truth"]
        matched = row["matched"]
        recall = row["recall"]
        color = get_category_color(category)

        draw.text((50, y), category, fill=text)
        draw.text((300, y), f"{matched}/{gt_count}", fill=muted)
        draw.rounded_rectangle(
            [bar_left, y + 4, bar_left + bar_width, y + 24],
            radius=10,
            fill=(231, 225, 216),
        )
        fill_width = int(bar_width * recall)
        if fill_width > 0:
            draw.rounded_rectangle(
                [bar_left, y + 4, bar_left + fill_width, y + 24],
                radius=10,
                fill=color,
            )
        draw.text((bar_left + bar_width + 22, y), _format_percentage(recall), fill=text)

    gallery_top = row_top + len(category_rows) * row_height + 50
    if sample_image_paths:
        draw.text((50, gallery_top), "Beispielbilder", fill=text)
        draw.text((50, gallery_top + 26), "Farben entsprechen den Kategorien; GT und Prediction teilen sich dieselbe Basisfarbe.", fill=muted)

        thumb_top = gallery_top + 60
        thumb_left = 50
        thumb_size = (350, 230)
        thumb_gap = 24

        for index, image_path in enumerate(sample_image_paths[:4]):
            # Die Galerie zeigt nur einige Beispielbilder, damit der Report
            # informativ bleibt, ohne zu groß zu werden.
            x = thumb_left + index * (thumb_size[0] + thumb_gap)
            with PilImage.open(image_path) as sample:
                thumb = ImageOps.fit(sample.convert("RGB"), thumb_size)
            report.paste(thumb, (x, thumb_top))
            draw.rounded_rectangle(
                [x, thumb_top, x + thumb_size[0], thumb_top + thumb_size[1]],
                radius=12,
                outline=(226, 216, 202),
                width=2,
            )
            draw.text((x, thumb_top + thumb_size[1] + 10), image_path.stem, fill=muted)

    return report


def _format_percentage(value: float) -> str:
    # Prozentformatierung wird an einer zentralen Stelle gebündelt, damit
    # alle Ausgaben dieselbe Darstellung verwenden.
    return f"{value * 100:.1f}%"
